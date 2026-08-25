"""Open-Meteo historical forecast ingestion and issue-time-safe alignment.

The module deliberately keeps the HTTP boundary small.  Tests can provide a
``transport`` callable, while production uses the standard library and caches
the raw response before parsing it.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OPEN_METEO_HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_SINGLE_RUN_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
OPEN_METEO_MODEL = "ecmwf_ifs025"
HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "cloud_cover_low",
    "precipitation",
    "shortwave_radiation",
    "surface_pressure",
)
WEATHER_REALS = (
    "weather_temperature_2m",
    "weather_relative_humidity_2m",
    "weather_dew_point_2m",
    "weather_wind_speed_10m",
    "weather_wind_direction_10m",
    "weather_cloud_cover",
    "weather_low_cloud_cover",
    "weather_precipitation",
    "weather_shortwave_radiation",
    "weather_surface_pressure",
    "weather_available",
    "weather_issue_timestamp_available",
    "weather_lead_hours",
)
WEATHER_PHYSICAL_REALS = tuple(
    column for column in WEATHER_REALS if column.startswith("weather_")
    and column
    not in {"weather_available", "weather_issue_timestamp_available", "weather_lead_hours"}
)
WEATHER_MASKS = tuple(f"{column}_available" for column in WEATHER_PHYSICAL_REALS)
CANONICAL_WEATHER_UNITS = {
    "temperature_2m": "degC", "relative_humidity_2m": "%", "dew_point_2m": "degC",
    "wind_speed_10m": "m/s", "wind_direction_10m": "deg", "precipitation": "mm",
    "shortwave_radiation": "W/m2", "surface_pressure": "hPa",
    "cloud_cover_low": "%", "cloud_cover": "% (masked compatibility only)",
}
_COLUMN_MAP = {
    name: ("weather_low_cloud_cover" if name == "cloud_cover_low" else f"weather_{name}")
    for name in HOURLY_VARIABLES
}


@dataclass(frozen=True)
class OpenMeteoRequest:
    latitude: float
    longitude: float
    start_date: str
    end_date: str
    mode: str = "historical"
    model: str = OPEN_METEO_MODEL
    run_timestamp_utc: pd.Timestamp | None = None
    forecast_days: int | None = None

    def params(self) -> dict[str, str]:
        api_model = (
            "ecmwf_ifs"
            if self.run_timestamp_utc is not None and self.model == "ecmwf_ifs025"
            else self.model
        )
        params = {
            "latitude": f"{self.latitude:.6f}",
            "longitude": f"{self.longitude:.6f}",
            "hourly": ",".join(HOURLY_VARIABLES),
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
            "shortwave_radiation_unit": "watt_per_square_meter",
            "surface_pressure_unit": "hPa",
            "timezone": "UTC",
            "models": api_model,
        }
        if self.run_timestamp_utc is not None:
            params["run"] = (
                pd.Timestamp(self.run_timestamp_utc).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M")
            )
        else:
            params["start_date"] = self.start_date
            params["end_date"] = self.end_date
        if self.forecast_days is not None:
            params["forecast_days"] = str(self.forecast_days)
        return params


def _cache_key(request: OpenMeteoRequest) -> str:
    payload = json.dumps(request.params(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


class OpenMeteoClient:
    """Fetch and cache Open-Meteo responses, batching coordinates per request."""

    def __init__(
        self,
        cache_dir: str | Path = "data/cache/openmeteo",
        transport: Callable[[str], Any] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.transport = transport or self._download

    @staticmethod
    def _download(url: str) -> dict[str, Any]:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))

    def fetch(self, request: OpenMeteoRequest, *, use_cache: bool = True) -> dict[str, Any]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{_cache_key(request)}.json"
        if use_cache and path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        base = (
            OPEN_METEO_SINGLE_RUN_URL if request.mode == "single_run" else OPEN_METEO_HISTORICAL_URL
        )
        url = f"{base}?{urllib.parse.urlencode(request.params())}"
        payload = self.transport(url)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return payload

    @staticmethod
    def cache_hash(request: OpenMeteoRequest) -> str:
        """Return the stable cache/content-address key for manifest recording."""
        return _cache_key(request)

    def fetch_many(
        self, requests: list[OpenMeteoRequest], *, use_cache: bool = True
    ) -> list[dict[str, Any]]:
        """Fetch requests in one coordinate batch where date/model ranges match."""
        if not requests:
            return []
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        groups: dict[tuple[object, ...], list[OpenMeteoRequest]] = {}
        for request in requests:
            key = (
                request.start_date,
                request.end_date,
                request.mode,
                request.model,
                request.run_timestamp_utc,
                request.forecast_days,
            )
            groups.setdefault(key, []).append(request)
        result: list[dict[str, Any]] = []
        for group in groups.values():
            pending = []
            cached: dict[str, dict[str, Any]] = {}
            for request in group:
                path = self.cache_dir / f"{_cache_key(request)}.json"
                if use_cache and path.is_file():
                    cached[_cache_key(request)] = json.loads(path.read_text(encoding="utf-8"))
                else:
                    pending.append(request)
            if not pending:
                result.extend(cached[_cache_key(request)] for request in group)
                continue
            if len(pending) == 1:
                result.append(self.fetch(pending[0], use_cache=use_cache))
                continue
            first = pending[0]
            batch = OpenMeteoRequest(
                latitude=0,
                longitude=0,
                start_date=first.start_date,
                end_date=first.end_date,
                mode=first.mode,
                model=first.model,
                run_timestamp_utc=first.run_timestamp_utc,
                forecast_days=first.forecast_days,
            )
            params = batch.params()
            params["latitude"] = ",".join(f"{r.latitude:.6f}" for r in pending)
            params["longitude"] = ",".join(f"{r.longitude:.6f}" for r in pending)
            base = (
                OPEN_METEO_SINGLE_RUN_URL
                if first.mode == "single_run"
                else OPEN_METEO_HISTORICAL_URL
            )
            payload = self.transport(f"{base}?{urllib.parse.urlencode(params)}")
            values = payload if isinstance(payload, list) else [payload]
            if len(values) != len(pending):
                raise ValueError(
                    "Open-Meteo batch returned "
                    f"{len(values)} responses for {len(pending)} coordinates"
                )
            for request, value in zip(pending, values, strict=True):
                path = self.cache_dir / f"{_cache_key(request)}.json"
                path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
                cached[_cache_key(request)] = value
            result.extend(cached[_cache_key(request)] for request in group)
        return result


def parse_openmeteo_response(
    payload: dict[str, Any],
    *,
    site_id: str,
    provider: str = "open-meteo",
    model: str = OPEN_METEO_MODEL,
    issue_timestamp_utc: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Parse one API response into UTC hourly rows; absent variables stay NaN."""
    hourly = payload.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []), utc=True, errors="coerce")
    issue = (
        issue_timestamp_utc
        or payload.get("forecast_reference_utc")
        or payload.get("issue_timestamp_utc")
    )
    issue_ts = pd.to_datetime(issue, utc=True, errors="coerce") if issue else pd.NaT
    out = pd.DataFrame({"site_id": site_id, "timestamp_utc": times})
    for source, target in _COLUMN_MAP.items():
        values = hourly.get(source, [np.nan] * len(times))
        out[target] = (
            pd.to_numeric(pd.Series(values), errors="coerce").reindex(range(len(out))).to_numpy()
        )
    # Open-Meteo's standard response currently provides total cloud cover;
    # keep the explicit low-cloud field for the shared TFT schema without
    # confusing total cloud cover with low cloud cover.
    # Total cloud is deliberately masked: Slovenian GRIB only supplies low cloud.
    out["weather_cloud_cover"] = np.nan
    out["weather_provider"] = provider
    out["weather_model"] = payload.get("model") or model
    out["weather_forecast_reference_utc"] = issue_ts
    out["weather_issue_timestamp_available"] = float(pd.notna(issue_ts))
    if pd.notna(issue_ts):
        out["weather_lead_hours"] = (out["timestamp_utc"] - issue_ts).dt.total_seconds() / 3600
    else:
        out["weather_lead_hours"] = np.nan
    for column in WEATHER_PHYSICAL_REALS:
        out[f"{column}_available"] = out[column].notna().astype(float)
    out["weather_available"] = out[list(WEATHER_PHYSICAL_REALS)].notna().any(axis=1).astype(float)
    return out


def resample_weather_to_grid(
    weather: pd.DataFrame, timestamps: pd.Series | pd.DatetimeIndex
) -> pd.DataFrame:
    """Interpolate numeric hourly values onto a UTC 15-minute grid."""
    if weather.empty:
        return pd.DataFrame(
            {"timestamp_utc": pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))}
        )
    source = weather.copy()
    source["timestamp_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True)
    source = (
        source.sort_values("timestamp_utc")
        .drop_duplicates("timestamp_utc")
        .set_index("timestamp_utc")
    )
    grid = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    numeric = [c for c in (*WEATHER_REALS, *WEATHER_MASKS) if c in source]
    metadata = [
        c
        for c in ("weather_provider", "weather_model", "weather_forecast_reference_utc")
        if c in source
    ]
    result = source[numeric + metadata].reindex(source.index.union(grid)).sort_index()
    result[numeric] = result[numeric].interpolate(method="time", limit_area="inside")
    if metadata:
        result[metadata] = result[metadata].ffill()
    result = result.reindex(grid).reset_index(names="timestamp_utc")
    for column in (*WEATHER_MASKS, "weather_issue_timestamp_available", "weather_available"):
        if column in result:
            result[column] = result[column].fillna(0.0)
    return result


def attach_weather(
    frame: pd.DataFrame, weather: pd.DataFrame, *, already_aligned: bool = False
) -> pd.DataFrame:
    """Attach weather by site/timestamp without filling across issue timestamps."""
    required = {"site_id", "timestamp_utc"}
    if not required.issubset(frame) or not required.issubset(weather):
        raise ValueError("Both frame and weather require site_id and timestamp_utc")
    left = frame.copy()
    left["timestamp_utc"] = pd.to_datetime(left["timestamp_utc"], utc=True)
    rows = []
    for site_id, group in left.groupby("site_id", sort=False):
        site_weather = weather[weather["site_id"].astype(str) == str(site_id)]
        aligned = (
            site_weather.copy()
            if already_aligned
            else resample_weather_to_grid(site_weather, group["timestamp_utc"])
        )
        aligned["timestamp_utc"] = pd.to_datetime(aligned["timestamp_utc"], utc=True)
        aligned["site_id"] = site_id
        rows.append(group.merge(aligned, on=["site_id", "timestamp_utc"], how="left"))
    return pd.concat(rows, ignore_index=True) if rows else left


def fetch_single_run_weather_archive(
    client: OpenMeteoClient,
    *,
    site_id: str,
    latitude: float,
    longitude: float,
    target_timestamps: pd.Series | pd.DatetimeIndex,
    model: str = OPEN_METEO_MODEL,
    forecast_days: int = 10,
    run_hours: tuple[int, ...] = (0, 6, 12, 18),
    availability_delay_hours: float = 6.0,
    conservative_cutoff_hours: float = 2.0,
) -> pd.DataFrame:
    """Fetch exact model runs and select the latest eligible run per target."""
    targets = pd.DatetimeIndex(pd.to_datetime(target_timestamps, utc=True)).sort_values().unique()
    if len(targets) == 0:
        return pd.DataFrame()
    target_start, target_end = targets.min(), targets.max()
    first_run = (target_start - pd.Timedelta(days=forecast_days)).floor("D")
    last_run = target_end.floor("D")
    run_days = pd.date_range(first_run, last_run, freq="D", tz="UTC")
    runs = [day + pd.Timedelta(hours=hour) for day in run_days for hour in run_hours]
    requests = []
    for run in runs:
        run_end = run + pd.Timedelta(days=forecast_days)
        if run >= target_end or run_end <= target_start:
            continue
        request_start = max(target_start.normalize(), run.normalize())
        request_end = min(target_end.normalize(), run_end.normalize())
        requests.append(
            OpenMeteoRequest(
                latitude=latitude,
                longitude=longitude,
                start_date=request_start.date().isoformat(),
                end_date=request_end.date().isoformat(),
                mode="single_run",
                model=model,
                run_timestamp_utc=run,
                forecast_days=forecast_days,
            )
        )
    candidates: list[pd.DataFrame] = []
    for request in requests:
        try:
            payload = client.fetch(request)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 400 and "requested model run is not available" in detail.lower():
                continue
            raise
        parsed = parse_openmeteo_response(
            payload,
            site_id=site_id,
            model=model,
            issue_timestamp_utc=request.run_timestamp_utc,
        )
        aligned = resample_weather_to_grid(parsed, targets)
        aligned["site_id"] = site_id
        issue = request.run_timestamp_utc
        lead = (aligned["timestamp_utc"] - issue).dt.total_seconds() / 3600
        valid = (
            aligned["timestamp_utc"]
            >= issue + pd.Timedelta(hours=availability_delay_hours)
        ) & (lead <= forecast_days * 24)
        aligned.loc[:, "weather_lead_hours"] = lead
        candidates.append(aligned.loc[valid].copy())
    if not candidates:
        result = pd.DataFrame({"site_id": site_id, "timestamp_utc": targets})
        for column in (*WEATHER_REALS, *WEATHER_MASKS):
            if column not in result:
                result[column] = np.nan
        return result
    all_candidates = pd.concat(candidates, ignore_index=True)
    all_candidates = all_candidates.sort_values(
        ["timestamp_utc", "weather_forecast_reference_utc"]
    ).drop_duplicates("timestamp_utc", keep="last")
    result = pd.DataFrame({"site_id": site_id, "timestamp_utc": targets}).merge(
        all_candidates, on=["site_id", "timestamp_utc"], how="left"
    )
    result["weather_issue_timestamp_available"] = (
        result["weather_forecast_reference_utc"].notna().astype(float)
    )
    result["weather_available"] = (
        result[list(_COLUMN_MAP.values())].notna().any(axis=1).astype(float)
    )
    result.attrs["cache_hashes"] = [client.cache_hash(request) for request in requests]
    result.attrs["run_count"] = len(requests)
    return result


def validate_weather_forecast_windows(windows: pd.DataFrame) -> None:
    """Fail fast on issue-time leakage or non-quarter-hour decoder rows."""
    required = {"target_timestamp_utc", "weather_forecast_reference_utc", "weather_lead_hours"}
    missing = required - set(windows)
    if missing:
        raise ValueError(f"Missing weather window columns: {sorted(missing)}")
    target = pd.to_datetime(windows["target_timestamp_utc"], utc=True)
    issue = pd.to_datetime(windows["weather_forecast_reference_utc"], utc=True)
    known = issue.notna()
    if (target[known] <= issue[known]).any():
        raise ValueError("Weather forecast window contains target at or before issue timestamp")
    lead = pd.to_numeric(windows["weather_lead_hours"], errors="coerce")
    if (lead[known] < 0).any():
        raise ValueError("Weather forecast lead time must be non-negative")
    if (
        (target - pd.Timestamp("1970-01-01", tz="UTC")) % pd.Timedelta(minutes=15)
        != pd.Timedelta(0)
    ).any():
        raise ValueError("Weather target timestamps are not aligned to 15 minutes")


def validate_operational_weather_availability(
    windows: pd.DataFrame,
    *,
    availability_delay_hours: float = 6.0,
    conservative_cutoff_hours: float = 2.0,
) -> None:
    """Validate weather availability against target and PV issue timestamps."""
    validate_weather_forecast_windows(windows)
    issue = pd.to_datetime(windows["weather_forecast_reference_utc"], utc=True)
    target = pd.to_datetime(windows["target_timestamp_utc"], utc=True)
    known = issue.notna()
    if (
        target[known]
        < issue[known] + pd.Timedelta(hours=availability_delay_hours)
    ).any():
        raise ValueError("Weather row is used before the availability delay has elapsed")
    pv_column = "pv_forecast_issue_timestamp_utc"
    if pv_column in windows:
        pv_issue = pd.to_datetime(windows[pv_column], utc=True)
        comparable = known & pv_issue.notna()
        if (
            issue[comparable]
            > pv_issue[comparable] - pd.Timedelta(hours=conservative_cutoff_hours)
        ).any():
            raise ValueError("Weather forecast run was issued after the conservative PV cutoff")
