"""Issue-time-aware ECMWF GRIB ingestion for the Slovenian archive.

The cfgrib import is intentionally lazy: the rest of Aurora and its unit tests
remain usable on machines which do not have ecCodes installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GRIB_VARIABLES = {
    "t2m": "weather_temperature_2m",
    "d2m": "weather_dew_point_2m",
    "u10": "_u10",
    "v10": "_v10",
    "tp": "weather_precipitation",
    "sp": "weather_surface_pressure",
    "ssrd": "weather_shortwave_radiation",
    "lcc": "weather_low_cloud_cover",
}
ACCUMULATED = {"tp", "ssrd"}
GRIB_CANONICAL_UNITS = {
    "temperature": "degC",
    "dew_point": "degC",
    "wind_speed": "m/s",
    "precipitation": "mm",
    "shortwave_radiation": "W/m2",
    "surface_pressure": "hPa",
    "low_cloud_cover": "%",
    "relative_humidity": "%",
}


def kelvin_to_celsius(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float) - 273.15


def pascal_to_hpa(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float) / 100.0


def metres_to_mm(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float) * 1000.0


def fraction_to_percent(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float) * 100.0


def accumulated_energy_to_wm2(
    values: Any, valid_times: Any, groups: Iterable[Any] | None = None
) -> np.ndarray:
    """Convert independently de-accumulated J/m2 values to interval-average W/m2."""
    group_values = None if groups is None else list(groups)
    energy = deaccumulate(values, group_values)
    times = pd.DatetimeIndex(pd.to_datetime(valid_times, utc=True))
    if groups is None:
        groups = np.zeros(len(times), dtype=int)
    else:
        groups = group_values
    group = np.asarray(list(groups))
    hours = np.ones(len(times), dtype=float)
    for key in pd.unique(group):
        indices = np.flatnonzero(group == key)
        for position, index in enumerate(indices):
            if position:
                hours[index] = max(
                    (times[index] - times[indices[position - 1]]).total_seconds() / 3600,
                    1e-9,
                )
    return energy / (hours * 3600.0)


def valid_time_from_issue_step(
    issue_timestamp: Any, forecast_step_hours: Any
) -> pd.Timestamp | pd.DatetimeIndex:
    """Convert a GRIB reference time and step (hours) to valid UTC time."""
    issue = pd.to_datetime(issue_timestamp, utc=True)
    return issue + pd.to_timedelta(forecast_step_hours, unit="h")


def derive_relative_humidity(temperature_c: Any, dew_point_c: Any) -> np.ndarray:
    """Derive RH (%) using the Magnus formula, preserving missing values."""
    t = np.asarray(temperature_c, dtype=float)
    d = np.asarray(dew_point_c, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rh = 100.0 * np.exp((17.625 * d / (243.04 + d)) - (17.625 * t / (243.04 + t)))
    return np.clip(rh, 0.0, 100.0)


def derive_wind(u10: Any, v10: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return wind speed (m/s) and meteorological direction (degrees)."""
    u, v = np.asarray(u10, dtype=float), np.asarray(v10, dtype=float)
    speed = np.hypot(u, v)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    direction = np.where(np.isfinite(speed), direction, np.nan)
    return speed, direction


def nearest_grid_point(
    latitudes: Any, longitudes: Any, latitude: float, longitude: float
) -> tuple[int, int]:
    """Return indices of the nearest latitude/longitude grid point."""
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    if lat.ndim == 1 and lon.ndim == 1:
        distance = (lat[:, None] - latitude) ** 2 + (lon[None, :] - longitude) ** 2
    else:
        distance = (lat - latitude) ** 2 + (lon - longitude) ** 2
    return tuple(int(x) for x in np.unravel_index(np.nanargmin(distance), distance.shape))


def deaccumulate(values: Any, groups: Iterable[Any] | None = None) -> np.ndarray:
    """De-accumulate values independently per issue/run, clamping resets."""
    value = np.asarray(values, dtype=float)
    if groups is None:
        groups = np.zeros(len(value), dtype=int)
    group = np.asarray(list(groups))
    result = np.full(value.shape, np.nan, dtype=float)
    for key in pd.unique(group):
        indices = np.flatnonzero(group == key)
        previous = np.nan
        for index in indices:
            current = value[index]
            if np.isfinite(current):
                result[index] = (
                    max(0.0, current - previous) if np.isfinite(previous) else max(0.0, current)
                )
                previous = current
    return result


def select_latest_eligible_issue(candidates: pd.DataFrame, targets: Any) -> pd.DataFrame:
    """Select latest issue satisfying issue < target <= issue + horizon."""
    required = {"issue_timestamp_utc", "forecast_horizon_hours"}
    missing = required - set(candidates)
    if missing:
        raise ValueError(f"Missing issue-selection columns: {sorted(missing)}")
    target_index = pd.DatetimeIndex(pd.to_datetime(targets, utc=True))
    rows = []
    for target in target_index:
        eligible = candidates[
            (pd.to_datetime(candidates["issue_timestamp_utc"], utc=True) < target)
            & (
                target
                <= pd.to_datetime(candidates["issue_timestamp_utc"], utc=True)
                + pd.to_timedelta(candidates["forecast_horizon_hours"], unit="h")
            )
        ]
        if not eligible.empty:
            rows.append(
                eligible.assign(target_timestamp_utc=target)
                .sort_values("issue_timestamp_utc")
                .iloc[-1]
            )
    return (
        pd.DataFrame(rows).reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=candidates.columns)
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_grib_metadata(dataset: Any, variable: str | None = None) -> pd.DataFrame:
    """Flatten one cfgrib/xarray dataset, retaining issue, step and valid time."""
    name = variable or next(iter(dataset.data_vars))
    frame = dataset[[name]].to_dataframe().reset_index()
    attrs = getattr(dataset, "attrs", {})
    issue = attrs.get("GRIB_refTime", attrs.get("forecast_reference_time", attrs.get("time")))
    if issue is None and "time" in frame:
        # ECMWF forecast archives commonly expose the reference time as the
        # per-row xarray ``time`` coordinate rather than a dataset attribute.
        issue = frame["time"]
    if "valid_time" not in frame:
        step = frame.get("step", 0)
        frame["valid_time"] = valid_time_from_issue_step(issue, step)
    frame["issue_timestamp_utc"] = pd.to_datetime(issue, utc=True)
    step = pd.to_timedelta(frame["step"] if "step" in frame else 0)
    if isinstance(step, pd.Series):
        frame["forecast_step_hours"] = step.dt.total_seconds() / 3600
    else:
        frame["forecast_step_hours"] = step.total_seconds() / 3600
    frame["valid_time_utc"] = pd.to_datetime(frame["valid_time"], utc=True)
    frame["value"] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def _require_cfgrib() -> Any:
    try:
        import cfgrib  # type: ignore
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "GRIB preparation requires the [nwp] extra (cfgrib and ecCodes)."
        ) from exc
    return cfgrib


def discover_grib_files(root: str | Path, variable: str) -> list[Path]:
    base = Path(root)
    return sorted((base / f"NWP_data_{variable}").glob(f"{variable}_*.grib")) + sorted(
        (base / f"NWP_data_{variable}_scda").glob(f"{variable}_*.grib")
    )


def _archive_chunk_key(path: Path) -> tuple[str, str]:
    """Return the month/stream processing key encoded by an archive path."""
    match = re.search(r"(\d{4})-(\d{2})", path.name)
    month = match.group(0) if match else "unknown"
    stream = "scda" if path.parent.name.endswith("_scda") else "oper"
    return month, stream


def _point_records(dataset: Any, variable: str, sites: pd.DataFrame) -> pd.DataFrame:
    """Extract one small xarray slice per unique nearest grid point.

    In particular, this never calls ``to_dataframe`` on the spatial dataset.
    The resulting frame contains only the forecast dimensions and the points
    needed by the requested sites.
    """
    if variable not in dataset.data_vars:
        return pd.DataFrame()
    lat_name = next((x for x in ("latitude", "lat") if x in dataset.coords), None)
    lon_name = next((x for x in ("longitude", "lon") if x in dataset.coords), None)
    if lat_name is None or lon_name is None:
        return pd.DataFrame()
    lat_coord, lon_coord = dataset[lat_name], dataset[lon_name]
    lat_dims = tuple(lat_coord.dims)
    lon_dims = tuple(lon_coord.dims)
    paired_point_dim = lat_dims == lon_dims and len(lat_dims) == 1
    point_dims = tuple(dict.fromkeys(lat_dims + lon_dims))
    if not paired_point_dim and len(point_dims) != 2:
        return pd.DataFrame()
    lat_values, lon_values = lat_coord.values, lon_coord.values
    points: dict[tuple[int, int], list[str]] = {}
    for site in sites.itertuples(index=False):
        if paired_point_dim:
            distance = (lat_values - float(site.latitude)) ** 2 + (
                lon_values - float(site.longitude)
            ) ** 2
            nearest = (int(np.nanargmin(distance)), -1)
        else:
            nearest = nearest_grid_point(
                lat_values, lon_values, float(site.latitude), float(site.longitude)
            )
        points.setdefault(nearest, []).append(str(site.site_id))

    frames: list[pd.DataFrame] = []
    attrs = getattr(dataset, "attrs", {})
    for (first, second), site_ids in points.items():
        # For regular grids nearest_grid_point returns (lat, lon); for a
        # curvilinear grid the coordinate dimensions still define the order.
        indexer = (
            {point_dims[0]: first}
            if paired_point_dim
            else {point_dims[0]: first, point_dims[1]: second}
        )
        selected = dataset[[variable]].isel(indexer)
        frame = selected[[variable]].to_dataframe().reset_index()
        if frame.empty:
            continue
        issue = attrs.get("GRIB_refTime", attrs.get("forecast_reference_time", attrs.get("time")))
        if issue is None:
            issue = frame["time"] if "time" in frame else pd.Timestamp("NaT", tz="UTC")
        step = frame["step"] if "step" in frame else 0
        if pd.api.types.is_numeric_dtype(step):
            step_delta = pd.to_timedelta(step, unit="h", errors="coerce")
        else:
            step_delta = pd.to_timedelta(step, errors="coerce")
        frame["issue_timestamp_utc"] = pd.to_datetime(issue, utc=True)
        frame["forecast_step_hours"] = (
            step_delta.dt.total_seconds() / 3600
            if isinstance(step_delta, pd.Series)
            else step_delta.total_seconds() / 3600
        )
        if "valid_time" in frame:
            frame["valid_time_utc"] = pd.to_datetime(frame["valid_time"], utc=True)
        else:
            frame["valid_time_utc"] = valid_time_from_issue_step(
                frame["issue_timestamp_utc"], step_delta
            )
        frame["value"] = pd.to_numeric(frame[variable], errors="coerce")
        frame = frame[["issue_timestamp_utc", "forecast_step_hours", "valid_time_utc", "value"]]
        for site_id in site_ids:
            site_frame = frame.copy()
            site_frame["site_id"] = site_id
            site_frame["variable"] = variable
            frames.append(site_frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _process_grib_chunk(
    records: list[pd.DataFrame],
    target_index: pd.DatetimeIndex,
    accumulator: pd.DataFrame,
    availability_delay_hours: float = 6.0,
    conservative_cutoff_hours: float = 2.0,
) -> None:
    """Merge one month/stream into the target-sized accumulator in place."""
    if not records:
        return
    values = pd.concat(records, ignore_index=True)
    keys = ["site_id", "issue_timestamp_utc", "forecast_step_hours", "valid_time_utc"]
    values = values.sort_values(["site_id", "issue_timestamp_utc", "valid_time_utc"])
    wide = values.pivot_table(
        index=keys, columns="variable", values="value", aggfunc="first"
    ).reset_index()
    for variable in GRIB_VARIABLES:
        if variable not in wide:
            wide[variable] = np.nan
    groups = wide["site_id"].astype(str) + "|" + wide["issue_timestamp_utc"].astype(str)
    for variable in (ACCUMULATED - {"ssrd"}):
        wide[variable] = wide.groupby(
            ["site_id", "issue_timestamp_utc"], sort=False, group_keys=False
        )[variable].transform(lambda series: deaccumulate(series.to_numpy()))
    wide["t2m"] = kelvin_to_celsius(wide.t2m)
    wide["d2m"] = kelvin_to_celsius(wide.d2m)
    wide["tp"] = metres_to_mm(wide.tp)
    wide["sp"] = pascal_to_hpa(wide.sp)
    wide["ssrd"] = accumulated_energy_to_wm2(wide.ssrd, wide.valid_time_utc, groups)
    wide["lcc"] = fraction_to_percent(wide.lcc)
    wide["weather_temperature_2m"] = wide.t2m
    wide["weather_dew_point_2m"] = wide.d2m
    wide["weather_precipitation"] = wide.tp
    wide["weather_surface_pressure"] = wide.sp
    wide["weather_shortwave_radiation"] = wide.ssrd
    wide["weather_low_cloud_cover"] = wide.lcc
    wide["weather_relative_humidity_2m"] = derive_relative_humidity(wide.t2m, wide.d2m)
    wide["weather_wind_speed_10m"], wide["weather_wind_direction_10m"] = derive_wind(
        wide.u10, wide.v10
    )
    weather_columns = [x for x in GRIB_VARIABLES.values() if not x.startswith("_")]
    derived_columns = weather_columns + [
        "weather_relative_humidity_2m",
        "weather_wind_speed_10m",
        "weather_wind_direction_10m",
    ]
    candidates: list[pd.DataFrame] = []
    for (site_id, issue), group in wide.groupby(["site_id", "issue_timestamp_utc"], sort=False):
        issue = pd.Timestamp(issue)
        eligible = target_index[
            (target_index >= issue + pd.Timedelta(hours=availability_delay_hours))
            & (target_index <= issue + pd.Timedelta(hours=12))
        ]
        if len(eligible) == 0:
            continue
        source = group.set_index("valid_time_utc").sort_index()
        source = source[derived_columns].reindex(source.index.union(eligible)).sort_index()
        source[derived_columns] = source[derived_columns].interpolate(
            method="time", limit_area="inside"
        )
        candidate = source.reindex(eligible)
        candidate = candidate[derived_columns].copy()
        candidate["site_id"] = site_id
        candidate["target_timestamp_utc"] = candidate.index
        candidate["_issue"] = issue
        candidates.append(candidate.reset_index(drop=True))
    if not candidates:
        return
    updates = pd.concat(candidates, ignore_index=True)
    update_index = pd.MultiIndex.from_frame(
        updates[["site_id", "target_timestamp_utc"]]
    )
    updates = updates.loc[~update_index.duplicated(keep="last")].copy()
    update_index = pd.MultiIndex.from_frame(
        updates[["site_id", "target_timestamp_utc"]]
    )
    current = accumulator["_issue"].reindex(update_index)
    newer = current.isna().to_numpy() | (
        updates["_issue"].to_numpy() > current.to_numpy()
    )
    if not newer.any():
        return
    selected = updates.loc[newer]
    selected_index = pd.MultiIndex.from_frame(
        selected[["site_id", "target_timestamp_utc"]]
    )
    accumulator.loc[selected_index, "_issue"] = selected["_issue"].to_numpy()
    accumulator.loc[selected_index, derived_columns] = selected[derived_columns].to_numpy()


def _request_signature(sites: pd.DataFrame, target_index: pd.DatetimeIndex) -> str:
    payload = {
        "sites": sites[["site_id", "latitude", "longitude"]]
        .astype({"site_id": str})
        .sort_values("site_id")
        .to_dict("records"),
        "targets": [timestamp.isoformat() for timestamp in target_index],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _save_grib_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        pd.to_pickle(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_grib_checkpoint(
    path: Path, signature: str
) -> tuple[pd.DataFrame | None, set[tuple[str, str]]]:
    if not path.is_file():
        return None, set()
    try:
        state = pd.read_pickle(path)
    except (EOFError, OSError, ValueError):
        return None, set()
    if state.get("signature") != signature:
        return None, set()
    completed = {tuple(item) for item in state.get("completed", [])}
    return state.get("accumulator"), completed


def read_grib_weather_archive(
    root: str | Path,
    sites: pd.DataFrame,
    targets: Any,
    checkpoint_root: str | Path | None = None,
    availability_delay_hours: float = 6.0,
    conservative_cutoff_hours: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the archive incrementally, retaining only requested points and targets.

    Completed month/stream chunks can be resumed from ``checkpoint_root``.
    The checkpoint contains only the target-sized accumulator, never GRIB data.
    """
    _require_cfgrib()
    import cfgrib  # type: ignore

    target_index = pd.DatetimeIndex(pd.to_datetime(targets, utc=True)).sort_values().unique()
    if len(target_index) == 0 or sites.empty:
        return pd.DataFrame(), {"source_files": [], "source_hashes": {}}
    first_month = (target_index.min() - pd.Timedelta(hours=12)).strftime("%Y-%m")
    last_month = target_index.max().strftime("%Y-%m")
    source_files = []
    chunks: dict[tuple[str, str], list[tuple[str, Path]]] = {}
    for variable in GRIB_VARIABLES:
        for path in discover_grib_files(root, variable):
            month, stream = _archive_chunk_key(path)
            if month != "unknown" and not first_month <= month <= last_month:
                continue
            source_files.append(path)
            chunks.setdefault((month, stream), []).append((variable, path))

    output_columns = [x for x in GRIB_VARIABLES.values() if not x.startswith("_")]
    derived_columns = output_columns + [
        "weather_relative_humidity_2m", "weather_wind_speed_10m", "weather_wind_direction_10m"
    ]
    index = pd.MultiIndex.from_product(
        [sites["site_id"].astype(str), target_index], names=["site_id", "timestamp_utc"]
    )
    accumulator = pd.DataFrame(index=index, columns=derived_columns, dtype=float)
    accumulator["_issue"] = pd.Series(pd.NaT, index=index, dtype="datetime64[ns, UTC]")
    signature = _request_signature(sites, target_index)
    checkpoint_path = (
        Path(checkpoint_root) / "state.pkl" if checkpoint_root is not None else None
    )
    completed: set[tuple[str, str]] = set()
    if checkpoint_path is not None:
        saved_accumulator, completed = _load_grib_checkpoint(checkpoint_path, signature)
        if saved_accumulator is not None and saved_accumulator.index.equals(accumulator.index):
            accumulator = saved_accumulator
    for (month, stream), files in sorted(chunks.items()):
        chunk_key = (month, stream)
        if chunk_key in completed:
            print(f"[GRIB] resume {month} {stream} (checkpointed)", flush=True)
            continue
        chunk_started = time.perf_counter()
        print(f"[GRIB] processing {month} {stream}", flush=True)
        records: list[pd.DataFrame] = []
        for variable, path in files:
            file_started = time.perf_counter()
            before = len(records)
            print(f"[GRIB]   opening {path.name}", flush=True)
            for dataset in cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""}):
                try:
                    points = _point_records(dataset, variable, sites)
                    if not points.empty:
                        records.append(points)
                finally:
                    close = getattr(dataset, "close", None)
                    if close is not None:
                        close()
            print(
                f"[GRIB]   finished {path.name}: {len(records) - before} point frames "
                f"in {time.perf_counter() - file_started:.1f}s",
                flush=True,
            )
        _process_grib_chunk(
            records,
            target_index,
            accumulator,
            availability_delay_hours,
            conservative_cutoff_hours,
        )
        completed.add(chunk_key)
        if checkpoint_path is not None:
            _save_grib_checkpoint(
                checkpoint_path,
                {
                    "signature": signature,
                    "completed": sorted(completed),
                    "accumulator": accumulator,
                },
            )
        elapsed = time.perf_counter() - chunk_started
        print(
            f"[GRIB] completed {month} {stream}: {len(records)} point frames in {elapsed:.1f}s",
            flush=True,
        )
        del records

    result = accumulator.reset_index()
    if not result.empty:
        result = result.rename(columns={"_issue": "weather_forecast_reference_utc"})
        # lcc is not total cloud cover. Keep the established total-cloud field
        # explicitly masked so downstream code cannot mistake the two.
        result["weather_cloud_cover"] = np.nan
        result["weather_lead_hours"] = (
            result.timestamp_utc - result.weather_forecast_reference_utc
        ).dt.total_seconds() / 3600
        result["weather_issue_timestamp_available"] = (
            result["weather_forecast_reference_utc"].notna().astype(float)
        )
        physical_columns = output_columns + [
            "weather_relative_humidity_2m",
            "weather_wind_speed_10m",
            "weather_wind_direction_10m",
        ]
        result["weather_available"] = (
            result[
                physical_columns
            ]
            .notna()
            .any(axis=1)
            .astype(float)
        )
        for column in physical_columns:
            result[f"{column}_available"] = result[column].notna().astype(float)
        result["weather_provider"], result["weather_model"] = "ecmwf-grib", "ecmwf-ifs"
    request_files = list(Path(root).glob("NWP_data_*/*.req"))
    all_files = sorted(set(source_files + request_files))
    missing_counts = (
        {
            column: int(result[column].isna().sum()) if column in result else len(index)
            for column in output_columns
        }
        if not result.empty
        else {column: len(index) for column in output_columns}
    )
    manifest = {
        "source_files": [str(x) for x in all_files],
        "source_hashes": {str(x): sha256_file(x) for x in all_files},
        "variable_mappings": GRIB_VARIABLES,
        "canonical_units": GRIB_CANONICAL_UNITS,
        "streams": ["oper", "scda"],
        "issue_cycles_utc": ["00:00", "06:00", "12:00", "18:00"],
        "forecast_horizon_hours": 12,
        "availability_delay_hours": availability_delay_hours,
        "conservative_row_cutoff_hours": conservative_cutoff_hours,
        "deaccumulation": "per issue/reference time; negative resets clamped to zero",
        "total_cloud": "masked; lcc is exposed only as weather_low_cloud_cover",
        "missing_variable_counts": missing_counts,
        "grid_selection": "nearest latitude/longitude grid point per site",
        "issue_time_coverage": {
            "rows": int(result["weather_issue_timestamp_available"].sum())
            if not result.empty
            else 0,
            "fraction": float(result["weather_issue_timestamp_available"].mean())
            if not result.empty
            else 0.0,
        },
    }
    return result, manifest
