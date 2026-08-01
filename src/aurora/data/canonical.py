from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurora.data.hirvensalmi import hirvensalmi_records_to_frame, parse_hirvensalmi_workbook
from aurora.data.schema import CANONICAL_PV_COLUMNS, CANONICAL_PV_SCHEMA_VERSION
from aurora.features.solar import calculate_clear_sky_irradiance, calculate_solar_angles
from aurora.features.time import harmonic_features

INTERVAL_MINUTES = 15
INTERVAL_HOURS = INTERVAL_MINUTES / 60
HIRVENSALMI_CAPACITY_MW = 4.08


@dataclass(frozen=True)
class SiteSpec:
    site_id: str
    installed_capacity_mw: float
    latitude: float
    longitude: float
    timezone: str


HIRVENSALMI_SITE = SiteSpec(
    site_id="hirvensalmi",
    installed_capacity_mw=HIRVENSALMI_CAPACITY_MW,
    latitude=61.64,
    longitude=26.78,
    timezone="Europe/Helsinki",
)


def utc_grid_time_idx(timestamps: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Return absolute quarter-hour indices so gaps remain visible to TFT."""
    idx = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    elapsed = idx - pd.Timestamp("1970-01-01", tz="UTC")
    return np.asarray(elapsed // pd.Timedelta(minutes=INTERVAL_MINUTES), dtype=np.int64)


def capacity_factor(energy_mwh: pd.Series, capacity_mw: pd.Series | float) -> pd.Series:
    denominator = np.asarray(capacity_mw, dtype=float) * INTERVAL_HOURS
    if np.any(denominator <= 0):
        raise ValueError("Installed capacity must be positive.")
    return pd.Series(np.asarray(energy_mwh, dtype=float) / denominator, index=energy_mwh.index)


def canonicalize_site_frame(
    observations: pd.DataFrame,
    site: SiteSpec,
    source: str,
    exclusions: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reindex one site to its UTC grid while preserving missing observations."""
    required = {"timestamp_utc", "interval_energy_mwh"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Missing canonical input columns: {sorted(missing)}")

    raw = observations.copy()
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce")
    malformed = int(raw["timestamp_utc"].isna().sum()) + sum(
        item.get("reason") == "malformed_timestamp" for item in (exclusions or [])
    )
    raw = raw.dropna(subset=["timestamp_utc"])
    duplicate_count = int(raw["timestamp_utc"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"UTC conversion produced {duplicate_count} duplicate timestamps.")
    raw = raw.sort_values("timestamp_utc").set_index("timestamp_utc")
    if raw.empty:
        raise ValueError(f"No usable observations for site {site.site_id!r}.")

    grid = pd.date_range(raw.index.min(), raw.index.max(), freq="15min", tz="UTC")
    frame = raw.reindex(grid).rename_axis("timestamp_utc").reset_index()
    frame["site_id"] = site.site_id
    frame["installed_capacity_mw"] = site.installed_capacity_mw
    frame["latitude"] = site.latitude
    frame["longitude"] = site.longitude
    frame["timezone"] = site.timezone
    frame["source"] = frame.get("source", pd.Series(index=frame.index, dtype=object)).fillna(source)
    frame["source_sheet"] = frame.get("source_sheet", pd.Series(index=frame.index, dtype=object))
    frame["source_row"] = frame.get("source_row", pd.Series(index=frame.index, dtype="Int64"))
    frame["planned_mwh"] = pd.to_numeric(
        frame.get("planned_mwh", pd.Series(index=frame.index, dtype=float)), errors="coerce"
    )
    frame["interval_energy_mwh"] = pd.to_numeric(frame["interval_energy_mwh"], errors="coerce")
    frame.loc[frame["interval_energy_mwh"] < 0, "interval_energy_mwh"] = np.nan
    frame["observation_available"] = frame["interval_energy_mwh"].notna()
    frame["quality_status"] = np.where(frame["observation_available"], "ok", "missing")
    frame["capacity_factor"] = capacity_factor(
        frame["interval_energy_mwh"], frame["installed_capacity_mw"]
    )
    frame["time_idx"] = utc_grid_time_idx(frame["timestamp_utc"])
    frame = add_deterministic_features(frame)

    gap_count = int((~frame["observation_available"]).sum())
    cf = frame.loc[frame["observation_available"], "capacity_factor"]
    manifest = {
        "schema_version": CANONICAL_PV_SCHEMA_VERSION,
        "site": asdict(site),
        "source": source,
        "row_count": int(len(frame)),
        "observed_row_count": int(frame["observation_available"].sum()),
        "gap_count": gap_count,
        "duplicate_utc_timestamps": duplicate_count,
        "malformed_timestamp_count": malformed,
        "exclusions": exclusions or [],
        "timestamp_min_utc": frame["timestamp_utc"].min().isoformat(),
        "timestamp_max_utc": frame["timestamp_utc"].max().isoformat(),
        "capacity_checks": {
            "capacity_factor_min": float(cf.min()) if len(cf) else None,
            "capacity_factor_max": float(cf.max()) if len(cf) else None,
            "above_one_count": int((cf > 1).sum()),
            "below_zero_count": int((cf < 0).sum()),
        },
    }
    leading = list(CANONICAL_PV_COLUMNS)
    trailing = [column for column in frame.columns if column not in leading]
    return frame[leading + trailing], manifest


def add_deterministic_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    pieces: list[pd.DataFrame] = []
    for _, site_frame in out.groupby("site_id", sort=False):
        timestamps = pd.DatetimeIndex(site_frame["timestamp_utc"])
        local = timestamps.tz_convert(str(site_frame["timezone"].iloc[0]))
        harmonics = harmonic_features(local)
        harmonics.index = site_frame.index
        angles = calculate_solar_angles(
            timestamps,
            float(site_frame["latitude"].iloc[0]),
            float(site_frame["longitude"].iloc[0]),
        )
        angles.index = site_frame.index
        clear_sky = calculate_clear_sky_irradiance(
            timestamps,
            float(site_frame["latitude"].iloc[0]),
            float(site_frame["longitude"].iloc[0]),
        )
        features = pd.concat([harmonics, angles], axis=1)
        features["clear_sky_norm"] = np.asarray(clear_sky, dtype=float) / 1361.0
        pieces.append(features)
    features = pd.concat(pieces).sort_index()
    for column in features.columns:
        out[column] = features[column]
    return out


def prepare_hirvensalmi_dataset(
    workbook_path: Path,
    output_path: Path | None = None,
    site: SiteSpec = HIRVENSALMI_SITE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parsed = parse_hirvensalmi_workbook(
        workbook_path, site_id=site.site_id, source_timezone=site.timezone
    )
    raw = hirvensalmi_records_to_frame(parsed.records).rename(
        columns={"actual_mwh": "interval_energy_mwh", "quality_flag": "quality_status"}
    )
    frame, manifest = canonicalize_site_frame(
        raw,
        site=site,
        source=workbook_path.name,
        exclusions=list(parsed.exclusions),
    )
    manifest["source_hashes"] = {str(workbook_path): sha256_file(workbook_path)}
    if output_path is not None:
        write_processed_dataset(frame, manifest, output_path)
    return frame, manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_processed_dataset(
    frame: pd.DataFrame, manifest: dict[str, Any], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(output_path, index=False)
    except ImportError as exc:
        raise ImportError(
            "Parquet output requires pyarrow; install the project dependencies."
        ) from exc
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def read_processed_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame
