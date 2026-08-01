from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from aurora.data.canonical import (
    SiteSpec,
    canonicalize_site_frame,
    sha256_file,
    write_processed_dataset,
)

PV_COLUMN_ALIASES = ("site_id", "alt_id", "plant_id", "id")
LATITUDE_ALIASES = ("latitude", "lat")
LONGITUDE_ALIASES = ("longitude", "lon", "lng")
CAPACITY_MW_ALIASES = ("installed_capacity_mw", "capacity_mw")
CAPACITY_KW_ALIASES = (
    "installed_capacity_kw",
    "capacity_kw",
    "Pinst_kW",
    "kwp",
    "pv_max",
)


def prepare_slovenian_dataset(
    project_root: Path,
    metadata_path: Path,
    output_path: Path | None = None,
    pv_path: Path | None = None,
    timestamp_timezone: str = "UTC",
    site_timezone: str = "Europe/Ljubljana",
    value_kind: str = "power_kw",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Adapt the restored project's 233-column wide pickle and plant metadata."""
    pv_path = pv_path or project_root / "data" / "processed" / "data.p"
    power = _read_table(pv_path)
    metadata = _read_table(metadata_path)
    metadata = _normalize_metadata(metadata)
    if not isinstance(power.index, pd.DatetimeIndex):
        timestamp_column = _first_existing(power, ("timestamp", "datetime", "time"))
        power[timestamp_column] = pd.to_datetime(power[timestamp_column])
        power = power.set_index(timestamp_column)
    timestamps = pd.DatetimeIndex(power.index)
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize(timestamp_timezone)
    timestamps = timestamps.tz_convert("UTC")
    power.index = timestamps

    metadata["site_id"] = metadata["site_id"].astype(str)
    metadata = metadata.set_index("site_id", drop=False)
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    missing_sites: list[str] = []
    for raw_site_id in power.columns:
        site_id = str(raw_site_id)
        if site_id not in metadata.index:
            missing_sites.append(site_id)
            continue
        row = metadata.loc[site_id]
        if isinstance(row, pd.DataFrame):
            raise ValueError(f"Duplicate metadata rows for Slovenian site {site_id!r}.")
        values = pd.to_numeric(power[raw_site_id], errors="coerce")
        if value_kind == "power_kw":
            energy_mwh = values / 1000 * 0.25
        elif value_kind == "energy_mwh":
            energy_mwh = values
        else:
            raise ValueError("value_kind must be 'power_kw' or 'energy_mwh'.")
        raw = pd.DataFrame({"timestamp_utc": timestamps, "interval_energy_mwh": energy_mwh})
        site = SiteSpec(
            site_id=site_id,
            installed_capacity_mw=float(row["installed_capacity_mw"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            timezone=str(row.get("timezone", site_timezone)),
        )
        frame, manifest = canonicalize_site_frame(raw, site, source=str(pv_path))
        frames.append(frame)
        manifests.append(manifest)
    if missing_sites:
        raise ValueError(
            f"Metadata is missing {len(missing_sites)} PV sites, including {missing_sites[:10]}."
        )
    if not frames:
        raise ValueError("No Slovenian PV series matched the metadata.")
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["site_id", "timestamp_utc"]
    )
    manifest = {
        "schema_version": manifests[0]["schema_version"],
        "country": "Slovenia",
        "site_count": int(combined["site_id"].nunique()),
        "row_count": int(len(combined)),
        "observed_row_count": int(combined["observation_available"].sum()),
        "gap_count": int((~combined["observation_available"]).sum()),
        "timestamp_min_utc": combined["timestamp_utc"].min().isoformat(),
        "timestamp_max_utc": combined["timestamp_utc"].max().isoformat(),
        "source_hashes": {
            str(pv_path): sha256_file(pv_path),
            str(metadata_path): sha256_file(metadata_path),
        },
        "sites": manifests,
        "adapter": {
            "source_layout": "wide_datetime_index",
            "source_units": value_kind,
            "source_series_count": int(power.shape[1]),
        },
    }
    if output_path is not None:
        write_processed_dataset(combined, manifest, output_path)
    return combined, manifest


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".p", ".pkl", ".pickle"}:
        value = pd.read_pickle(path)
    elif suffix == ".parquet":
        value = pd.read_parquet(path)
    else:
        value = pd.read_csv(path)
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"Expected a pandas DataFrame in {path}.")
    return value.copy()


def _normalize_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    out = metadata.copy()
    if not any(column in out.columns for column in PV_COLUMN_ALIASES):
        if out.index.name in PV_COLUMN_ALIASES:
            out = out.reset_index()
        else:
            raise ValueError(
                "Metadata needs a site ID column or a named site ID index; "
                f"got index name {out.index.name!r}."
            )
    out = out.rename(
        columns={
            _first_existing(out, PV_COLUMN_ALIASES): "site_id",
            _first_existing(out, LATITUDE_ALIASES): "latitude",
            _first_existing(out, LONGITUDE_ALIASES): "longitude",
        }
    )
    mw_column = next((column for column in CAPACITY_MW_ALIASES if column in out), None)
    kw_column = next((column for column in CAPACITY_KW_ALIASES if column in out), None)
    if mw_column:
        out["installed_capacity_mw"] = pd.to_numeric(out[mw_column], errors="raise")
    elif kw_column:
        out["installed_capacity_mw"] = pd.to_numeric(out[kw_column], errors="raise") / 1000
    else:
        raise ValueError(
            "Metadata needs installed capacity in MW or kW; pv_max is accepted "
            "for the restored layout."
        )
    return out


def _first_existing(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"Expected one of columns {names}; got {list(frame.columns)}.")
