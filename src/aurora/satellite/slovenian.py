"""Reader for the historical Satip Slovenian SEVIRI Zarr archives.

The archives are deliberately read one timestamp at a time.  This keeps the
adapter usable for the roughly 22k products without creating a multi-terabyte
in-memory array.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aurora.satellite.config import SiteConfig
from aurora.satellite.preprocess import extract_center_patch, validate_patch

_STAMP = re.compile(r"(?P<date>\d{8})(?P<hour>\d{2})(?P<minute>\d{2})")
NON_HRV_ORDER = (
    "VIS006", "VIS008", "IR_016", "IR_039", "WV_062", "WV_073",
    "IR_087", "IR_097", "IR_108", "IR_120", "IR_134",
)
_REFLECTANCE = {"VIS006", "VIS008", "IR_016", "HRV"}


@dataclass(frozen=True)
class SlovenianArchive:
    timestamp_utc: pd.Timestamp
    nonhrv_path: Path
    hrv_path: Path


def _timestamp(path: Path) -> pd.Timestamp:
    match = _STAMP.search(path.name)
    if not match:
        raise ValueError(f"Cannot determine timestamp from archive name: {path.name}")
    return pd.to_datetime(
        match.group("date") + match.group("hour") + match.group("minute"),
        format="%Y%m%d%H%M", utc=True,
    )


def discover_archives(root: str | Path) -> list[SlovenianArchive]:
    """Return only paired non-HRV/HRV products, sorted by sensing time."""
    root = Path(root)
    if not root.is_dir():
        return []
    nonhrv: dict[pd.Timestamp, Path] = {}
    hrv: dict[pd.Timestamp, Path] = {}
    # The restored archive has one directory per timestamp. Iterating those
    # directories directly avoids an expensive recursive walk over the USB
    # filesystem while retaining compatibility with a flat archive root.
    for entry in root.iterdir():
        paths = entry.glob("*.zarr.zip") if entry.is_dir() else (entry,)
        for path in paths:
            if not path.is_file() or not path.name.endswith(".zarr.zip"):
                continue
            stamp = _timestamp(path)
            (hrv if "hrv" in path.name.lower() else nonhrv)[stamp] = path
    return [
        SlovenianArchive(ts, nonhrv[ts], hrv[ts])
        for ts in sorted(nonhrv.keys() & hrv.keys())
    ]


def _read_group(path: Path):
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("The Slovenian Zarr adapter requires the satellite extra") from exc
    store = zarr.storage.ZipStore(str(path), mode="r")
    try:
        group = zarr.open_group(store=store, mode="r")
        # Copy only the arrays needed by one patch; the ZipStore is then closed.
        return {name: np.asarray(group[name]) for name in ("data", "variable", "lat", "lon")}
    finally:
        store.close()


def _physical(values: np.ndarray, channel: str) -> np.ndarray:
    """Undo Satip's unit interval representation when it is present.

    Older Satip exports contain physical brightness temperatures already,
    while some batches contain [0, 1] values.  Reflectance channels retain
    their unit-interval physical units; thermal channels use the documented
    180--330 K Satip brightness-temperature range.
    """
    result = np.asarray(values, dtype=np.float32)
    finite = result[np.isfinite(result)]
    if finite.size and float(np.nanmax(finite)) <= 1.0 and channel not in _REFLECTANCE:
        result = 180.0 + 150.0 * result
    return result


def _nearest_index(grid: np.ndarray, value: float) -> int:
    axis = np.nanmean(grid, axis=1 if grid.shape[0] > grid.shape[1] else 0)
    return int(np.nanargmin(np.abs(axis - value)))


def _resample_hrv(hrv: dict[str, np.ndarray], lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Nearest-neighbour resample HRV onto the non-HRV grid."""
    hlat, hlon = hrv["lat"], hrv["lon"]
    row_axis = np.nanmean(hlat, axis=1)
    col_axis = np.nanmean(hlon, axis=0)
    rows = np.abs(row_axis[:, None] - lat.ravel()[None, :]).argmin(axis=0)
    cols = np.abs(col_axis[:, None] - lon.ravel()[None, :]).argmin(axis=0)
    return hrv["data"][0, rows, cols, 0].reshape(lat.shape)


def read_archive(archive: SlovenianArchive) -> dict[str, np.ndarray]:
    """Decode and reorder a paired archive into canonical physical channels."""
    nonhrv, hrv = _read_group(archive.nonhrv_path), _read_group(archive.hrv_path)
    names = [str(x) for x in nonhrv["variable"]]
    channels = {name: _physical(nonhrv["data"][0, :, :, names.index(name)], name)
                for name in NON_HRV_ORDER}
    hrv_values = _physical(hrv["data"][0, :, :, 0], "HRV")
    channels["HRV"] = _resample_hrv({**hrv, "data": hrv_values[None, :, :, None]},
                                     nonhrv["lat"], nonhrv["lon"])
    channels["__lat"] = nonhrv["lat"]
    channels["__lon"] = nonhrv["lon"]
    return channels


def site_patch(archive: SlovenianArchive, site: SiteConfig) -> np.ndarray:
    channels = read_archive(archive)
    return _site_patch_from_channels(channels, site)


def _site_patch_from_channels(
    channels: dict[str, np.ndarray], site: SiteConfig
) -> np.ndarray:
    """Extract one site patch from an already-decoded product."""
    channels = channels.copy()
    lat, lon = channels.pop("__lat"), channels.pop("__lon")
    distance = (lat - site.latitude) ** 2 + (lon - site.longitude) ** 2
    row, col = np.unravel_index(np.nanargmin(distance), lat.shape)
    patch = extract_center_patch(channels, int(row), int(col), site.patch_size)
    validate_patch(patch, 12, site.patch_size)
    return patch.astype(np.float32, copy=False)


def iter_site_patches(
    root: str | Path,
    sites: pd.DataFrame,
    *,
    start: object | None = None,
    end: object | None = None,
    max_archives: int | None = None,
):
    """Yield site patches while decoding every satellite product only once."""
    lookup = {
        str(row.site_id): SiteConfig(str(row.site_id), float(row.latitude), float(row.longitude))
        for row in sites.itertuples()
    }
    start_ts = pd.to_datetime(start, utc=True) if start is not None else None
    end_ts = pd.to_datetime(end, utc=True) if end is not None else None
    archives = [
        archive
        for archive in discover_archives(root)
        if (start_ts is None or archive.timestamp_utc >= start_ts)
        and (end_ts is None or archive.timestamp_utc <= end_ts)
    ]
    if max_archives is not None:
        archives = archives[:max_archives]
    for archive in archives:
        channels = read_archive(archive)
        for site_id, site in lookup.items():
            yield site_id, archive.timestamp_utc, _site_patch_from_channels(channels, site)


def sample_site_patches(
    root: str | Path,
    sites: pd.DataFrame,
    count: int,
    *,
    start: object | None = None,
    end: object | None = None,
) -> Iterator[np.ndarray]:
    """Sample patches evenly across the requested time span and all sites."""
    if count <= 0:
        raise ValueError("count must be positive")
    lookup = [
        SiteConfig(str(row.site_id), float(row.latitude), float(row.longitude))
        for row in sites.itertuples()
    ]
    if not lookup:
        return
    start_ts = pd.to_datetime(start, utc=True) if start is not None else None
    end_ts = pd.to_datetime(end, utc=True) if end is not None else None
    archives = [
        archive
        for archive in discover_archives(root)
        if (start_ts is None or archive.timestamp_utc >= start_ts)
        and (end_ts is None or archive.timestamp_utc <= end_ts)
    ]
    if not archives:
        return
    required_archives = min(len(archives), int(np.ceil(count / len(lookup))))
    indices = np.linspace(0, len(archives) - 1, required_archives, dtype=int)
    yielded = 0
    for index in np.unique(indices):
        channels = read_archive(archives[int(index)])
        for site in lookup:
            yield _site_patch_from_channels(channels, site)
            yielded += 1
            if yielded >= count:
                return
