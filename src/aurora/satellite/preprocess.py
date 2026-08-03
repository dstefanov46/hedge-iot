from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from aurora.satellite.config import CHANNEL_NAMES, SiteConfig


def validate_channels(
    channels: dict[str, np.ndarray], expected: tuple[str, ...] = CHANNEL_NAMES
) -> None:
    missing = [name for name in expected if name not in channels]
    if missing:
        raise ValueError(f"Missing satellite channels: {missing}")
    if len(channels) != len(expected):
        raise ValueError(f"Expected exactly {len(expected)} channels")
    shapes = {np.asarray(value).shape for value in channels.values()}
    if len(shapes) != 1:
        raise ValueError(f"Channels do not share spatial dimensions: {shapes}")
    if len(next(iter(shapes))) != 2:
        raise ValueError("Satellite channels must be two-dimensional")


def extract_center_patch(
    channels: dict[str, np.ndarray], row: int, col: int, size: int = 64
) -> np.ndarray:
    validate_channels(channels)
    if size <= 0 or size % 2:
        raise ValueError("patch size must be a positive even number")
    arrays = [np.asarray(channels[name], dtype=np.float32) for name in CHANNEL_NAMES]
    half = size // 2
    output = np.full((len(arrays), size, size), np.nan, dtype=np.float32)
    for index, array in enumerate(arrays):
        r0, r1, c0, c1 = row - half, row + half, col - half, col + half
        sr0, sr1 = max(0, r0), min(array.shape[0], r1)
        sc0, sc1 = max(0, c0), min(array.shape[1], c1)
        output[index, sr0 - r0 : sr1 - r0, sc0 - c0 : sc1 - c0] = array[sr0:sr1, sc0:sc1]
    return output


def validate_patch(
    patch: np.ndarray,
    expected_channels: int = 12,
    expected_size: int = 64,
    value_ranges: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    patch = np.asarray(patch)
    if patch.shape[0] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {patch.shape[0]}")
    if patch.shape[1:] != (expected_size, expected_size):
        raise ValueError(f"Expected {(expected_size, expected_size)} patch, got {patch.shape[1:]}")
    missing = float(np.isnan(patch).mean())
    if missing > 1:
        raise ValueError("invalid missing fraction")
    if np.isinf(patch).any():
        raise ValueError("satellite patch contains infinite values")
    if value_ranges is not None:
        if len(value_ranges) != expected_channels:
            raise ValueError("value_ranges must contain one range per channel")
        for index, (lower, upper) in enumerate(value_ranges):
            values = patch[index][np.isfinite(patch[index])]
            if values.size and (values.min() < lower or values.max() > upper):
                raise ValueError(f"channel {index} contains out-of-range values")
    return {
        "height": int(patch.shape[1]),
        "width": int(patch.shape[2]),
        "missing_fraction": missing,
        "quality_status": "missing" if missing == 1 else ("degraded" if missing > 0 else "ok"),
    }


def normalize_patch(
    patch: np.ndarray, means: np.ndarray | None = None, scales: np.ndarray | None = None,
    version: str = "seviri-global-v1",
) -> tuple[np.ndarray, dict[str, Any]]:
    patch = np.asarray(patch, dtype=np.float32)
    if patch.shape[0] != 12:
        raise ValueError("normalization expects 12 channels")
    means = np.zeros(12, dtype=np.float32) if means is None else np.asarray(means, dtype=np.float32)
    scales = (
        np.ones(12, dtype=np.float32) if scales is None else np.asarray(scales, dtype=np.float32)
    )
    if means.shape != (12,) or scales.shape != (12,) or np.any(scales <= 0):
        raise ValueError("invalid normalization parameters")
    return ((patch - means[:, None, None]) / scales[:, None, None]).astype(np.float32), {
        "version": version,
        "means": means.tolist(),
        "scales": scales.tolist(),
    }


def fit_normalization_stats(
    patches: list[np.ndarray] | np.ndarray, version: str = "seviri-global-v1"
) -> dict[str, Any]:
    """Fit per-channel statistics from supplied patches only.

    Callers must pass training-period patches; this function deliberately has no
    access to split labels and therefore cannot silently pool validation/test data.
    """
    values = np.asarray(patches, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (12, 64, 64):
        raise ValueError("expected training patches with shape (n, 12, 64, 64)")
    means = np.nanmean(values, axis=(0, 2, 3))
    scales = np.nanstd(values, axis=(0, 2, 3))
    if not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise ValueError("normalization cannot be fitted from all-missing channels")
    scales = np.maximum(scales, 1e-6)
    return {
        "version": version,
        "means": means.astype(float).tolist(),
        "scales": scales.astype(float).tolist(),
        "training_patch_count": int(values.shape[0]),
    }


def save_normalization_stats(stats: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def load_normalization_stats(path: str | Path) -> tuple[np.ndarray, np.ndarray, str]:
    stats = json.loads(Path(path).read_text(encoding="utf-8"))
    means = np.asarray(stats["means"], dtype=np.float32)
    scales = np.asarray(stats["scales"], dtype=np.float32)
    if means.shape != (12,) or scales.shape != (12,):
        raise ValueError("normalization statistics must contain 12 means and scales")
    return means, scales, str(stats["version"])


def save_patch(patch: np.ndarray, uri: str | Path, metadata: dict[str, Any]) -> str:
    """Save dense patch as Zarr when installed; otherwise use a portable NPZ fallback."""
    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import zarr

        if path.exists():
            shutil.rmtree(path)

        zarr.save(str(path), np.asarray(patch, dtype=np.float32))
        path.with_suffix(path.suffix + ".json").write_text(
            __import__("json").dumps(metadata), encoding="utf-8"
        )
        return str(path)
    except ImportError:
        # Keep the canonical .zarr name even in minimal installations.  The
        # directory contains a NumPy payload and remains portable/readable.
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "data.npy", np.asarray(patch, dtype=np.float32))
        path.with_suffix(path.suffix + ".json").write_text(
            __import__("json").dumps(metadata), encoding="utf-8"
        )
        return str(path)


def load_patch(uri: str | Path) -> np.ndarray:
    """Read either a real Zarr patch or the portable fallback store."""
    path = Path(uri)
    if path.is_dir() and (path / "data.npy").exists():
        return np.load(path / "data.npy")
    if path.suffix == ".zarr":
        import zarr
        return np.asarray(zarr.open(path, mode="r"))
    return np.load(path)["patch"]


def preprocess_channels(
    channels: dict[str, np.ndarray],
    site: SiteConfig,
    product_id: str,
    timestamp: datetime,
    patch_uri: str,
    means: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    normalization_version: str = "seviri-global-v1",
) -> dict[str, Any]:
    if (
        timestamp.tzinfo is None
        or timestamp.utcoffset() is None
        or timestamp.astimezone(UTC) != timestamp
    ):
        raise ValueError("timestamp must be UTC")
    validate_channels(channels)
    center = (np.asarray(next(iter(channels.values())).shape) // 2).astype(int)
    patch = extract_center_patch(channels, int(center[0]), int(center[1]), site.patch_size)
    normalized_patch, normalization = normalize_patch(patch, means, scales, normalization_version)
    metadata = validate_patch(normalized_patch, 12, site.patch_size)
    return {
        "site_id": site.site_id,
        "timestamp_utc": timestamp.isoformat(),
        "product_id": product_id,
        "provider": "EUMETSAT",
        "product": "EO:EUM:DAT:MSG:HRSEVIRI",
        "patch_uri": save_patch(
            normalized_patch,
            patch_uri,
            {
                "channel_names": list(CHANNEL_NAMES),
                "site": asdict(site),
                "normalization": normalization,
                "timestamp_utc": timestamp.isoformat(),
                "product_id": product_id,
                "missing_fraction": metadata["missing_fraction"],
                "quality_status": metadata["quality_status"],
            },
        ),
        "channel_names": list(CHANNEL_NAMES),
        **metadata,
        "normalization_version": normalization["version"],
    }


def preprocess_physical_channels(
    channels: dict[str, np.ndarray], site: SiteConfig, product_id: str,
    timestamp: datetime, patch_uri: str, *, source_archive_checksum: str | None = None,
    preprocessing_version: str = "physical-patch-v1",
) -> dict[str, Any]:
    """Decode and retain a validated physical-unit patch without normalization."""
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    timestamp = timestamp.astimezone(UTC)
    validate_channels(channels)
    center = (np.asarray(next(iter(channels.values())).shape) // 2).astype(int)
    patch = extract_center_patch(channels, int(center[0]), int(center[1]), site.patch_size)
    metadata = validate_patch(patch, len(CHANNEL_NAMES), site.patch_size)
    metadata = {
        "site": asdict(site), "site_id": site.site_id, "product_id": product_id,
        "sensing_timestamp": timestamp.isoformat(), "timestamp_utc": timestamp.isoformat(),
        "channel_names": list(CHANNEL_NAMES), "patch_dimensions": list(patch.shape),
        "source_archive_checksum": source_archive_checksum,
        "preprocessing_version": preprocessing_version, **metadata,
    }
    return {**metadata, "patch_uri": save_patch(patch, patch_uri, metadata)}


def decode_native(path: str | Path, site: SiteConfig) -> dict[str, np.ndarray]:
    try:
        from satpy import Scene
    except ImportError as exc:
        raise RuntimeError("Satpy is required to decode native SEVIRI products") from exc
    source = Path(path)
    temporary_directory: Path | None = None
    if zipfile.is_zipfile(source):
        temporary_directory = Path(tempfile.mkdtemp(prefix="aurora-seviri-"))
        with zipfile.ZipFile(source) as archive:
            members = [member for member in archive.namelist() if member.lower().endswith(".nat")]
            if len(members) != 1:
                raise ValueError(f"Native archive must contain exactly one .nat payload: {source}")
            member = members[0]
            source = temporary_directory / Path(member).name
            with archive.open(member) as source_handle, source.open("wb") as target:
                shutil.copyfileobj(source_handle, target)
    elif source.suffix.lower() != ".nat":
        temporary_directory = tempfile.mkdtemp(prefix="aurora-seviri-")
        temporary = Path(temporary_directory) / f"{source.stem}.nat"
        shutil.copyfile(source, temporary)
        source = temporary
    try:
        scene = Scene(reader="seviri_l1b_native", filenames=[str(source)])
        scene.load(list(CHANNEL_NAMES))
        # Satpy's resampling uses the native common area and keeps HRV alignment explicit.
        area = scene["VIS006"].area
        scene = scene.resample(area)
        return {name: np.asarray(scene[name].values) for name in CHANNEL_NAMES}
    finally:
        if temporary_directory is not None:
            shutil.rmtree(temporary_directory, ignore_errors=True)
