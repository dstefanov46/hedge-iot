"""Build the shared satellite feature contract used by TFT transfer."""
from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterable
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd

from aurora.satellite.embedding import FrozenCNNEncoder
from aurora.satellite.features import align_satellite_features, derive_patch_features
from aurora.satellite.preprocess import fit_normalization_stats, load_patch, normalize_patch
from aurora.satellite.slovenian import (
    discover_archives,
    iter_site_patches,
    sample_site_patches,
)

EMBEDDING_COLUMNS = [f"satellite_embedding_{i:02d}" for i in range(32)]
SATELLITE_COLUMNS = EMBEDDING_COLUMNS + [
    "satellite_missing", "satellite_cloud_index", "satellite_irradiance_proxy"
]
TRANSFER_CONTRACT_VERSION = "satellite-transfer-v2"


def _record_row(site_id: str, timestamp: object, patch: np.ndarray, encoder: FrozenCNNEncoder,
                means: np.ndarray, scales: np.ndarray) -> dict[str, object]:
    normalized, _ = normalize_patch(patch, means, scales)
    row = derive_patch_features(patch, site_id=site_id, timestamp_utc=timestamp)
    row.update({name: float(value) for name, value in zip(
        EMBEDDING_COLUMNS, encoder.encode(normalized), strict=True
    )})
    row["satellite_sensing_timestamp_utc"] = pd.Timestamp(timestamp)
    row["satellite_issue_timestamp_utc"] = pd.Timestamp(timestamp)
    return row


def _missing_features() -> dict[str, float]:
    return {**{name: 0.0 for name in EMBEDDING_COLUMNS},
            "satellite_missing": 1.0, "satellite_cloud_index": 0.0,
            "satellite_irradiance_proxy": 0.0}


def _feature_frame(
    records: Iterable[tuple[str, object, np.ndarray]], encoder, means, scales
) -> pd.DataFrame:
    rows = [_record_row(site, timestamp, patch, encoder, means, scales)
            for site, timestamp, patch in records]
    return pd.DataFrame(rows)


def finnish_patch_records(patch_root: str | Path) -> list[tuple[str, pd.Timestamp, np.ndarray]]:
    """Read local Finnish sidecars, resolving old absolute URIs by basename."""
    root = Path(patch_root)
    records = []
    for sidecar in root.glob("*.zarr.json"):
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        uri = sidecar.with_suffix("")
        if not uri.exists():
            continue
        records.append((
            str(metadata["site_id"]),
            pd.to_datetime(metadata["timestamp_utc"], utc=True),
            load_patch(uri),
        ))
    return records


def prepare_satellite_transfer(
    slovenian: pd.DataFrame,
    finnish: pd.DataFrame,
    *,
    slovenian_zarr_root: str | Path,
    finnish_patch_root: str | Path,
    checkpoint_path: str | Path,
    training_start: object | None = None,
    training_end: object | None = None,
    max_training_patches: int = 4096,
    max_source_archives: int | None = None,
    max_finnish_patches: int | None = None,
    max_alignment_minutes: float = 15.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Attach one frozen, Slovenian-trained encoder to both country frames."""
    sites = slovenian[["site_id", "latitude", "longitude"]].drop_duplicates("site_id")
    start = pd.to_datetime(training_start or slovenian["timestamp_utc"].min(), utc=True)
    end = pd.to_datetime(training_end or slovenian["timestamp_utc"].max(), utc=True)
    training = list(sample_site_patches(
        slovenian_zarr_root, sites, max_training_patches, start=start, end=end
    ))
    if not training:
        raise ValueError("No Slovenian satellite patches fall inside the encoder training interval")
    stats = fit_normalization_stats(np.stack(training))
    means = np.asarray(stats["means"], dtype=np.float32)
    scales = np.asarray(stats["scales"], dtype=np.float32)
    encoder = FrozenCNNEncoder().fit(np.stack(training))
    metadata = encoder.save_checkpoint(
        checkpoint_path,
        training_interval={"start": start.isoformat(), "end": end.isoformat()},
        normalization={
            **stats, "training_start": start.isoformat(), "training_end": end.isoformat()
        },
    )
    source_records = iter_site_patches(
        slovenian_zarr_root, sites, start=slovenian["timestamp_utc"].min(),
        end=slovenian["timestamp_utc"].max(), max_archives=max_source_archives,
    )
    finnish_records = finnish_patch_records(finnish_patch_root)
    if max_finnish_patches is not None:
        finnish_records = finnish_records[:max_finnish_patches]
    sol_sat = _feature_frame(source_records, encoder, means, scales)
    fin_sat = _feature_frame(finnish_records, encoder, means, scales)
    sol = align_satellite_features(
        slovenian, sol_sat, tolerance_minutes=max_alignment_minutes
    )
    fin = align_satellite_features(
        finnish, fin_sat, tolerance_minutes=max_alignment_minutes
    )
    for frame in (sol, fin):
        for column, value in _missing_features().items():
            if column not in frame:
                frame[column] = value
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(value)
    metadata["normalization"] = {
        **stats, "training_start": start.isoformat(), "training_end": end.isoformat()
    }
    return sol, fin, metadata


def materialize_satellite_transfer(
    slovenian_dataset: str | Path,
    finnish_dataset: str | Path,
    *,
    slovenian_zarr_root: str | Path,
    finnish_patch_root: str | Path,
    checkpoint_path: str | Path,
    output_root: str | Path,
    training_start: object | None = None,
    training_end: object | None = None,
    max_training_patches: int = 4096,
    max_alignment_minutes: float = 15.0,
    max_source_archives: int | None = None,
    max_finnish_patches: int | None = None,
) -> tuple[Path, Path, dict[str, object], dict[str, str]]:
    """Build country datasets without retaining a year of image patches in RAM."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    source_path = Path(slovenian_dataset)
    target_path = Path(finnish_dataset)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    prepared_source = destination / "slovenia.parquet"
    prepared_target = destination / "finnish.parquet"

    site_table = pq.read_table(
        source_path, columns=["site_id", "latitude", "longitude", "timestamp_utc"]
    ).to_pandas()
    sites = site_table[["site_id", "latitude", "longitude"]].drop_duplicates("site_id")
    source_start = pd.to_datetime(site_table["timestamp_utc"].min(), utc=True)
    source_end = pd.to_datetime(site_table["timestamp_utc"].max(), utc=True)
    del site_table
    encoder_start = pd.to_datetime(training_start or source_start, utc=True)
    encoder_end = pd.to_datetime(training_end or source_end, utc=True)

    training = list(sample_site_patches(
        slovenian_zarr_root, sites, max_training_patches,
        start=encoder_start, end=encoder_end,
    ))
    if not training:
        raise ValueError("No Slovenian satellite patches fall inside the encoder training interval")
    training_array = np.stack(training)
    stats = fit_normalization_stats(training_array)
    means = np.asarray(stats["means"], dtype=np.float32)
    scales = np.asarray(stats["scales"], dtype=np.float32)
    normalized_training = (
        (training_array - means[None, :, None, None])
        / scales[None, :, None, None]
    )
    encoder = FrozenCNNEncoder().fit(normalized_training)
    del training, training_array, normalized_training
    metadata = encoder.save_checkpoint(
        checkpoint_path,
        training_interval={"start": encoder_start.isoformat(), "end": encoder_end.isoformat()},
        normalization={
            **stats, "training_start": encoder_start.isoformat(),
            "training_end": encoder_end.isoformat(),
        },
    )
    metadata["contract_version"] = TRANSFER_CONTRACT_VERSION

    with tempfile.TemporaryDirectory(prefix="satellite-features-", dir=destination) as temp:
        shard_root = Path(temp)
        source_coverage = _write_slovenian_feature_shards(
            sites, slovenian_zarr_root, shard_root, encoder, means, scales,
            source_start, source_end, max_source_archives=max_source_archives,
        )
        source_alignment = _align_source_shards(
            source_path, sites, shard_root, prepared_source,
            max_alignment_minutes=max_alignment_minutes,
        )
        target = pq.read_table(target_path).to_pandas()
        target_features = _finnish_feature_frame(
            finnish_patch_root, encoder, means, scales,
            max_patches=max_finnish_patches,
        )
        aligned_target = align_satellite_features(
            target, target_features, tolerance_minutes=max_alignment_minutes
        )
        aligned_target = _fill_satellite_columns(aligned_target)
        pq.write_table(
            pa.Table.from_pandas(aligned_target, preserve_index=False),
            prepared_target, compression="zstd",
        )
        target_alignment = _alignment_summary(aligned_target)

    source_hashes = {
        str(source_path.resolve()): _sha256(source_path),
        str(target_path.resolve()): _sha256(target_path),
        f"{Path(slovenian_zarr_root).resolve()}#inventory": _archive_inventory_hash(
            slovenian_zarr_root, start=source_start, end=source_end,
            max_archives=max_source_archives,
        ),
        f"{Path(finnish_patch_root).resolve()}#inventory": _patch_inventory_hash(
            finnish_patch_root, max_patches=max_finnish_patches
        ),
    }
    metadata["source_coverage"] = source_coverage
    metadata["alignment"] = {"slovenia": source_alignment, "finland": target_alignment}
    metadata_path = destination / "encoder_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    for dataset, country, alignment in (
        (prepared_source, "Slovenia", source_alignment),
        (prepared_target, "Finland", target_alignment),
    ):
        dataset.with_suffix(".manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": TRANSFER_CONTRACT_VERSION,
                    "country": country,
                    "source_hashes": source_hashes,
                    "alignment": alignment,
                    "encoder_checkpoint_sha256": metadata["checkpoint_sha256"],
                },
                indent=2, sort_keys=True,
            ),
            encoding="utf-8",
        )
    return prepared_source, prepared_target, metadata, source_hashes


def _feature_rows(
    site_ids: list[str], timestamps: list[pd.Timestamp], patches: np.ndarray,
    encoder: FrozenCNNEncoder, means: np.ndarray, scales: np.ndarray,
) -> list[dict[str, object]]:
    normalized = (
        (np.asarray(patches, dtype=np.float32) - means[None, :, None, None])
        / scales[None, :, None, None]
    )
    embeddings = encoder.encode_batch(normalized)
    visible = patches[:, :3]
    infrared = patches[:, 3:11]
    visible_mean = np.nanmean(visible, axis=(1, 2, 3))
    infrared_mean = np.nanmean(infrared, axis=(1, 2, 3))
    missing = np.isnan(patches).mean(axis=(1, 2, 3))
    rows = []
    for index, site_id in enumerate(site_ids):
        row: dict[str, object] = {
            "site_id": str(site_id),
            "timestamp_utc": timestamps[index],
            "satellite_sensing_timestamp_utc": timestamps[index],
            "satellite_missing": float(missing[index]),
            "satellite_cloud_index": float(
                visible_mean[index] / (infrared_mean[index] + 1e-6)
            ),
            "satellite_irradiance_proxy": float(visible_mean[index]),
        }
        row.update({
            name: float(value)
            for name, value in zip(EMBEDDING_COLUMNS, embeddings[index], strict=True)
        })
        rows.append(row)
    return rows


def _write_slovenian_feature_shards(
    sites: pd.DataFrame, root: str | Path, shard_root: Path,
    encoder: FrozenCNNEncoder, means: np.ndarray, scales: np.ndarray,
    start: pd.Timestamp, end: pd.Timestamp, *, max_source_archives: int | None,
) -> dict[str, object]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    site_ids = sites["site_id"].astype(str).tolist()
    buffers: dict[str, list[dict[str, object]]] = {site: [] for site in site_ids}
    writers: dict[str, pq.ParquetWriter] = {}
    archive_count = 0

    def flush() -> None:
        for site_id, rows in buffers.items():
            if not rows:
                continue
            table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
            path = shard_root / f"site-{site_id}.parquet"
            if site_id not in writers:
                writers[site_id] = pq.ParquetWriter(path, table.schema, compression="zstd")
            writers[site_id].write_table(table)
            rows.clear()

    records = iter_site_patches(
        root, sites, start=start, end=end, max_archives=max_source_archives
    )
    try:
        for _, group in groupby(records, key=itemgetter(1)):
            batch = list(group)
            ids = [str(item[0]) for item in batch]
            timestamps = [pd.Timestamp(item[1]) for item in batch]
            patches = np.stack([item[2] for item in batch])
            for row in _feature_rows(ids, timestamps, patches, encoder, means, scales):
                buffers[str(row["site_id"])].append(row)
            archive_count += 1
            if archive_count % 128 == 0:
                flush()
        flush()
    finally:
        for writer in writers.values():
            writer.close()
    return {
        "archive_count": archive_count,
        "site_count": len(site_ids),
        "feature_row_count": archive_count * len(site_ids),
    }


def _align_source_shards(
    source_path: Path, sites: pd.DataFrame, shard_root: Path, destination: Path,
    *, max_alignment_minutes: float,
) -> dict[str, object]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    matched = rows = 0
    try:
        for site_id in sites["site_id"].astype(str):
            source = pq.read_table(
                source_path, filters=[("site_id", "=", site_id)]
            ).to_pandas()
            shard = shard_root / f"site-{site_id}.parquet"
            satellite = pq.read_table(shard).to_pandas() if shard.exists() else pd.DataFrame(
                columns=["site_id", "timestamp_utc"]
            )
            aligned = align_satellite_features(
                source, satellite, tolerance_minutes=max_alignment_minutes
            )
            aligned = _fill_satellite_columns(aligned)
            summary = _alignment_summary(aligned)
            matched += int(summary["matched_rows"])
            rows += len(aligned)
            table = pa.Table.from_pandas(aligned, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return {"rows": rows, "matched_rows": matched, "matched_rate": matched / rows if rows else 0.0}


def _finnish_feature_frame(
    root: str | Path, encoder: FrozenCNNEncoder, means: np.ndarray, scales: np.ndarray,
    *, max_patches: int | None,
) -> pd.DataFrame:
    sidecars = sorted(Path(root).glob("*.zarr.json"))
    if max_patches is not None:
        sidecars = sidecars[:max_patches]
    rows: list[dict[str, object]] = []
    for offset in range(0, len(sidecars), 128):
        ids: list[str] = []
        timestamps: list[pd.Timestamp] = []
        patches: list[np.ndarray] = []
        for sidecar in sidecars[offset : offset + 128]:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            store = sidecar.with_suffix("")
            if not store.exists():
                continue
            ids.append(str(metadata["site_id"]))
            timestamps.append(pd.to_datetime(metadata["timestamp_utc"], utc=True))
            patches.append(load_patch(store))
        if patches:
            rows.extend(_feature_rows(
                ids, timestamps, np.stack(patches), encoder, means, scales
            ))
    return pd.DataFrame(rows)


def _fill_satellite_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column, value in _missing_features().items():
        if column not in result:
            result[column] = value
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(value).astype(
            np.float32
        )
    return result


def _alignment_summary(frame: pd.DataFrame) -> dict[str, object]:
    sensing = frame.get("satellite_sensing_timestamp_utc", pd.Series(dtype=object))
    matched = int(sensing.notna().sum())
    rows = len(frame)
    return {"rows": rows, "matched_rows": matched, "matched_rate": matched / rows if rows else 0.0}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_inventory_hash(
    root: str | Path, *, start: pd.Timestamp, end: pd.Timestamp,
    max_archives: int | None,
) -> str:
    archives = [
        archive for archive in discover_archives(root)
        if start <= archive.timestamp_utc <= end
    ]
    if max_archives is not None:
        archives = archives[:max_archives]
    digest = hashlib.sha256()
    for archive in archives:
        for path in (archive.nonhrv_path, archive.hrv_path):
            stat = path.stat()
            digest.update(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _patch_inventory_hash(root: str | Path, *, max_patches: int | None) -> str:
    sidecars = sorted(Path(root).glob("*.zarr.json"))
    if max_patches is not None:
        sidecars = sidecars[:max_patches]
    digest = hashlib.sha256()
    for sidecar in sidecars:
        stat = sidecar.stat()
        digest.update(f"{sidecar.name}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()
