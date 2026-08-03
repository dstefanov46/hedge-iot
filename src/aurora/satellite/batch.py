"""Safe, resumable satellite collection and processing orchestration."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurora.satellite.acquisition import collect_in_chunks, validate_native_archive
from aurora.satellite.config import SatelliteConfig, SiteConfig
from aurora.satellite.features import derive_patch_features
from aurora.satellite.preprocess import (
    decode_native,
    load_patch,
    preprocess_physical_channels,
    validate_patch,
)
from aurora.satellite.storage import atomic_copy, sha256_file, write_json


def UTC_NOW() -> str:
    return datetime.now(UTC).isoformat()


def _local(value: str | Path) -> Path:
    if "://" in str(value) and not str(value).startswith("file://"):
        raise ValueError("batch processing/deletion currently requires local roots")
    return Path(str(value).replace("file://", "")).resolve()


def _manifest_paths(config: SatelliteConfig) -> tuple[Path, Path]:
    root = _local(config.raw_uri)
    return root / "manifest.json", root / "manifest.events.jsonl"


def _write_manifest(
    path: Path, records: list[dict[str, Any]], site: SiteConfig, config: SatelliteConfig
) -> None:
    payload = {"site": site.__dict__, "collection_run": config.metadata(site), "products": records}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _event(
    events: Path,
    product_id: str,
    old: dict[str, Any],
    new: dict[str, Any],
    error: str | None = None,
) -> None:
    events.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "product_id": product_id,
        "from": old,
        "to": new,
        "error": error,
        "updated_at_utc": UTC_NOW(),
    }
    with events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _state(record: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "archive_checksum": record.get("checksum"),
        "processing_checksum": None,
        "backup_status": "pending",
        "backup_checksum": None,
        "deletion_status": "pending",
        "attempt_count": 0,
        "last_error": record.get("error_state"),
        "updated_at_utc": UTC_NOW(),
    }
    for key, value in defaults.items():
        record.setdefault(key, value)
    return record


def _transition(
    record: dict[str, Any], status: str, events: Path, error: str | None = None
) -> None:
    old = dict(record)
    record["processing_status"] = status
    record["last_error"] = error
    record["attempt_count"] = int(record.get("attempt_count", 0)) + 1
    record["updated_at_utc"] = UTC_NOW()
    _event(events, str(record["product_id"]), old, record, error)


def _records(config: SatelliteConfig) -> list[dict[str, Any]]:
    manifest, events = _manifest_paths(config)
    if not manifest.exists():
        return []
    records = {
        str(x["product_id"]): _state(x)
        for x in json.loads(manifest.read_text(encoding="utf-8")).get("products", [])
    }
    if events.exists():
        for line in events.read_text(encoding="utf-8").splitlines():
            if line.strip():
                event = json.loads(line)
                target = event.get("to")
                if target and str(event["product_id"]) in records:
                    records[str(event["product_id"])] = _state(target)
    return sorted(
        records.values(), key=lambda x: (str(x.get("sensing_time", "")), str(x["product_id"]))
    )


def _artifact_checksum(patch_uri: str) -> str:
    path = Path(patch_uri)
    digest = hashlib.sha256()
    if path.is_file():
        return sha256_file(path)
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(child.relative_to(path).as_posix().encode())
            digest.update(child.read_bytes())
    sidecar = path.with_suffix(path.suffix + ".json")
    if sidecar.exists():
        digest.update(sidecar.read_bytes())
    return digest.hexdigest()


def _copy_tree_verified(
    source: Path, destination: Path, inventory: dict[str, str], prefix: str = ""
) -> None:
    for item in sorted(source.rglob("*")):
        if item.is_file():
            relative = f"{prefix}{item.relative_to(source).as_posix()}"
            target = destination / relative
            atomic_copy(item, str(target))
            inventory[relative] = sha256_file(target)


def backup_batch(
    config: SatelliteConfig, site: SiteConfig, batch_id: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    if not config.backup_uri:
        raise ValueError("backup_uri is required before satellite deletion")
    raw, processed, backup = (
        _local(config.raw_uri),
        _local(config.processed_uri),
        _local(config.backup_uri),
    )
    if (
        backup in {raw, processed}
        or raw in backup.parents
        or processed in backup.parents
        or backup in raw.parents
        or backup in processed.parents
    ):
        raise ValueError("backup_uri must be separate from raw_uri and processed_uri")
    if not processed.is_dir():
        raise ValueError(f"processed satellite directory does not exist: {processed}")
    inventory: dict[str, str] = {}
    _copy_tree_verified(processed, backup, inventory, "processed/")
    manifest, events = _manifest_paths(config)
    raw_parquet = raw / "manifest.parquet"
    for source, name in (
        (manifest, "raw/manifest.json"),
        (events, "raw/manifest.events.jsonl"),
        (raw_parquet, "raw/manifest.parquet"),
    ):
        if source.exists():
            target = backup / name
            atomic_copy(source, str(target))
            inventory[name] = sha256_file(target)
    metadata = backup / "batch-config.json"
    write_json(
        str(metadata),
        {"batch_id": batch_id, "site": site.__dict__, "config": config.metadata(site)},
    )
    inventory["batch-config.json"] = sha256_file(metadata)
    # Verify the copied bytes, then publish the marker last.
    for relative, checksum in inventory.items():
        if sha256_file(backup / relative) != checksum:
            raise ValueError(f"backup checksum mismatch: {relative}")
    marker = {"batch_id": batch_id, "completed_at_utc": UTC_NOW(), "checksums": inventory}
    write_json(str(backup / "BACKUP_COMPLETE.json"), marker)
    for record in records:
        if record.get("processing_status") == "processed":
            record["backup_status"] = "verified"
            record["backup_checksum"] = hashlib.sha256(
                json.dumps(inventory, sort_keys=True).encode()
            ).hexdigest()
    return marker


def _safe_archive_path(archive: str | Path, raw_root: Path) -> Path:
    target = Path(archive).resolve()
    if target == raw_root or raw_root not in target.parents:
        raise ValueError(f"archive is outside configured raw root: {target}")
    return target


def delete_verified_native(
    config: SatelliteConfig, records: list[dict[str, Any]], *, dry_run: bool = False
) -> list[str]:
    if not config.backup_uri:
        raise ValueError("backup_uri is required for native deletion")
    raw, processed, backup = (
        _local(config.raw_uri),
        _local(config.processed_uri),
        _local(config.backup_uri),
    )
    marker_path = backup / "BACKUP_COMPLETE.json"
    if not marker_path.exists():
        raise ValueError("verified backup completion marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    eligible: list[str] = []
    for record in records:
        if record.get("deletion_status") == "deleted":
            continue
        if (
            record.get("processing_status") != "processed"
            or record.get("backup_status") != "verified"
        ):
            continue
        archive = _safe_archive_path(record.get("archive_uri") or record.get("local_uri"), raw)
        patch = Path(record.get("patch_uri", ""))
        if not patch.exists() or not patch.with_suffix(patch.suffix + ".json").exists():
            continue
        # The backup inventory must contain the patch and its sidecar.
        for item in (patch, patch.with_suffix(patch.suffix + ".json")):
            if item.is_file():
                rel = f"processed/{item.relative_to(processed).as_posix()}"
                if (
                    rel not in marker["checksums"]
                    or sha256_file(backup / rel) != marker["checksums"][rel]
                ):
                    raise ValueError(f"backup checksum mismatch: {rel}")
        eligible.append(str(archive))
        if not dry_run and archive.exists():
            old = dict(record)
            archive.unlink()
            record["deletion_status"] = "deleted"
            record["updated_at_utc"] = UTC_NOW()
            _event(_manifest_paths(config)[1], str(record["product_id"]), old, record)
    return eligible


def run_satellite_batch(
    start: datetime,
    end: datetime,
    site: SiteConfig,
    config: SatelliteConfig,
    *,
    client: Any = None,
    decoder: Callable[[str | Path, SiteConfig], dict[str, np.ndarray]] = decode_native,
    delete_native: bool = False,
    dry_run: bool = False,
    chunk_hours: int | None = None,
) -> dict[str, Any]:
    """Run collect → process → backup → optional safe deletion."""
    config.validate_collection()
    start, end = config.ensure_utc(start), config.ensure_utc(end)
    manifest_path, events = _manifest_paths(config)
    all_records: list[dict[str, Any]] = []
    cursor, step = start, timedelta(hours=chunk_hours or config.collection_chunk_hours)
    while cursor < end:
        chunk_end = min(end, cursor + step)
        collect_in_chunks(
            cursor,
            chunk_end,
            site,
            config,
            client=client,
            chunk_hours=chunk_hours,
            validate_archives=True,
        )
        records = _records(config)
        processed_root = _local(config.processed_uri) / "physical_patches"
        feature_rows: list[dict[str, Any]] = []
        for record in records:
            timestamp = datetime.fromisoformat(
                str(record.get("sensing_time", "")).replace("Z", "+00:00")
            )
            if (
                not (cursor <= timestamp < chunk_end)
                or record.get("processing_status") == "processed"
            ):
                continue
            try:
                archive = Path(record.get("archive_uri") or record.get("local_uri"))
                validate_native_archive(archive)
                record["archive_checksum"] = sha256_file(archive)
                channels = decoder(archive, site)
                patch_uri = processed_root / f"{record['product_id']}.zarr"
                result = preprocess_physical_channels(
                    channels,
                    site,
                    str(record["product_id"]),
                    timestamp,
                    str(patch_uri),
                    source_archive_checksum=record["archive_checksum"],
                )
                validate_patch(load_patch(result["patch_uri"]), 12, site.patch_size)
                record.update(result)
                record["processing_checksum"] = _artifact_checksum(result["patch_uri"])
                feature_rows.append(
                    derive_patch_features(
                        load_patch(result["patch_uri"]),
                        result["patch_uri"],
                        site_id=site.site_id,
                        timestamp_utc=result["timestamp_utc"],
                        product_id=str(record["product_id"]),
                    )
                )
                _transition(record, "processed", events)
            except Exception as exc:
                _transition(record, "failed", events, str(exc))
        if feature_rows:
            output = _local(config.processed_uri) / "features.parquet"
            old = pd.read_parquet(output) if output.exists() else pd.DataFrame()
            pd.concat([old, pd.DataFrame(feature_rows)], ignore_index=True).drop_duplicates(
                ["product_id"], keep="last"
            ).to_parquet(output, index=False)
        _write_manifest(manifest_path, records, site, config)
        all_records = records
        cursor = chunk_end
    batch_id = hashlib.sha256(
        f"{start.isoformat()}:{end.isoformat()}:{site.site_id}".encode()
    ).hexdigest()[:16]
    if config.backup_uri and any(r.get("processing_status") == "processed" for r in all_records):
        backup_batch(config, site, batch_id, all_records)
        _write_manifest(manifest_path, all_records, site, config)
    if delete_native:
        delete_verified_native(config, all_records, dry_run=dry_run)
        _write_manifest(manifest_path, all_records, site, config)
    failed = [r for r in all_records if r.get("processing_status") == "failed"]
    return {
        "batch_id": batch_id,
        "products": all_records,
        "failed_products": failed,
        "eligible_deletions": [
            r["product_id"]
            for r in all_records
            if r.get("processing_status") == "processed" and r.get("backup_status") == "verified"
        ],
    }


def finalize_satellite_folds(
    records: pd.DataFrame | list[dict[str, Any]],
    folds: list[dict[str, Any]],
    processed_root: str | Path,
    model_root: str | Path,
    *,
    epochs: int = 1,
) -> list[dict[str, Any]]:
    """Fit fold normalization and encoders using training timestamps only."""
    from aurora.satellite.embedding import FrozenCNNEncoder

    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    results = []
    for fold in folds:
        fid = fold.get("fold_id", fold.get("id"))
        start = pd.Timestamp(fold["train_start"])
        end = pd.Timestamp(fold["train_end"])
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        train = frame[(frame.timestamp_utc >= start) & (frame.timestamp_utc <= end)]
        if train.empty:
            raise ValueError(f"fold {fid} has no training patches")
        physical = np.stack([load_patch(x) for x in train.patch_uri])
        from aurora.satellite.preprocess import (
            fit_normalization_stats,
            normalize_patch,
            save_normalization_stats,
        )

        stats = fit_normalization_stats(physical)
        fold_dir = _local(processed_root) / "folds" / f"fold_{fid}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_normalization_stats(stats, fold_dir / "normalization_stats.json")
        encoder = FrozenCNNEncoder().fit(
            np.stack(
                [
                    normalize_patch(p, np.asarray(stats["means"]), np.asarray(stats["scales"]))[0]
                    for p in physical
                ]
            ),
            epochs=epochs,
        )
        checkpoint = _local(model_root) / f"fold_{fid}" / "encoder.pt"
        metadata = encoder.save_checkpoint(
            checkpoint,
            training_interval={"start": start.isoformat(), "end": end.isoformat()},
            normalization=stats,
        )
        rows = []
        for _, row in frame.iterrows():
            normalized = normalize_patch(
                load_patch(row.patch_uri), np.asarray(stats["means"]), np.asarray(stats["scales"])
            )[0]
            rows.append(
                {
                    "product_id": row.product_id,
                    "timestamp_utc": row.timestamp_utc,
                    **{
                        f"embedding_{i:02d}": float(v)
                        for i, v in enumerate(encoder.encode(normalized))
                    },
                }
            )
        pd.DataFrame(rows).to_parquet(fold_dir / "embeddings.parquet", index=False)
        results.append(
            {
                "fold_id": fid,
                "normalization": stats,
                "checkpoint": str(checkpoint),
                "metadata": metadata,
            }
        )
    return results
