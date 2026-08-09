"""Safe, resumable satellite collection and processing orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from functools import partial
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurora.satellite.acquisition import (
    EumetsatClient,
    discover_products,
    download_product,
)
from aurora.satellite.config import SatelliteConfig, SiteConfig
from aurora.satellite.features import derive_patch_features
from aurora.satellite.preprocess import (
    EmptySatelliteProductError,
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


def _parse_utc(value: Any) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("discovery coverage timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def _merge_discovery_coverage(
    intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """Normalize and merge overlapping or adjacent half-open intervals."""
    normalized = []
    for start, end in intervals:
        normalized_start, normalized_end = _parse_utc(start), _parse_utc(end)
        if normalized_end > normalized_start:
            normalized.append((normalized_start, normalized_end))
    normalized.sort()
    merged: list[tuple[datetime, datetime]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _load_discovery_coverage(manifest: Path) -> list[tuple[datetime, datetime]]:
    """Load only explicitly trusted coverage; legacy scalar checkpoints are ignored."""
    if not manifest.exists():
        return []
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8")).get(
            "collection_run", {}
        ).get("discovery_coverage_utc", [])
        intervals = []
        for interval in raw:
            if isinstance(interval, dict):
                intervals.append((_parse_utc(interval["start"]), _parse_utc(interval["end"])))
            elif isinstance(interval, (list, tuple)) and len(interval) == 2:
                intervals.append((_parse_utc(interval[0]), _parse_utc(interval[1])))
        return _merge_discovery_coverage(intervals)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return []


def _has_explicit_discovery_coverage(manifest: Path) -> bool:
    if not manifest.exists():
        return False
    try:
        collection_run = json.loads(manifest.read_text(encoding="utf-8")).get(
            "collection_run", {}
        )
        return "discovery_coverage_utc" in collection_run
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _serialize_discovery_coverage(
    intervals: list[tuple[datetime, datetime]],
) -> list[dict[str, str]]:
    return [
        {"start": start.isoformat(), "end": end.isoformat()}
        for start, end in _merge_discovery_coverage(intervals)
    ]


def _contiguous_coverage_end(
    intervals: list[tuple[datetime, datetime]], start: datetime
) -> datetime:
    cursor = _parse_utc(start)
    for interval_start, interval_end in _merge_discovery_coverage(intervals):
        if interval_end <= cursor:
            continue
        if interval_start > cursor:
            break
        cursor = max(cursor, interval_end)
    return cursor


def _interval_is_covered(
    start: datetime,
    end: datetime,
    intervals: list[tuple[datetime, datetime]],
) -> bool:
    return any(
        covered_start <= start and end <= covered_end
        for covered_start, covered_end in intervals
    )


def _chunk_intervals(
    start: datetime, end: datetime, step: timedelta
) -> list[tuple[datetime, datetime]]:
    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor + step)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def _write_manifest(
    path: Path,
    records: list[dict[str, Any]],
    site: SiteConfig,
    config: SatelliteConfig,
    *,
    discovery_coverage_utc: list[tuple[datetime, datetime]] | None = None,
) -> None:
    collection_run = config.metadata(site)
    coverage = (
        _load_discovery_coverage(path)
        if discovery_coverage_utc is None
        else _merge_discovery_coverage(discovery_coverage_utc)
    )
    collection_run["discovery_coverage_utc"] = _serialize_discovery_coverage(coverage)
    try:
        configured_start = _parse_utc(config.start_utc)
    except (TypeError, ValueError):
        configured_start = coverage[0][0] if coverage else datetime.min.replace(tzinfo=UTC)
    # Retained for human-readable progress and older tooling only. Resume logic
    # exclusively trusts the explicit coverage intervals above.
    collection_run["discovery_checkpoint_utc"] = _contiguous_coverage_end(
        coverage, configured_start
    ).isoformat()
    payload = {"site": site.__dict__, "collection_run": collection_run, "products": records}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)
    try:
        from aurora.satellite.acquisition import manifest_to_parquet
        parquet = path.with_name("manifest.parquet")
        parquet_tmp = parquet.with_suffix(".parquet.tmp")
        manifest_to_parquet(records, parquet_tmp)
        os.replace(parquet_tmp, parquet)
    except Exception:
        # JSON is the authoritative recovery log; parquet is a convenience view.
        parquet_tmp = path.with_name("manifest.parquet.tmp")
        parquet_tmp.unlink(missing_ok=True)


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
        "archive_checksum": record.get("archive_checksum"),
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


def _resume_discovery_cursor(
    manifest: Path,
    records: list[dict[str, Any]],
    start: datetime,
    end: datetime,
    step: timedelta,
) -> datetime:
    """Return the first explicitly uncovered chunk (legacy scalar is metadata only)."""
    del records
    coverage = _load_discovery_coverage(manifest)
    for chunk_start, chunk_end in _chunk_intervals(start, end, step):
        if not _interval_is_covered(chunk_start, chunk_end, coverage):
            return chunk_start
    return end


def _discovery_statistics(
    start: datetime,
    end: datetime,
    step: timedelta,
    coverage: list[tuple[datetime, datetime]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    chunks = _chunk_intervals(start, end, step)
    covered_chunks = sum(
        _interval_is_covered(chunk_start, chunk_end, coverage)
        for chunk_start, chunk_end in chunks
    )
    observed: set[datetime] = set()
    slot_seconds = 15 * 60
    for record in records:
        try:
            sensing_time = _parse_utc(record.get("sensing_time"))
        except (TypeError, ValueError):
            continue
        # HRSEVIRI sensing starts are commonly several seconds after the nominal
        # quarter-hour. Associate them with the nearest UTC source slot rather
        # than incorrectly reporting every slightly offset product as missing.
        nominal_seconds = int((sensing_time.timestamp() + slot_seconds / 2) // slot_seconds)
        nominal = datetime.fromtimestamp(nominal_seconds * slot_seconds, tz=UTC)
        if start <= nominal < end:
            observed.add(nominal)

    missing = []
    slot = start
    slot_step = timedelta(seconds=slot_seconds)
    while slot < end:
        slot_end = min(end, slot + slot_step)
        if _interval_is_covered(slot, slot_end, coverage) and slot not in observed:
            missing.append(slot.isoformat())
        slot = slot_end
    return {
        "covered_chunks": int(covered_chunks),
        "remaining_uncovered_chunks": len(chunks) - int(covered_chunks),
        "total_chunks": len(chunks),
        "confirmed_missing_15_minute_source_slots": len(missing),
        "missing_source_timestamps": missing,
        "coverage_intervals": _serialize_discovery_coverage(coverage),
    }


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


def _quarantine_legacy_empty_patch(record: dict[str, Any], processed_root: Path) -> list[Path]:
    """Move a previously published all-NaN patch out of the canonical store."""
    patch = Path(str(record.get("patch_uri", "")))
    if not patch.exists():
        return []
    try:
        legacy = load_patch(patch)
    except Exception:
        return []
    if np.isfinite(legacy).any():
        return []
    quarantine = processed_root / "quarantine" / "all_nan_patches"
    quarantine.mkdir(parents=True, exist_ok=True)
    destination = quarantine / patch.name
    if destination.exists():
        destination = quarantine / f"{patch.stem}-{time.time_ns()}{patch.suffix}"
    shutil.move(str(patch), str(destination))
    moved = [destination]
    sidecar = patch.with_suffix(patch.suffix + ".json")
    if sidecar.exists():
        sidecar_destination = destination.with_suffix(destination.suffix + ".json")
        shutil.move(str(sidecar), str(sidecar_destination))
        moved.append(sidecar_destination)
    record["quarantined_patch_uri"] = str(destination)
    record.pop("patch_uri", None)
    record.pop("processing_checksum", None)
    return moved


def _purge_feature_products(root: Path, product_ids: set[str]) -> None:
    """Remove terminal source gaps from canonical and incremental feature tables."""
    if not product_ids:
        return
    paths = [root / "features.parquet"]
    parts = root / "feature_parts"
    if parts.exists():
        paths.extend(sorted(parts.glob("part-*.parquet")))
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if "product_id" not in frame or not frame["product_id"].astype(str).isin(product_ids).any():
            continue
        filtered = frame.loc[~frame["product_id"].astype(str).isin(product_ids)]
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        filtered.to_parquet(temporary, index=False)
        os.replace(temporary, path)


def _worker_runtime_diagnostics() -> dict[str, Any]:
    """Collect lightweight diagnostics without requiring optional packages."""
    try:
        native_threads = len(os.listdir("/proc/self/task"))
    except OSError:
        native_threads = None
    try:
        import importlib.metadata
        satpy_version = importlib.metadata.version("satpy")
    except Exception:
        satpy_version = None
    dask_scheduler = dask_workers = None
    try:
        import dask
        dask_scheduler = dask.config.get("scheduler", default=None)
        dask_workers = dask.config.get("num_workers", default=None)
    except Exception:
        pass
    threadpools = []
    try:
        from threadpoolctl import threadpool_info
        threadpools = [
            {
                "prefix": info.get("prefix"),
                "internal_api": info.get("internal_api"),
                "num_threads": info.get("num_threads"),
            }
            for info in threadpool_info()
        ]
    except Exception:
        pass
    return {
        "native_threads": native_threads,
        "satpy_version": satpy_version,
        "dask_scheduler": dask_scheduler,
        "dask_workers": dask_workers,
        "threadpools": threadpools,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "DASK_NUM_WORKERS",
            )
            if os.environ.get(name) is not None
        },
    }


def _cached_archive_metadata(
    record: dict[str, Any], archive: Path
) -> tuple[dict[str, Any] | None, str | None]:
    """Return collection metadata when it still describes the local archive."""
    stat = archive.stat()
    if (
        record.get("archive_validation")
        and record.get("archive_checksum")
        and record.get("archive_size") == stat.st_size
        and record.get("archive_mtime_ns") == stat.st_mtime_ns
    ):
        return record["archive_validation"], record["archive_checksum"]
    return None, None


def _process_product(
    record: dict[str, Any], site: SiteConfig, decoder: Callable[..., Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Process one product without touching manifest or shared feature files."""
    worker_started_wall = time.perf_counter()
    worker_started_epoch = time.time_ns()
    worker_started_cpu = time.process_time()
    runtime_before = _worker_runtime_diagnostics()
    decode_started_wall = time.perf_counter()
    decode_started_cpu = time.process_time()
    archive = Path(record.get("archive_uri") or record.get("local_uri"))
    timestamp = datetime.fromisoformat(str(record["sensing_time"]).replace("Z", "+00:00"))
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None
    if threadpool_limits is None:
        channels = decoder(archive, site)
        runtime_during_decode = _worker_runtime_diagnostics()
    else:
        with threadpool_limits(limits=1):
            channels = decoder(archive, site)
            runtime_during_decode = _worker_runtime_diagnostics()
    runtime_after_decode = _worker_runtime_diagnostics()
    decode_wall = time.perf_counter() - decode_started_wall
    decode_cpu = time.process_time() - decode_started_cpu
    # The coordinator supplies the deterministic patch location through the record.
    patch_started_wall = time.perf_counter()
    patch_started_cpu = time.process_time()
    result = preprocess_physical_channels(
        channels, site, str(record["product_id"]), timestamp, str(record["patch_uri"]),
        source_archive_checksum=record.get("archive_checksum"),
    )
    patch = load_patch(result["patch_uri"])
    validate_patch(patch, 12, site.patch_size)
    result["processing_checksum"] = _artifact_checksum(result["patch_uri"])
    patch_wall = time.perf_counter() - patch_started_wall
    patch_cpu = time.process_time() - patch_started_cpu
    feature_started = time.perf_counter()
    feature_cpu_started = time.process_time()
    feature = derive_patch_features(
        patch, result["patch_uri"], site_id=site.site_id,
        timestamp_utc=result["timestamp_utc"], product_id=str(record["product_id"]),
    )
    feature_wall = time.perf_counter() - feature_started
    feature_cpu = time.process_time() - feature_cpu_started
    return (
        {**record, **result, "processing_checksum": result["processing_checksum"]},
        feature,
        {
            "pid": os.getpid(),
            "process_name": __import__("multiprocessing").current_process().name,
            "cpu_count": os.cpu_count(),
            "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "native_threads_before": runtime_before["native_threads"],
            "native_threads_during_decode": runtime_during_decode["native_threads"],
            "native_threads_after_decode": runtime_after_decode["native_threads"],
            "satpy_version": runtime_after_decode["satpy_version"],
            "dask_scheduler": runtime_after_decode["dask_scheduler"],
            "dask_workers": runtime_after_decode["dask_workers"],
            "threadpools": runtime_after_decode["threadpools"],
            "threadpools_during_decode": runtime_during_decode["threadpools"],
            "thread_environment": runtime_after_decode["thread_environment"],
            "worker_start_ns": worker_started_epoch,
            "worker_end_ns": time.time_ns(),
            "worker_wall": time.perf_counter() - worker_started_wall,
            "worker_cpu": time.process_time() - worker_started_cpu,
            "decode_wall": decode_wall,
            "decode_cpu": decode_cpu,
            "patch_wall": patch_wall,
            "patch_cpu": patch_cpu,
            "feature_wall": feature_wall,
            "feature_cpu": feature_cpu,
        },
    )


def _max_interval_concurrency(intervals: list[tuple[int, int]]) -> int:
    events: list[tuple[int, int]] = []
    for start, end in intervals:
        if end > start:
            events.extend(((start, 1), (end, -1)))
    active = maximum = 0
    for _, change in sorted(events, key=lambda event: (event[0], event[1])):
        active += change
        maximum = max(maximum, active)
    return maximum


def _bounded_map(executor, values, function, limit: int):
    """Yield completed futures while keeping at most ``limit`` queued."""
    iterator = iter(values)
    pending = set()
    for _ in range(limit):
        try:
            value = next(iterator)
            future = executor.submit(function, value)
            future.product_id = value.get("product_id") if isinstance(value, dict) else None
            pending.add(future)
        except StopIteration:
            break
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            yield future
            try:
                value = next(iterator)
                future = executor.submit(function, value)
                future.product_id = value.get("product_id") if isinstance(value, dict) else None
                pending.add(future)
            except StopIteration:
                pass


def _write_feature_part(
    root: Path, rows: list[dict[str, Any]], chunk_id: str
) -> Path | None:
    if not rows:
        return None
    parts = root / "feature_parts"
    parts.mkdir(parents=True, exist_ok=True)
    target = parts / f"part-{chunk_id}.parquet"
    temporary = parts / f".{target.name}.tmp-{os.getpid()}"
    pd.DataFrame(rows).drop_duplicates(["product_id"], keep="last").to_parquet(temporary, index=False)
    os.replace(temporary, target)
    return target


def _compact_features(root: Path) -> None:
    parts = root / "feature_parts"
    frames = []
    output = root / "features.parquet"
    if output.exists():
        frames.append(pd.read_parquet(output))
    for part in sorted(parts.glob("part-*.parquet")) if parts.exists() else []:
        try:
            frames.append(pd.read_parquet(part))
        except Exception:
            continue
    if not frames:
        return
    result = pd.concat(frames, ignore_index=True).drop_duplicates(["product_id"], keep="last")
    temporary = root / f".features.parquet.tmp-{os.getpid()}"
    result.to_parquet(temporary, index=False)
    os.replace(temporary, output)


def _copy_tree_verified(
    source: Path, destination: Path, inventory: dict[str, str], prefix: str = ""
) -> None:
    for item in sorted(source.rglob("*")):
        if item.is_file():
            relative = f"{prefix}{item.relative_to(source).as_posix()}"
            target = destination / relative
            atomic_copy(item, str(target))
            inventory[relative] = sha256_file(target)


def _backup_roots(config: SatelliteConfig) -> tuple[Path, Path, Path]:
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
    return raw, processed, backup


def _artifact_files(path: Path) -> list[Path]:
    files = [path] if path.is_file() else [item for item in sorted(path.rglob("*")) if item.is_file()]
    sidecar = path.with_suffix(path.suffix + ".json")
    if sidecar.is_file() and sidecar not in files:
        files.append(sidecar)
    return files


def backup_checkpoint(
    config: SatelliteConfig,
    site: SiteConfig,
    batch_id: str,
    records: list[dict[str, Any]],
    *,
    extra_artifacts: list[Path] | None = None,
) -> dict[str, Any]:
    """Incrementally back up one completed chunk and publish a cumulative marker."""
    raw, processed, backup = _backup_roots(config)
    marker_path = backup / "BACKUP_COMPLETE.json"
    inventory: dict[str, str] = {}
    if marker_path.exists():
        prior = json.loads(marker_path.read_text(encoding="utf-8"))
        inventory.update(prior.get("checksums", {}))
    touched: set[str] = set()

    sources: list[Path] = []
    for record in records:
        patch = Path(str(record.get("patch_uri", ""))).resolve()
        if patch != processed and processed not in patch.parents:
            raise ValueError(f"patch is outside configured processed root: {patch}")
        patch_files = _artifact_files(patch)
        if not patch_files:
            raise ValueError(f"processed patch has no files: {patch}")
        sources.extend(patch_files)
    for artifact in extra_artifacts or []:
        source = artifact.resolve()
        if source != processed and processed not in source.parents:
            raise ValueError(f"artifact is outside configured processed root: {source}")
        sources.extend(_artifact_files(source))

    for source in dict.fromkeys(sources):
        relative = f"processed/{source.relative_to(processed).as_posix()}"
        target = backup / relative
        atomic_copy(source, str(target))
        inventory[relative] = sha256_file(target)
        touched.add(relative)

    manifest, events = _manifest_paths(config)
    for source, name in (
        (manifest, "raw/manifest.json"),
        (events, "raw/manifest.events.jsonl"),
        (raw / "manifest.parquet", "raw/manifest.parquet"),
    ):
        if source.exists():
            target = backup / name
            atomic_copy(source, str(target))
            inventory[name] = sha256_file(target)
            touched.add(name)

    metadata = backup / "batch-config.json"
    write_json(
        str(metadata),
        {"batch_id": batch_id, "site": site.__dict__, "config": config.metadata(site)},
    )
    inventory["batch-config.json"] = sha256_file(metadata)
    touched.add("batch-config.json")

    # Previously inventoried files were verified before their marker was
    # published. Verify only this checkpoint's new/updated bytes here.
    for relative in touched:
        if sha256_file(backup / relative) != inventory[relative]:
            raise ValueError(f"backup checksum mismatch: {relative}")
    marker = {
        "batch_id": batch_id,
        "completed_at_utc": UTC_NOW(),
        "mode": "incremental-checkpoint",
        "checksums": inventory,
    }
    write_json(str(marker_path), marker)
    inventory_checksum = hashlib.sha256(
        json.dumps(inventory, sort_keys=True).encode()
    ).hexdigest()
    for record in records:
        if record.get("processing_status") == "processed":
            record["backup_status"] = "verified"
            record["backup_checksum"] = inventory_checksum
    return marker


def backup_batch(
    config: SatelliteConfig, site: SiteConfig, batch_id: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    raw, processed, backup = _backup_roots(config)
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
        # Every file in the patch store and its sidecar must be present in the
        # cumulative verified inventory before deleting the native archive.
        for item in _artifact_files(patch):
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
    """Run a bounded download/process pipeline and coordinator-owned finalization."""
    started = time.perf_counter()
    timings = {
        "discovery": 0.0, "download_validation": 0.0, "queue_wait": 0.0,
        "processing": 0.0, "feature_part_writing": 0.0, "feature_compaction": 0.0,
        "processing_future_latency": 0.0, "processing_future_wait": 0.0,
        "processing_future_wait_max": 0.0,
        "processing_worker_wall": 0.0, "processing_worker_cpu": 0.0,
        "processing_max_concurrency": 0.0,
        "collection_download": 0.0,
        "archive_validation_checksum": 0.0,
        "native_decoding": 0.0,
        "patch_preprocessing_writing": 0.0,
        "feature_extraction": 0.0,
        "backup_verification": 0.0,
        "native_deletion": 0.0,
        "total_elapsed": 0.0,
    }
    config.validate_collection()
    start, end = config.ensure_utc(start), config.ensure_utc(end)
    if end <= start:
        raise ValueError("end must be after start")
    manifest_path, events = _manifest_paths(config)
    client = client or EumetsatClient(config)
    existing = _records(config)
    if manifest_path.exists() and not _has_explicit_discovery_coverage(manifest_path):
        # Persist the legacy migration before contacting the provider. The old
        # high-water scalar remains represented only by the newly derived
        # informational checkpoint and grants no trusted discovery coverage.
        _write_manifest(
            manifest_path,
            existing,
            site,
            config,
            discovery_coverage_utc=[],
        )
    by_id = {str(r["product_id"]): r for r in existing}
    interval_ids: set[str] = set()
    processed_root = _local(config.processed_uri) / "physical_patches"
    feature_root = _local(config.processed_uri)
    custom_decoder = decoder is not decode_native
    if custom_decoder and config.processing_workers > 1:
        # Thread execution keeps existing injected test decoders usable; real
        # Satpy uses the process pool below. A non-picklable decoder cannot be
        # sent to worker processes, so it is explicitly constrained to threads.
        try:
            import pickle
            pickle.dumps(decoder)
            process_executor = ProcessPoolExecutor(max_workers=config.processing_workers)
        except Exception:
            process_executor = ThreadPoolExecutor(max_workers=config.processing_workers)
    else:
        process_executor = ThreadPoolExecutor(max_workers=1) if custom_decoder else ProcessPoolExecutor(max_workers=config.processing_workers)
    batch_id = hashlib.sha256(f"{start.isoformat()}:{end.isoformat()}:{site.site_id}".encode()).hexdigest()[:16]
    counts = {
        "submitted": 0,
        "completed": 0,
        "failed": 0,
        "unavailable": 0,
        "downloaded": 0,
        "processed": 0,
    }
    all_records = list(by_id.values())
    worker_intervals: list[tuple[int, int]] = []
    process_wait_total = 0.0
    process_wait_max = 0.0
    process_latency_total = 0.0
    worker_wall_total = 0.0
    worker_cpu_total = 0.0
    worker_stats: dict[str, dict[str, Any]] = {}
    step = timedelta(hours=chunk_hours or config.collection_chunk_hours)
    discovery_coverage = _load_discovery_coverage(manifest_path)
    discovery_chunks = [
        chunk
        for chunk in _chunk_intervals(start, end, step)
        if not _interval_is_covered(chunk[0], chunk[1], discovery_coverage)
    ]
    chunk_index = 0
    recovery_records = []
    for record in existing:
        try:
            timestamp = datetime.fromisoformat(
                str(record.get("sensing_time", "")).replace("Z", "+00:00")
            ).astimezone(UTC)
        except (ValueError, TypeError):
            continue
        if not start <= timestamp < end:
            continue
        if record.get("processing_status") == "unavailable_source":
            interval_ids.add(str(record["product_id"]))
        elif record.get("processing_status") not in {
            "processed", "deleted", "unavailable_source", "error", "invalid_archive"
        }:
            archive = Path(record.get("archive_uri") or record.get("local_uri", ""))
            if archive.is_file():
                recovery_records.append(record)
                interval_ids.add(str(record["product_id"]))
    unavailable_ids: set[str] = set()
    try:
        while chunk_index < len(discovery_chunks) or recovery_records:
            discovery_enabled = chunk_index < len(discovery_chunks)
            if discovery_enabled:
                cursor, chunk_end = discovery_chunks[chunk_index]
            else:
                cursor = chunk_end = start
            items = []
            if discovery_enabled:
                discovery_started = time.perf_counter()
                items = discover_products(
                    cursor,
                    chunk_end,
                    site,
                    config,
                    client=client,
                    existing_records=list(by_id.values()),
                )
                timings["discovery"] += time.perf_counter() - discovery_started
            feature_rows: list[dict[str, Any]] = []
            process_pending = set()
            processing_scheduled_ids: set[str] = set()
            chunk_processed_ids: set[str] = set()
            chunk_quarantined_artifacts: list[Path] = []
            process_function = partial(_process_product, site=site, decoder=decoder)
            process_limit = config.processing_workers + config.pipeline_queue_size

            def consume_process(future) -> None:
                nonlocal process_wait_total, process_wait_max, process_latency_total, worker_wall_total, worker_cpu_total
                processing_scheduled_ids.discard(
                    str(getattr(future, "product_id", ""))
                )
                future_latency = None
                if hasattr(future, "submitted_at"):
                    future_latency = time.perf_counter() - future.submitted_at
                try:
                    updated, feature, metrics = future.result()
                    by_id[str(updated["product_id"])] = updated
                    _transition(updated, "processed", events)
                    feature_rows.append(feature)
                    worker_intervals.append((metrics["worker_start_ns"], metrics["worker_end_ns"]))
                    worker_wall_total += metrics["worker_wall"]
                    worker_cpu_total += metrics["worker_cpu"]
                    pid = str(metrics["pid"])
                    stats = worker_stats.setdefault(pid, {
                        "pid": metrics["pid"],
                        "process_name": metrics["process_name"],
                        "cpu_count": metrics["cpu_count"],
                        "affinity_cpus": metrics["affinity_cpus"],
                        "native_threads_before": metrics["native_threads_before"],
                        "native_threads_during_decode": metrics["native_threads_during_decode"],
                        "native_threads_after_decode": metrics["native_threads_after_decode"],
                        "satpy_version": metrics["satpy_version"],
                        "dask_scheduler": metrics["dask_scheduler"],
                        "dask_workers": metrics["dask_workers"],
                        "threadpools": metrics["threadpools"],
                        "threadpools_during_decode": metrics["threadpools_during_decode"],
                        "thread_environment": metrics["thread_environment"],
                        "tasks": 0,
                        "wall_seconds": 0.0,
                        "cpu_seconds": 0.0,
                    })
                    stats["tasks"] += 1
                    stats["wall_seconds"] += metrics["worker_wall"]
                    stats["cpu_seconds"] += metrics["worker_cpu"]
                    if future_latency is not None:
                        process_latency_total += future_latency
                        process_wait = max(0.0, future_latency - metrics["worker_wall"])
                        process_wait_total += process_wait
                        process_wait_max = max(process_wait_max, process_wait)
                    timings["native_decoding"] += metrics["decode_wall"]
                    timings["patch_preprocessing_writing"] += metrics["patch_wall"]
                    timings["feature_extraction"] += metrics["feature_wall"]
                    counts["completed"] += 1
                    counts["processed"] += 1
                    chunk_processed_ids.add(str(updated["product_id"]))
                except EmptySatelliteProductError:
                    product_id = str(getattr(future, "product_id", ""))
                    failed_record = by_id.get(product_id)
                    if failed_record:
                        chunk_quarantined_artifacts.extend(
                            _quarantine_legacy_empty_patch(failed_record, feature_root)
                        )
                        if int(failed_record.get("attempt_count", 0)) < 1:
                            _transition(
                                failed_record,
                                "failed",
                                events,
                                EmptySatelliteProductError.reason,
                            )
                            submit_processing(
                                failed_record,
                                force=bool(getattr(future, "force", False)),
                            )
                        else:
                            failed_record["unavailable_reason"] = (
                                EmptySatelliteProductError.reason
                            )
                            _transition(
                                failed_record,
                                "unavailable_source",
                                events,
                                EmptySatelliteProductError.reason,
                            )
                            unavailable_ids.add(product_id)
                except Exception as exc:
                    product_id = str(getattr(future, "product_id", ""))
                    failed_record = by_id.get(product_id)
                    if failed_record:
                        _transition(failed_record, "failed", events, str(exc))
                    counts["failed"] += 1

            def submit_processing(record: dict[str, Any], *, force: bool = False) -> None:
                product_id = str(record["product_id"])
                timestamp = datetime.fromisoformat(str(record.get("sensing_time", "")).replace("Z", "+00:00"))
                if (
                    product_id in processing_scheduled_ids
                    or (not force and not (cursor <= timestamp < chunk_end))
                    or record.get("processing_status")
                    in {"processed", "deleted", "unavailable_source", "error", "invalid_archive"}
                ):
                    return
                archive = Path(record.get("archive_uri") or record.get("local_uri"))
                if not archive.is_file():
                    return
                record["patch_uri"] = str(processed_root / f"{record['product_id']}.zarr")
                wait_started = time.perf_counter()
                while len(process_pending) >= process_limit:
                    done, still_pending = wait(process_pending, return_when=FIRST_COMPLETED)
                    process_pending.clear()
                    process_pending.update(still_pending)
                    for done_future in done:
                        consume_process(done_future)
                future = process_executor.submit(process_function, record)
                future.product_id = record["product_id"]
                future.force = force
                future.submitted_at = time.perf_counter()
                process_pending.add(future)
                processing_scheduled_ids.add(product_id)
                timings["queue_wait"] += time.perf_counter() - wait_started
                counts["submitted"] += 1

            process_started = time.perf_counter()
            download_started = time.perf_counter()
            for recovery_record in recovery_records:
                submit_processing(recovery_record, force=True)
            recovery_records = []
            with ThreadPoolExecutor(max_workers=config.download_workers) as download_executor:
                download_futures = _bounded_map(
                    download_executor, items,
                    lambda item: download_product(item, config, client=client, validate_archives=True),
                    config.download_workers + config.pipeline_queue_size,
                )
                for future in download_futures:
                    record = future.result()
                    by_id[str(record["product_id"])] = record
                    interval_ids.add(str(record["product_id"]))
                    counts["downloaded"] += int(record.get("processing_status") in {"downloaded", "already_present", "processed", "failed"})
                    if record.get("processing_status") in {"error", "invalid_archive"}:
                        counts["failed"] += 1
                    submit_processing(record)
                    done, still_pending = wait(process_pending, timeout=0, return_when=FIRST_COMPLETED) if process_pending else (set(), set())
                    process_pending = set(still_pending)
                    for done_future in done:
                        consume_process(done_future)
            timings["download_validation"] += time.perf_counter() - download_started
            timings["collection_download"] = timings["download_validation"]
            while process_pending:
                done, still_pending = wait(process_pending, return_when=FIRST_COMPLETED)
                process_pending = set(still_pending)
                for future in done:
                    consume_process(future)
            timings["processing"] += time.perf_counter() - process_started
            _purge_feature_products(feature_root, unavailable_ids)
            unavailable_ids.clear()
            part_started = time.perf_counter()
            feature_part = _write_feature_part(
                feature_root, feature_rows, cursor.strftime("%Y%m%dT%H%M%S")
            )
            timings["feature_part_writing"] += time.perf_counter() - part_started
            checkpoint_records = [
                by_id[product_id]
                for product_id in chunk_processed_ids
                if product_id in by_id
                and by_id[product_id].get("processing_status") == "processed"
            ]
            ordered_records = sorted(
                by_id.values(),
                key=lambda x: (str(x.get("sensing_time", "")), str(x["product_id"])),
            )
            if discovery_enabled:
                discovery_coverage = _merge_discovery_coverage(
                    [*discovery_coverage, (cursor, chunk_end)]
                )
            _write_manifest(
                manifest_path,
                ordered_records,
                site,
                config,
                discovery_coverage_utc=discovery_coverage,
            )
            if config.backup_uri and (checkpoint_records or discovery_enabled):
                backup_started = time.perf_counter()
                backup_checkpoint(
                    config, site, f"{batch_id}-checkpoint-{cursor:%Y%m%dT%H%M%S}",
                    checkpoint_records,
                    extra_artifacts=(
                        ([feature_part] if feature_part else [])
                        + chunk_quarantined_artifacts
                    ),
                )
                timings["backup_verification"] += time.perf_counter() - backup_started
                _write_manifest(
                    manifest_path,
                    ordered_records,
                    site,
                    config,
                    discovery_coverage_utc=discovery_coverage,
                )
                if delete_native:
                    deletion_started = time.perf_counter()
                    delete_verified_native(config, checkpoint_records, dry_run=dry_run)
                    timings["native_deletion"] += time.perf_counter() - deletion_started
                    _write_manifest(
                        manifest_path,
                        ordered_records,
                        site,
                        config,
                        discovery_coverage_utc=discovery_coverage,
                    )
            if discovery_enabled:
                chunk_index += 1
        process_executor.shutdown(wait=True)
        timings["processing_future_latency"] = process_latency_total
        timings["processing_future_wait"] = process_wait_total
        timings["processing_future_wait_max"] = process_wait_max
        timings["processing_worker_wall"] = worker_wall_total
        timings["processing_worker_cpu"] = worker_cpu_total
        timings["processing_max_concurrency"] = float(_max_interval_concurrency(worker_intervals))
        compact_started = time.perf_counter()
        _compact_features(feature_root)
        timings["feature_compaction"] = time.perf_counter() - compact_started
        all_records = list(by_id.values())
        if config.backup_uri and any(r.get("processing_status") == "processed" for r in all_records):
            stage_started = time.perf_counter()
            backup_batch(config, site, batch_id, all_records)
            timings["backup_verification"] += time.perf_counter() - stage_started
            _write_manifest(manifest_path, all_records, site, config)
        if delete_native:
            stage_started = time.perf_counter()
            delete_verified_native(config, all_records, dry_run=dry_run)
            timings["native_deletion"] += time.perf_counter() - stage_started
            _write_manifest(manifest_path, all_records, site, config)
    finally:
        try:
            process_executor.shutdown(wait=True)
        except Exception:
            pass
        timings["total_elapsed"] = time.perf_counter() - started
    interval_records = [r for r in by_id.values() if str(r.get("product_id")) in interval_ids]
    failed = [r for r in interval_records if r.get("processing_status") in {"failed", "error", "invalid_archive"}]
    unavailable = [
        r for r in interval_records if r.get("processing_status") == "unavailable_source"
    ]
    counts["failed"] = len(failed)
    counts["unavailable"] = len(unavailable)
    discovery_stats = _discovery_statistics(
        start, end, step, discovery_coverage, list(by_id.values())
    )
    counts.update(
        {
            "covered_chunks": discovery_stats["covered_chunks"],
            "remaining_uncovered_chunks": discovery_stats["remaining_uncovered_chunks"],
            "confirmed_missing_source_slots": discovery_stats[
                "confirmed_missing_15_minute_source_slots"
            ],
        }
    )
    return {
        "batch_id": batch_id,
        "products": sorted(interval_records, key=lambda x: (str(x.get("sensing_time", "")), str(x["product_id"]))),
        "failed_products": failed,
        "unavailable_products": unavailable,
        "eligible_deletions": [
            r["product_id"]
            for r in all_records
            if r.get("processing_status") == "processed" and r.get("backup_status") == "verified"
        ],
        "timings": timings,
        "workers": {"download": config.download_workers, "processing": config.processing_workers},
        "processing_executor": {
            "type": type(process_executor).__name__,
            "configured_workers": config.processing_workers,
            "observed_workers": len(worker_stats),
            "workers": sorted(worker_stats.values(), key=lambda item: item["pid"]),
        },
        "processing_worker_stats": sorted(worker_stats.values(), key=lambda item: item["pid"]),
        "counts": counts,
        "discovery": discovery_stats,
        "interval": {"start": start.isoformat(), "end": end.isoformat()},
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
