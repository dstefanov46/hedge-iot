from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from aurora.satellite.config import PRODUCT_COLLECTION, SatelliteConfig, SiteConfig
from aurora.satellite.storage import (
    atomic_copy,
    read_json,
    sha256_file,
    uri_exists,
    write_json,
)

LOGGER = logging.getLogger(__name__)


class ProductClient(Protocol):
    def search(
        self,
        collection: str,
        start: datetime,
        end: datetime,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict[str, Any]]: ...
    def download(self, product: dict[str, Any], destination: Path) -> None: ...


def discover_products(
    start: datetime, end: datetime, site: SiteConfig, config: SatelliteConfig,
    client: ProductClient | None = None, bbox: tuple[float, float, float, float] | None = None,
    existing_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Discover products and construct work records without writing shared state."""
    start, end = config.ensure_utc(start), config.ensure_utc(end)
    client = client or EumetsatClient(config)
    existing = {str(r.get("product_id")): r for r in (existing_records or [])}
    result = []
    for product in client.search(config.collection or PRODUCT_COLLECTION, start, end, bbox):
        product_id = str(product["id"])
        sensing = product.get("sensing_time")
        sensing_date = sensing.astimezone(UTC) if isinstance(sensing, datetime) else start
        root = config.raw_uri.rstrip("/")
        destination = Path(f"{root}/{sensing_date:%Y/%m/%d}/{product_id}.zip")
        for alternate in (f"{root}/{sensing_date:%Y/%m/%d}/{product_id}.nat",
                          f"{root}/{sensing_date:%Y/%m/%d}/{product_id}.native"):
            if not destination.exists() and Path(alternate).exists():
                destination = Path(alternate)
        record = {
            "product_id": product_id, "sensing_time": str(product.get("sensing_time") or ""),
            "satellite": str(product.get("satellite") or "MSG"), "collection": config.collection,
            "local_uri": str(destination), "archive_uri": str(destination), "checksum": None,
            "checksum_algorithm": product.get("checksum_algorithm", "sha256"),
            "processing_status": "downloaded", "error_state": None,
        }
        prior = existing.get(product_id)
        if prior:
            # A discovered product may already be fully processed and have its
            # native archive intentionally deleted. Preserve the complete
            # coordinator-owned record so rediscovery cannot discard quality,
            # provenance, site, or recovery metadata.
            record.update(prior)
            record.update(
                {
                    "product_id": product_id,
                    "sensing_time": str(product.get("sensing_time") or ""),
                    "satellite": str(product.get("satellite") or prior.get("satellite") or "MSG"),
                    "collection": config.collection,
                }
            )
        result.append({"record": record, "product": product})
    return result


def download_product(
    item: dict[str, Any], config: SatelliteConfig, *, client: ProductClient,
    validate_archives: bool = True,
) -> dict[str, Any]:
    """Download and validate one product; publication is atomic and last."""
    record, product = dict(item["record"]), item["product"]
    if (
        record.get("processing_status") == "processed"
        and record.get("backup_status") == "verified"
        and record.get("deletion_status") == "deleted"
    ):
        # This is the terminal state for native-deletion runs. The missing
        # archive is intentional and must not trigger another provider fetch.
        return record
    if record.get("processing_status") == "unavailable_source":
        # The validated native archive is intentionally retained as evidence,
        # but a documented source gap is terminal on later resumptions.
        return record
    destination = Path(record["archive_uri"])
    if destination.is_file():
        record["processing_status"] = record.get("processing_status") if record.get("processing_status") in {"processed", "failed", "deleted"} else "already_present"
        record["archive_checksum"] = sha256_file(destination)
    else:
        expected = product.get("checksum") or product.get("sha256")
        algorithm = str(record.get("checksum_algorithm", "sha256")).lower()
        with tempfile.NamedTemporaryFile(suffix=".native", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            for attempt in range(config.retry_attempts):
                try:
                    client.download(product, temporary)
                    break
                except Exception:
                    if attempt + 1 == config.retry_attempts:
                        raise
                    time.sleep(min(2**attempt, 16))
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise ValueError(f"Downloaded empty product {record['product_id']}")
            digest = hashlib.md5(temporary.read_bytes()).hexdigest() if algorithm == "md5" else sha256_file(temporary)
            if expected and digest.lower() != str(expected).lower():
                raise ValueError(f"Checksum mismatch for product {record['product_id']}")
            if validate_archives:
                validation = validate_native_archive(temporary)
                record["archive_validation"] = {**validation, "archive_uri": str(destination)}
            atomic_copy(temporary, str(destination))
            record["checksum"] = digest
            record["archive_checksum"] = sha256_file(destination)
        except Exception as exc:
            record["processing_status"] = "invalid_archive" if "ZIP" in str(exc) or "payload" in str(exc) else "error"
            record["error_state"] = str(exc)
        finally:
            temporary.unlink(missing_ok=True)
    if destination.is_file():
        record["archive_checksum"] = record.get("archive_checksum") or sha256_file(destination)
        stat = destination.stat()
        record["archive_size"], record["archive_mtime_ns"] = stat.st_size, stat.st_mtime_ns
        if validate_archives and not record.get("archive_validation"):
            try:
                record["archive_validation"] = validate_native_archive(destination)
            except Exception as exc:
                record["processing_status"], record["error_state"] = "invalid_archive", str(exc)
    return record


def validate_native_archive(path: str | Path) -> dict[str, Any]:
    """Validate that *path* is a readable HRSEVIRI ZIP with one NAT payload."""
    source = Path(path)
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError(f"Satellite product is not a readable ZIP: {source}")
    with zipfile.ZipFile(source) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Satellite ZIP has a corrupt member: {source}")
        nat = [name for name in archive.namelist() if name.lower().endswith(".nat")]
        if len(nat) != 1:
            raise ValueError(f"Expected exactly one .nat payload, found {len(nat)}: {source}")
        info = archive.getinfo(nat[0])
        if info.file_size == 0:
            raise ValueError(f"Satellite .nat payload is empty: {source}")
        return {"archive_uri": str(source), "nat_member": nat[0], "nat_size": info.file_size}


class EumetsatClient:
    """Small adapter around EUMDAC; imports the optional SDK only when used."""

    def __init__(self, config: SatelliteConfig):
        config.validate_credentials()
        self.config = config
        try:
            import eumdac
        except ImportError as exc:
            raise RuntimeError("Install the satellite optional dependencies to use EUMDAC") from exc
        self._eumdac = eumdac

    def _token(self):
        return self._eumdac.AccessToken(
            (self.config.api_key, self.config.api_secret),
            validity=self.config.token_validity_seconds,
        )

    @staticmethod
    def _exception_chain(exc: BaseException):
        """Yield an SDK exception and its wrapped causes without looping."""
        pending = [exc]
        seen: set[int] = set()
        while pending:
            current = pending.pop(0)
            if id(current) in seen:
                continue
            seen.add(id(current))
            yield current
            cause = getattr(current, "__cause__", None)
            context = getattr(current, "__context__", None)
            if isinstance(cause, BaseException):
                pending.append(cause)
            if isinstance(context, BaseException) and context is not cause:
                pending.append(context)

    @staticmethod
    def _normalise_failure_url(value: Any) -> str | None:
        if value is None:
            return None
        geturl = getattr(value, "geturl", None)
        if callable(geturl):
            value = geturl()
        return str(value) if value else None

    @classmethod
    def _failure_details(
        cls, exc: BaseException
    ) -> tuple[int | None, str | None]:
        """Recover HTTP metadata hidden by EUMDAC's wrapper exceptions."""
        fallback_status: int | None = None
        fallback_url: str | None = None
        for current in cls._exception_chain(exc):
            response = getattr(current, "response", None)
            request = getattr(current, "request", None) or getattr(
                response, "request", None
            )
            status = getattr(response, "status_code", None)
            url = cls._normalise_failure_url(
                getattr(response, "url", None) or getattr(request, "url", None)
            )

            extra_info = getattr(current, "extra_info", None)
            if isinstance(extra_info, dict):
                if not isinstance(status, int):
                    extra_status = extra_info.get("status")
                    if isinstance(extra_status, int):
                        status = extra_status
                if not url:
                    url = cls._normalise_failure_url(extra_info.get("url"))

            if isinstance(status, int) and url:
                return status, url
            if isinstance(status, int) and fallback_status is None:
                fallback_status = status
            if url and fallback_url is None:
                fallback_url = url
        return fallback_status, fallback_url

    @staticmethod
    def _is_token_url(url: str | None) -> bool:
        if not url:
            return False
        return urlsplit(url).path.rstrip("/") == "/token"

    @classmethod
    def _is_retryable_discovery_failure(cls, exc: BaseException) -> bool:
        try:
            from requests.exceptions import ConnectionError as RequestsConnectionError
            from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError
            from requests.exceptions import RetryError as RequestsRetryError
            from requests.exceptions import Timeout as RequestsTimeout

            request_failures: tuple[type[BaseException], ...] = (
                RequestsConnectionError,
                RequestsJSONDecodeError,
                RequestsRetryError,
                RequestsTimeout,
            )
        except ImportError:
            request_failures = ()
        wrapped_failures = (
            json.JSONDecodeError,
            ConnectionError,
            TimeoutError,
            *request_failures,
        )
        if any(
            isinstance(current, wrapped_failures)
            for current in cls._exception_chain(exc)
        ):
            return True
        status, url = cls._failure_details(exc)
        if status is not None:
            return status in {408, 429} or status >= 500 or (
                status == 404 and cls._is_token_url(url)
            )
        return False

    def _search_once(self, collection, start, end, bbox=None):
        # Both objects are deliberately attempt-scoped. A failed token must not
        # poison the next catalogue attempt through cached SDK state.
        datastore = self._eumdac.DataStore(self._token())
        collection_obj = datastore.get_collection(collection)
        query = {"dtstart": start, "dtend": end}
        if bbox is not None:
            query["bbox"] = bbox
        products = []
        for product in collection_obj.search(**query):
            sensing_time = product.sensing_start
            if sensing_time.tzinfo is None or sensing_time.utcoffset() is None:
                sensing_time = sensing_time.replace(tzinfo=UTC)
            else:
                sensing_time = sensing_time.astimezone(UTC)
            products.append(
                {
                    # EUMDAC 3.x exposes the product identifier through __str__,
                    # not a public ``id`` attribute.
                    "id": str(product),
                    "collection": collection,
                    "sensing_time": sensing_time,
                    "satellite": "MSG",
                    "checksum": getattr(product, "md5", None),
                    "checksum_algorithm": "md5",
                }
            )
        return products

    def search(self, collection, start, end, bbox=None):
        for attempt in range(1, self.config.discovery_retry_attempts + 1):
            try:
                return self._search_once(collection, start, end, bbox)
            except Exception as exc:
                status, url = self._failure_details(exc)
                retryable = self._is_retryable_discovery_failure(exc)
                if not retryable or attempt == self.config.discovery_retry_attempts:
                    if retryable:
                        LOGGER.error(
                            "EUMETSAT discovery failed after %d attempts (status=%s endpoint=%s)",
                            attempt,
                            status if status is not None else "connection",
                            urlsplit(url).path if url else "unknown",
                        )
                    raise
                delay = min(
                    2 ** (attempt - 1),
                    self.config.discovery_retry_max_backoff_seconds,
                )
                LOGGER.warning(
                    "Retrying EUMETSAT discovery (status=%s endpoint=%s attempt=%d/%d "
                    "delay=%.1fs)",
                    status if status is not None else "connection",
                    urlsplit(url).path if url else "unknown",
                    attempt,
                    self.config.discovery_retry_attempts,
                    delay,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def download(self, product, destination):
        datastore = self._eumdac.DataStore(self._token())
        item = datastore.get_product(product["collection"], product["id"])
        with item.open() as source, open(destination, "wb") as target:
            target.write(source.read())


def download_products(
    start: datetime,
    end: datetime,
    site: SiteConfig,
    config: SatelliteConfig,
    client: ProductClient | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    validate_archives: bool = False,
    timings: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    start, end = config.ensure_utc(start), config.ensure_utc(end)
    if end <= start:
        raise ValueError("end must be after start")
    client = client or EumetsatClient(config)
    products = client.search(config.collection or PRODUCT_COLLECTION, start, end, bbox)
    root = config.raw_uri.rstrip("/")
    manifest_uri = f"{root}/manifest.json"
    existing_records: list[dict[str, Any]] = []
    if uri_exists(manifest_uri):
        existing_payload = read_json(manifest_uri)
        existing_records = list(existing_payload.get("products", []))
    existing_by_id = {str(record.get("product_id")): record for record in existing_records}
    manifest: list[dict[str, Any]] = []
    for product in products:
        product_id = str(product["id"])
        sensing = product.get("sensing_time")
        sensing_date = sensing.astimezone(UTC) if isinstance(sensing, datetime) else start
        destination = f"{root}/{sensing_date:%Y/%m/%d}/{product_id}.zip"
        archive_nat_destination = f"{root}/{sensing_date:%Y/%m/%d}/{product_id}.nat"
        legacy_destination = f"{root}/{sensing_date:%Y/%m/%d}/{product_id}.native"
        if not uri_exists(destination):
            if uri_exists(archive_nat_destination):
                destination = archive_nat_destination
            elif uri_exists(legacy_destination):
                destination = legacy_destination
        record = {
            "product_id": product_id,
            "sensing_time": str(product.get("sensing_time") or ""),
            "satellite": str(product.get("satellite") or "MSG"),
            "collection": config.collection,
            "local_uri": destination,
            "archive_uri": destination,
            "checksum": None,
            "checksum_algorithm": product.get("checksum_algorithm", "sha256"),
            "processing_status": "downloaded",
            "error_state": None,
        }
        # Acquisition reruns must not erase completed processing/backup/deletion
        # state from the materialized manifest.
        prior = existing_by_id.get(product_id)
        if prior:
            for key in (
                "processing_status", "processing_checksum", "backup_status",
                "backup_checksum", "deletion_status", "attempt_count",
                "last_error", "updated_at_utc", "patch_uri", "channel_names",
                "archive_checksum", "archive_validation", "archive_size",
                "archive_mtime_ns",
            ):
                if key in prior:
                    record[key] = prior[key]
            if prior.get("deletion_status") == "deleted":
                # A successful deletion is terminal for acquisition reruns: do
                # not silently redownload the native input.
                manifest.append(record)
                continue
        expected_checksum = product.get("checksum") or product.get("sha256")
        downloaded_sha256: str | None = None
        if uri_exists(destination):
            if record.get("processing_status") not in {"processed", "failed"}:
                record["processing_status"] = "already_present"
            record["checksum"] = expected_checksum
        else:
            with __import__("tempfile").NamedTemporaryFile(suffix=".native", delete=False) as tmp:
                temporary = Path(tmp.name)
            try:
                for attempt in range(config.retry_attempts):
                    try:
                        client.download(product, temporary)
                        break
                    except Exception:
                        if attempt + 1 == config.retry_attempts:
                            raise
                        time.sleep(min(2**attempt, 16))
                if temporary.stat().st_size == 0:
                    raise ValueError(f"Downloaded empty product {product_id}")
                if record["checksum_algorithm"].lower() == "md5":
                    digest = hashlib.md5(temporary.read_bytes()).hexdigest()
                else:
                    record["checksum_algorithm"] = "sha256"
                    digest = sha256_file(temporary)
                    downloaded_sha256 = digest
                record["checksum"] = digest
                if (
                    expected_checksum
                    and digest.lower() != str(expected_checksum).lower()
                ):
                    raise ValueError(f"Checksum mismatch for product {product_id}")
                atomic_copy(temporary, destination)
            except Exception as exc:
                record["processing_status"] = "error"
                record["error_state"] = str(exc)
            finally:
                temporary.unlink(missing_ok=True)
        archive_stage_started = time.perf_counter()
        destination_path = Path(destination)
        if destination_path.is_file():
            if record["checksum"] is None:
                record["checksum"] = sha256_file(destination_path)
            # Keep a content checksum specifically for the local archive.  The
            # provider checksum may be MD5 and is not suitable for processing
            # provenance or unchanged-file checks.
            record["archive_checksum"] = downloaded_sha256 or sha256_file(destination_path)
            record["archive_size"] = destination_path.stat().st_size
            record["archive_mtime_ns"] = destination_path.stat().st_mtime_ns
            if validate_archives:
                try:
                    record["archive_validation"] = validate_native_archive(destination_path)
                except Exception as exc:
                    record["processing_status"] = "invalid_archive"
                    record["error_state"] = str(exc)
        if timings is not None:
            timings["archive_validation_checksum"] = timings.get(
                "archive_validation_checksum", 0.0
            ) + time.perf_counter() - archive_stage_started
        manifest.append(record)
    merged_by_id = {str(record["product_id"]): record for record in existing_records}
    merged_by_id.update({str(record["product_id"]): record for record in manifest})
    merged_manifest = sorted(
        merged_by_id.values(),
        key=lambda record: (
            str(record.get("sensing_time", "")),
            str(record["product_id"]),
        ),
    )
    validate_manifest(merged_manifest)
    write_json(
        manifest_uri,
        {
            "site": site.__dict__,
            "collection_run": config.metadata(site),
            "products": merged_manifest,
        },
    )
    if "://" not in root:
        manifest_to_parquet(merged_manifest, Path(root) / "manifest.parquet")
    return manifest


def collect_in_chunks(
    start: datetime,
    end: datetime,
    site: SiteConfig,
    config: SatelliteConfig,
    client: ProductClient | None = None,
    chunk_hours: int | None = None,
    validate_archives: bool = True,
    timings: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Collect a period in bounded, restartable chunks.

    Each call merges into the same manifest keyed by product ID, so rerunning an
    interrupted chunk does not create duplicate records or downloads.
    """
    config.validate_collection()
    start, end = config.ensure_utc(start), config.ensure_utc(end)
    step = __import__("datetime").timedelta(hours=chunk_hours or config.collection_chunk_hours)
    cursor = start
    collected: list[dict[str, Any]] = []
    while cursor < end:
        chunk_end = min(cursor + step, end)
        collected.extend(download_products(cursor, chunk_end, site, config, client=client,
                                           validate_archives=validate_archives,
                                           timings=timings))
        cursor = chunk_end
    return collected


def coverage_report(manifest: list[dict[str, Any]], expected_timestamps: Any) -> dict[str, Any]:
    """Compare successful products with a canonical 15-minute timestamp grid."""
    import pandas as pd

    expected = pd.to_datetime(expected_timestamps, utc=True)
    successful = [
        r for r in manifest if r.get("processing_status") in {"downloaded", "already_present"}
    ]
    observed = pd.to_datetime([r.get("sensing_time") for r in successful], utc=True)
    observed_set = set(observed.dropna())
    expected_set = set(expected)
    return {
        "expected_count": len(expected_set),
        "observed_count": len(observed_set & expected_set),
        "missing_timestamps": sorted(str(x) for x in expected_set - observed_set),
        "duplicate_product_ids": sorted(
            k for k, v in pd.Series([r.get("product_id") for r in manifest]).value_counts().items()
            if v > 1
        ),
        "duplicate_sensing_timestamps": sorted(
            str(x) for x, v in pd.Series(observed).value_counts().items() if v > 1
        ),
    }


def manifest_to_parquet(records: list[dict[str, Any]], output: str | Path) -> Path:
    """Persist the small acquisition manifest in the canonical tabular form."""
    import pandas as pd

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(destination, index=False)
    return destination


def validate_manifest(records: list[dict[str, Any]]) -> None:
    """Validate UTC sensing times and reject duplicate product timestamps."""
    from datetime import datetime

    seen: set[datetime] = set()
    seen_ids: set[str] = set()
    for record in records:
        product_id = str(record.get("product_id", ""))
        if product_id in seen_ids:
            raise ValueError(f"duplicate satellite product_id: {product_id}")
        seen_ids.add(product_id)
        value = str(record.get("sensing_time", "")).replace("Z", "+00:00")
        if not value:
            continue
        timestamp = datetime.fromisoformat(value)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("manifest sensing_time must be timezone-aware")
        timestamp = timestamp.astimezone(__import__("datetime").UTC)
        if timestamp in seen:
            raise ValueError(f"duplicate satellite timestamp: {timestamp.isoformat()}")
        seen.add(timestamp)
