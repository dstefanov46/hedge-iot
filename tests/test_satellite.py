from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest

from aurora.satellite.acquisition import (
    EumetsatClient,
    coverage_report,
    discover_products,
    download_product,
    download_products,
    validate_native_archive,
)
from aurora.satellite import batch as satellite_batch
from aurora.satellite.config import CHANNEL_NAMES, SatelliteConfig, SiteConfig
from aurora.satellite.embedding import FrozenCNNEncoder
from aurora.satellite.features import align_satellite_features, derive_patch_features
from aurora.satellite.preprocess import (
    EmptySatelliteProductError,
    extract_center_patch,
    fit_normalization_stats,
    validate_channels,
    validate_patch,
)


def channels(value=1.0):
    return {
        name: np.full((8, 8), value + i, dtype=np.float32) for i, name in enumerate(CHANNEL_NAMES)
    }


def test_channel_validation_and_center_patch():
    patch = extract_center_patch(channels(), 4, 4, 4)
    assert patch.shape == (12, 4, 4)
    validate_channels(channels())
    with pytest.raises(ValueError):
        validate_channels({"VIS006": np.zeros((2, 2))})


def test_features_and_embedding_are_deterministic():
    patch = np.ones((12, 64, 64), dtype=np.float32)
    features = derive_patch_features(patch)
    assert features["satellite_missing"] == 0
    first = FrozenCNNEncoder(seed=9).encode(patch)
    second = FrozenCNNEncoder(seed=9).encode(patch)
    assert first.shape == (32,)
    np.testing.assert_allclose(first, second)


def test_empty_patch_is_typed_and_missing_channel_features_are_nan():
    with pytest.raises(EmptySatelliteProductError):
        validate_patch(np.full((12, 64, 64), np.nan, dtype=np.float32))

    patch = np.ones((12, 64, 64), dtype=np.float32)
    patch[-1] = np.nan
    features = derive_patch_features(patch)
    assert np.isnan(features["satellite_hrv_mean"])
    assert np.isnan(features["satellite_hrv_min"])
    assert np.isfinite(features["satellite_cloud_index"])


def test_alignment_is_backward_only_and_tolerant():
    frame = pd.DataFrame(
        {
            "site_id": ["s", "s"],
            "timestamp_utc": pd.to_datetime(["2025-01-01T00:00Z", "2025-01-01T00:15Z"]),
        }
    )
    satellite = pd.DataFrame(
        {
            "site_id": ["s", "s"],
            "timestamp_utc": pd.to_datetime(
                ["2024-12-31T23:59Z", "2025-01-01T00:14Z"]
            ),
            "satellite_missing": [0.1, 0.2],
        }
    )
    result = align_satellite_features(frame, satellite)
    assert result["satellite_missing"].tolist() == [0.1, 0.2]


def test_download_is_idempotent_and_writes_checksum(tmp_path):
    class FakeClient:
        def __init__(self):
            self.downloads = 0

        def search(self, collection, start, end, bbox=None):
            return [{"id": "p1", "sensing_time": "2025-01-01T00:00:00Z", "satellite": "MSG"}]

        def download(self, product, destination):
            self.downloads += 1
            destination.write_bytes(b"native")

    client = FakeClient()
    config = SatelliteConfig(raw_uri=str(tmp_path / "native"))
    args = (
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 15, tzinfo=UTC),
        SiteConfig(),
        config,
    )
    first = download_products(*args, client=client)
    second = download_products(*args, client=client)
    assert client.downloads == 1
    assert first[0]["checksum"] == second[0]["checksum"]


def test_finalized_deleted_product_is_preserved_without_redownload(tmp_path):
    class FakeClient:
        def __init__(self):
            self.downloads = 0

        def search(self, collection, start, end, bbox=None):
            return [
                {
                    "id": "p1",
                    "sensing_time": datetime(2025, 1, 1, tzinfo=UTC),
                    "satellite": "MSG",
                    "checksum": "provider-md5",
                    "checksum_algorithm": "md5",
                }
            ]

        def download(self, product, destination):
            self.downloads += 1
            raise AssertionError("a finalized product must not be downloaded again")

    config = SatelliteConfig(raw_uri=str(tmp_path / "native"))
    prior = {
        "product_id": "p1",
        "sensing_time": "2025-01-01 00:00:00+00:00",
        "archive_uri": str(tmp_path / "native/2025/01/01/p1.zip"),
        "local_uri": str(tmp_path / "native/2025/01/01/p1.zip"),
        "processing_status": "processed",
        "processing_checksum": "patch-sha256",
        "backup_status": "verified",
        "backup_checksum": "inventory-sha256",
        "deletion_status": "deleted",
        "source_archive_checksum": "source-sha256",
        "quality_status": "degraded",
        "recovery_provenance": {"method": "sidecar"},
    }
    client = FakeClient()
    items = discover_products(
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 15, tzinfo=UTC),
        SiteConfig(),
        config,
        client=client,
        existing_records=[prior],
    )

    discovered = items[0]["record"]
    assert discovered["source_archive_checksum"] == "source-sha256"
    assert discovered["quality_status"] == "degraded"
    assert discovered["recovery_provenance"] == {"method": "sidecar"}
    completed = download_product(items[0], config, client=client)
    assert completed["deletion_status"] == "deleted"
    assert client.downloads == 0


def test_download_merges_distinct_products_into_manifest(tmp_path):
    class FakeClient:
        def __init__(self):
            self.product_number = 0

        def search(self, collection, start, end, bbox=None):
            self.product_number += 1
            minute = 15 * (self.product_number - 1)
            return [
                {
                    "id": f"p{self.product_number}",
                    "sensing_time": f"2025-01-01T00:{minute:02d}:00+00:00",
                    "satellite": "MSG",
                }
            ]

        def download(self, product, destination):
            destination.write_bytes(product["id"].encode())

    client = FakeClient()
    config = SatelliteConfig(raw_uri=str(tmp_path / "native"))
    start = datetime(2025, 1, 1, tzinfo=UTC)
    first_end = datetime(2025, 1, 1, 0, 15, tzinfo=UTC)
    second_start = first_end
    second_end = datetime(2025, 1, 1, 0, 30, tzinfo=UTC)
    download_products(start, first_end, SiteConfig(), config, client=client)
    download_products(second_start, second_end, SiteConfig(), config, client=client)

    import json

    payload = json.loads((tmp_path / "native" / "manifest.json").read_text())
    assert [record["product_id"] for record in payload["products"]] == ["p1", "p2"]


def test_satellite_config_is_loaded_from_toml(tmp_path):
    path = tmp_path / "satellite.toml"
    path.write_text(
        """[satellite]\napi_key = 'key'\napi_secret = 'secret'\nraw_uri = 's3://bucket/raw'\n""",
        encoding="utf-8",
    )
    config = SatelliteConfig.from_file(path)
    assert config.api_key == "key"
    assert config.api_secret == "secret"
    assert config.raw_uri == "s3://bucket/raw"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("token_validity_seconds", 0),
        ("discovery_retry_attempts", 0),
        ("discovery_retry_max_backoff_seconds", 0),
    ],
)
def test_discovery_configuration_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        SatelliteConfig(**{field: value})


def _http_error(status: int, url: str):
    from requests import Response
    from requests.exceptions import HTTPError

    response = Response()
    response.status_code = status
    response.url = url
    return HTTPError(f"status {status}", response=response)


def _eumetsat_client_with_outcomes(outcomes, config=None):
    attempts = {"tokens": [], "datastores": 0}

    class EmptyCollection:
        def search(self, **query):
            return []

    class DataStore:
        def __init__(self, token):
            self.outcome = outcomes[attempts["datastores"]]
            attempts["datastores"] += 1

        def get_collection(self, collection):
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            return EmptyCollection()

    def access_token(credentials, validity):
        attempts["tokens"].append(validity)
        return object()

    FakeEumdac = type(
        "FakeEumdac",
        (),
        {"DataStore": DataStore, "AccessToken": staticmethod(access_token)},
    )

    client = object.__new__(EumetsatClient)
    client.config = config or SatelliteConfig(
        api_key="key", api_secret="secret", discovery_retry_attempts=len(outcomes)
    )
    client._eumdac = FakeEumdac
    return client, attempts


def test_discovery_retries_token_404_and_5xx_with_fresh_clients(monkeypatch):
    outcomes = [
        _http_error(404, "https://api.eumetsat.int/token"),
        _http_error(503, "https://api.eumetsat.int/data/search-products/1.0.0/os"),
        None,
    ]
    client, attempts = _eumetsat_client_with_outcomes(outcomes)
    delays = []
    monkeypatch.setattr("aurora.satellite.acquisition.time.sleep", delays.append)

    products = client.search(
        "collection",
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
    )

    assert products == []
    assert attempts == {"tokens": [86400, 86400, 86400], "datastores": 3}
    assert delays == [1, 2]


class _WrappedEumdacError(Exception):
    def __init__(self, status=None, url=None):
        super().__init__("wrapped EUMDAC failure")
        self.extra_info = {
            key: value
            for key, value in {"status": status, "url": url}.items()
            if value is not None
        }


def test_discovery_retries_wrapped_eumdac_5xx_with_fresh_client(monkeypatch):
    endpoint = urlparse(
        "https://api.eumetsat.int/data/search-products/1.0.0/osdd"
    )
    wrapped = _WrappedEumdacError(500, endpoint)
    client, attempts = _eumetsat_client_with_outcomes([wrapped, None])
    delays = []
    monkeypatch.setattr("aurora.satellite.acquisition.time.sleep", delays.append)

    products = client.search(
        "collection",
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
    )

    assert products == []
    assert EumetsatClient._failure_details(wrapped) == (500, endpoint.geturl())
    assert attempts == {"tokens": [86400, 86400], "datastores": 2}
    assert delays == [1]


def test_discovery_retries_http_error_chained_by_sdk(monkeypatch):
    endpoint = "https://api.eumetsat.int/data/search-products/1.0.0/osdd"
    wrapped = RuntimeError("SDK wrapper")
    wrapped.__cause__ = _http_error(502, endpoint)
    client, attempts = _eumetsat_client_with_outcomes([wrapped, None])
    monkeypatch.setattr("aurora.satellite.acquisition.time.sleep", lambda delay: None)

    products = client.search(
        "collection",
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
    )

    assert products == []
    assert EumetsatClient._failure_details(wrapped) == (502, endpoint)
    assert attempts["datastores"] == 2


def test_discovery_wrapped_eumdac_5xx_exhausts_with_final_error(monkeypatch):
    endpoint = "https://api.eumetsat.int/data/search-products/1.0.0/osdd"
    errors = [
        _WrappedEumdacError(500, endpoint),
        _WrappedEumdacError(503, endpoint),
    ]
    client, attempts = _eumetsat_client_with_outcomes(errors)
    delays = []
    monkeypatch.setattr("aurora.satellite.acquisition.time.sleep", delays.append)

    with pytest.raises(_WrappedEumdacError) as caught:
        client.search(
            "collection",
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 2, tzinfo=UTC),
        )

    assert caught.value is errors[-1]
    assert attempts["datastores"] == 2
    assert delays == [1]


@pytest.mark.parametrize(
    ("status", "url"),
    [
        (401, "https://api.eumetsat.int/token"),
        (403, "https://api.eumetsat.int/token"),
        (404, "https://api.eumetsat.int/data/search-products/1.0.0/os"),
    ],
)
def test_discovery_non_retryable_http_errors_fail_immediately(monkeypatch, status, url):
    client, attempts = _eumetsat_client_with_outcomes(
        [_http_error(status, url)],
        SatelliteConfig(
            api_key="key", api_secret="secret", discovery_retry_attempts=8
        ),
    )
    monkeypatch.setattr(
        "aurora.satellite.acquisition.time.sleep",
        lambda delay: pytest.fail("non-retryable discovery failure slept"),
    )
    with pytest.raises(Exception, match=f"status {status}"):
        client.search(
            "collection",
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 2, tzinfo=UTC),
        )
    assert attempts["datastores"] == 1


@pytest.mark.parametrize(
    "failure",
    [
        _WrappedEumdacError(403, urlparse("https://api.eumetsat.int/token")),
        _WrappedEumdacError(),
    ],
    ids=["wrapped-403", "semantic-sdk-error"],
)
def test_discovery_non_retryable_wrapped_sdk_errors_fail_immediately(
    monkeypatch, failure
):
    client, attempts = _eumetsat_client_with_outcomes(
        [failure],
        SatelliteConfig(
            api_key="key", api_secret="secret", discovery_retry_attempts=8
        ),
    )
    monkeypatch.setattr(
        "aurora.satellite.acquisition.time.sleep",
        lambda delay: pytest.fail("non-retryable discovery failure slept"),
    )

    with pytest.raises(_WrappedEumdacError):
        client.search(
            "collection",
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 2, tzinfo=UTC),
        )

    assert attempts["datastores"] == 1


def test_discovery_backoff_caps_and_exhausts(monkeypatch):
    failures = [
        _http_error(500, "https://api.eumetsat.int/catalogue") for _ in range(8)
    ]
    client, attempts = _eumetsat_client_with_outcomes(failures)
    delays = []
    monkeypatch.setattr("aurora.satellite.acquisition.time.sleep", delays.append)
    with pytest.raises(Exception, match="status 500"):
        client.search(
            "collection",
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 2, tzinfo=UTC),
        )
    assert attempts["datastores"] == 8
    assert delays == [1, 2, 4, 8, 16, 32, 60]


def test_native_archive_and_coverage_report(tmp_path):
    import zipfile

    archive = tmp_path / "product.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("product.nat", b"native")
    assert validate_native_archive(archive)["nat_member"] == "product.nat"
    report = coverage_report(
        [
            {
                "product_id": "p1",
                "sensing_time": "2025-01-01T00:00:00Z",
                "processing_status": "downloaded",
            }
        ],
        pd.date_range("2025-01-01", periods=2, freq="15min", tz="UTC"),
    )
    assert report["observed_count"] == 1
    assert len(report["missing_timestamps"]) == 1


def test_normalization_is_fit_from_supplied_training_patches():
    training = np.zeros((2, 12, 64, 64), dtype=np.float32)
    training[1] = 2
    stats = fit_normalization_stats(training)
    assert stats["training_patch_count"] == 2
    assert stats["means"][0] == 1.0
    assert stats["scales"][0] > 0


def test_cached_archive_metadata_requires_unchanged_archive(tmp_path):
    archive = tmp_path / "product.zip"
    import zipfile

    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("product.nat", b"native")
    stat = archive.stat()
    validation = validate_native_archive(archive)
    record = {
        "archive_validation": validation,
        "archive_checksum": "cached-sha256",
        "archive_size": stat.st_size,
        "archive_mtime_ns": stat.st_mtime_ns,
    }
    cached_validation, cached_checksum = satellite_batch._cached_archive_metadata(record, archive)
    assert cached_validation == validation
    assert cached_checksum == "cached-sha256"
    assert satellite_batch._cached_archive_metadata({}, archive) == (None, None)

    archive.write_bytes(archive.read_bytes() + b"changed")
    assert satellite_batch._cached_archive_metadata(record, archive) == (None, None)


def test_batch_returns_timings_and_loads_written_patch_once(tmp_path, monkeypatch):
    import zipfile

    raw = tmp_path / "raw"
    archive = raw / "2025/01/01/p1.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("product.nat", b"native")

    class FakeClient:
        def search(self, collection, start, end, bbox=None):
            return [{"id": "p1", "sensing_time": "2025-01-01T00:00:00Z"}]

        def download(self, product, destination):
            raise AssertionError("the prepared archive should be reused")

    patch_loads = 0
    original_load_patch = satellite_batch.load_patch

    def counted_load_patch(uri):
        nonlocal patch_loads
        patch_loads += 1
        return original_load_patch(uri)

    monkeypatch.setattr(satellite_batch, "load_patch", counted_load_patch)
    result = satellite_batch.run_satellite_batch(
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 15, tzinfo=UTC),
        SiteConfig(),
        SatelliteConfig(raw_uri=str(raw), processed_uri=str(tmp_path / "processed")),
        client=FakeClient(),
        decoder=lambda archive, site: channels(),
    )

    assert result["products"][0]["processing_status"] == "processed"
    assert patch_loads == 1
    assert result["timings"]["total_elapsed"] >= 0
    assert {
        "collection_download", "archive_validation_checksum", "native_decoding",
        "patch_preprocessing_writing", "feature_extraction", "backup_verification",
        "native_deletion", "total_elapsed",
    } <= result["timings"].keys()


def test_incremental_checkpoint_copies_only_new_chunk_and_allows_verified_deletion(
    tmp_path, monkeypatch
):
    raw, processed, backup = tmp_path / "raw", tmp_path / "processed", tmp_path / "backup"
    raw.mkdir()
    processed.mkdir()
    (raw / "manifest.json").write_text('{"products": []}', encoding="utf-8")
    (raw / "manifest.events.jsonl").write_text("", encoding="utf-8")
    config = SatelliteConfig(
        raw_uri=str(raw), processed_uri=str(processed), backup_uri=str(backup)
    )
    records = []
    for product_id in ("p1", "p2"):
        patch = processed / "physical_patches" / f"{product_id}.zarr"
        patch.mkdir(parents=True)
        (patch / "data.npy").write_bytes(product_id.encode())
        patch.with_suffix(".zarr.json").write_text("{}", encoding="utf-8")
        archive = raw / f"{product_id}.zip"
        archive.write_bytes(product_id.encode())
        records.append(
            {
                "product_id": product_id,
                "processing_status": "processed",
                "patch_uri": str(patch),
                "archive_uri": str(archive),
                "deletion_status": "pending",
            }
        )

    copied_sources = []
    original_atomic_copy = satellite_batch.atomic_copy

    def tracked_copy(source, destination):
        copied_sources.append(str(source))
        original_atomic_copy(source, destination)

    monkeypatch.setattr(satellite_batch, "atomic_copy", tracked_copy)
    first = satellite_batch.backup_checkpoint(config, SiteConfig(), "c1", [records[0]])
    copied_sources.clear()
    second = satellite_batch.backup_checkpoint(config, SiteConfig(), "c2", [records[1]])

    assert not any("p1.zarr" in source for source in copied_sources)
    assert set(first["checksums"]) < set(second["checksums"])
    deleted = satellite_batch.delete_verified_native(config, records)
    assert set(deleted) == {str(raw / "p1.zip"), str(raw / "p2.zip")}
    assert not (raw / "p1.zip").exists()
    assert not (raw / "p2.zip").exists()


def _write_native_archive(path):
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("product.nat", b"native")


class _OneProductClient:
    def __init__(self):
        self.searches = 0
        self.downloads = 0

    def search(self, collection, start, end, bbox=None):
        self.searches += 1
        return [
            {
                "id": "p1",
                "sensing_time": datetime(2025, 1, 1, tzinfo=UTC),
                "satellite": "MSG",
            }
        ]

    def download(self, product, destination):
        self.downloads += 1
        raise AssertionError("prepared native archive should be retained and reused")


def test_empty_product_retries_once_becomes_terminal_and_is_skipped(tmp_path):
    raw = tmp_path / "raw"
    archive = raw / "2025/01/01/p1.zip"
    _write_native_archive(archive)
    config = SatelliteConfig(
        raw_uri=str(raw), processed_uri=str(tmp_path / "processed"), processing_workers=1
    )
    client = _OneProductClient()
    result = satellite_batch.run_satellite_batch(
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 15, tzinfo=UTC),
        SiteConfig(),
        config,
        client=client,
        decoder=lambda archive, site: channels(float("nan")),
    )

    product = result["products"][0]
    assert product["processing_status"] == "unavailable_source"
    assert product["unavailable_reason"] == "all_channels_non_finite"
    assert product["attempt_count"] == 2
    assert result["failed_products"] == []
    assert result["unavailable_products"] == [product]
    assert result["counts"]["unavailable"] == 1
    assert result["counts"]["failed"] == 0
    assert archive.exists()
    assert product["archive_checksum"]
    assert product["archive_validation"]["nat_member"] == "product.nat"
    assert not (tmp_path / "processed" / "features.parquet").exists()
    event_statuses = [
        __import__("json").loads(line)["to"]["processing_status"]
        for line in (raw / "manifest.events.jsonl").read_text().splitlines()
    ]
    assert event_statuses == ["failed", "unavailable_source"]

    class NoProviderCalls:
        def search(self, *args, **kwargs):
            raise AssertionError("completed discovery checkpoint should be reused")

        def download(self, *args, **kwargs):
            raise AssertionError("unavailable source must not be downloaded")

    resumed = satellite_batch.run_satellite_batch(
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 15, tzinfo=UTC),
        SiteConfig(),
        config,
        client=NoProviderCalls(),
        decoder=lambda archive, site: pytest.fail("unavailable source must not be decoded"),
    )
    assert resumed["failed_products"] == []
    assert len(resumed["unavailable_products"]) == 1
    assert resumed["counts"]["submitted"] == 0


def test_existing_failed_empty_product_gets_only_final_retry_and_quarantine(tmp_path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    archive = raw / "2025/01/01/p1.zip"
    _write_native_archive(archive)
    patch = processed / "physical_patches/p1.zarr"
    patch.mkdir(parents=True)
    np.save(patch / "data.npy", np.full((12, 64, 64), np.nan, dtype=np.float32))
    patch.with_suffix(".zarr.json").write_text('{"legacy": true}', encoding="utf-8")
    processed.mkdir(exist_ok=True)
    pd.DataFrame([{"product_id": "p1", "satellite_missing": 1.0}]).to_parquet(
        processed / "features.parquet", index=False
    )
    raw.mkdir(exist_ok=True)
    manifest = {
        "site": SiteConfig().__dict__,
        "collection_run": {},
        "products": [
            {
                "product_id": "p1",
                "sensing_time": "2025-01-01T00:00:00+00:00",
                "archive_uri": str(archive),
                "local_uri": str(archive),
                "archive_checksum": "retained-checksum",
                "archive_validation": {"nat_member": "product.nat", "nat_size": 6},
                "processing_status": "failed",
                "attempt_count": 1,
                "patch_uri": str(patch),
            }
        ],
    }
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = SatelliteConfig(
        raw_uri=str(raw), processed_uri=str(processed), processing_workers=1
    )

    class LegacyDiscovery:
        searches = 0

        def search(self, *args, **kwargs):
            self.searches += 1
            return [
                {
                    "id": "p1",
                    "sensing_time": datetime(2025, 1, 1, tzinfo=UTC),
                    "satellite": "MSG",
                }
            ]

        def download(self, *args, **kwargs):
            raise AssertionError("the retained archive must be reused")

    client = LegacyDiscovery()

    result = satellite_batch.run_satellite_batch(
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 15, tzinfo=UTC),
        SiteConfig(),
        config,
        client=client,
        decoder=lambda archive, site: channels(float("nan")),
    )

    product = result["unavailable_products"][0]
    assert result["counts"]["submitted"] == 1
    assert product["attempt_count"] == 2
    assert archive.exists()
    assert product["archive_checksum"]
    assert not patch.exists()
    quarantined = Path(product["quarantined_patch_uri"])
    assert quarantined.is_dir()
    assert processed / "quarantine" in quarantined.parents
    assert quarantined.with_suffix(".zarr.json").exists()
    assert pd.read_parquet(processed / "features.parquet").empty
    assert client.searches == 1


def test_generic_worker_failure_remains_failed(tmp_path):
    raw = tmp_path / "raw"
    _write_native_archive(raw / "2025/01/01/p1.zip")
    config = SatelliteConfig(
        raw_uri=str(raw), processed_uri=str(tmp_path / "processed"), processing_workers=1
    )

    def broken_decoder(archive, site):
        raise RuntimeError("decoder exploded")

    result = satellite_batch.run_satellite_batch(
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 15, tzinfo=UTC),
        SiteConfig(),
        config,
        client=_OneProductClient(),
        decoder=broken_decoder,
    )
    assert len(result["failed_products"]) == 1
    assert result["unavailable_products"] == []
    assert result["counts"]["failed"] == 1
    assert result["counts"]["submitted"] == 1


def test_failed_discovery_preserves_materialized_checkpoint(tmp_path):
    import json

    raw = tmp_path / "raw"
    raw.mkdir()
    checkpoint = "2025-01-01T00:15:00+00:00"
    payload = {
        "site": SiteConfig().__dict__,
        "collection_run": {"discovery_checkpoint_utc": checkpoint},
        "products": [],
    }
    manifest = raw / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    class BrokenDiscovery:
        def search(self, *args, **kwargs):
            raise ConnectionError("catalogue unavailable")

    with pytest.raises(ConnectionError, match="catalogue unavailable"):
        satellite_batch.run_satellite_batch(
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 1, 0, 30, tzinfo=UTC),
            SiteConfig(),
            SatelliteConfig(
                raw_uri=str(raw),
                processed_uri=str(tmp_path / "processed"),
                processing_workers=1,
            ),
            client=BrokenDiscovery(),
            decoder=lambda archive, site: channels(),
        )
    migrated = json.loads(manifest.read_text(encoding="utf-8"))
    assert migrated["products"] == []
    assert migrated["collection_run"]["discovery_coverage_utc"] == []
    assert migrated["collection_run"]["discovery_checkpoint_utc"] == (
        "2025-01-01T00:00:00+00:00"
    )


@pytest.mark.parametrize(
    "error_factory",
    [
        lambda: json.JSONDecodeError("malformed catalogue", "{", 1),
        lambda: __import__("requests").exceptions.JSONDecodeError(
            "malformed catalogue", "{", 1
        ),
    ],
)
def test_malformed_discovery_restarts_complete_search_with_fresh_client(
    monkeypatch, error_factory
):
    attempts = {"tokens": 0, "datastores": 0, "searches": 0}

    class Product:
        sensing_start = datetime(2025, 1, 1, tzinfo=UTC)
        md5 = None

        def __str__(self):
            return "p1"

    class Collection:
        def search(self, **query):
            attempts["searches"] += 1
            yield Product()
            if attempts["searches"] == 1:
                raise error_factory()

    class DataStore:
        def __init__(self, token):
            attempts["datastores"] += 1

        def get_collection(self, collection):
            return Collection()

    def access_token(credentials, validity):
        attempts["tokens"] += 1
        return object()

    client = object.__new__(EumetsatClient)
    client.config = SatelliteConfig(
        api_key="key", api_secret="secret", discovery_retry_attempts=2
    )
    client._eumdac = type(
        "FakeEumdac",
        (),
        {"DataStore": DataStore, "AccessToken": staticmethod(access_token)},
    )
    monkeypatch.setattr("aurora.satellite.acquisition.time.sleep", lambda delay: None)

    products = client.search(
        "collection",
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 1, tzinfo=UTC),
    )

    assert [product["id"] for product in products] == ["p1"]
    assert attempts == {"tokens": 2, "datastores": 2, "searches": 2}


def test_repeated_malformed_discovery_exhausts_retries_with_final_error(monkeypatch):
    errors = [
        json.JSONDecodeError("first malformed response", "{", 0),
        json.JSONDecodeError("final malformed response", "[", 0),
    ]
    client, attempts = _eumetsat_client_with_outcomes(errors)
    monkeypatch.setattr("aurora.satellite.acquisition.time.sleep", lambda delay: None)

    with pytest.raises(json.JSONDecodeError, match="final malformed response") as caught:
        client.search(
            "collection",
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 1, 1, tzinfo=UTC),
        )

    assert caught.value is errors[-1]
    assert attempts["datastores"] == 2


def test_discovery_coverage_merges_adjacent_intervals():
    start = datetime(2025, 1, 1, tzinfo=UTC)
    merged = satellite_batch._merge_discovery_coverage(
        [
            (start + timedelta(hours=8), start + timedelta(hours=12)),
            (start, start + timedelta(hours=4)),
            (start + timedelta(hours=4), start + timedelta(hours=8)),
            (start + timedelta(hours=16), start + timedelta(hours=20)),
        ]
    )
    assert merged == [
        (start, start + timedelta(hours=12)),
        (start + timedelta(hours=16), start + timedelta(hours=20)),
    ]

    stats = satellite_batch._discovery_statistics(
        start,
        start + timedelta(hours=1),
        timedelta(hours=1),
        [(start, start + timedelta(hours=1))],
        [{"sensing_time": (start + timedelta(seconds=11)).isoformat()}],
    )
    assert stats["confirmed_missing_15_minute_source_slots"] == 3


def test_legacy_scalar_and_later_record_do_not_skip_earlier_discovery(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    manifest = {
        "site": SiteConfig().__dict__,
        "collection_run": {
            "discovery_checkpoint_utc": (start + timedelta(hours=8)).isoformat()
        },
        "products": [
            {
                "product_id": "later",
                "sensing_time": (start + timedelta(hours=7)).isoformat(),
                "archive_uri": str(raw / "later.zip"),
                "local_uri": str(raw / "later.zip"),
                "processing_status": "processed",
                "backup_status": "verified",
                "deletion_status": "deleted",
            }
        ],
    }
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    class EmptyCatalogue:
        def __init__(self):
            self.searches = []

        def search(self, collection, chunk_start, chunk_end, bbox=None):
            self.searches.append((chunk_start, chunk_end))
            return []

    client = EmptyCatalogue()
    result = satellite_batch.run_satellite_batch(
        start,
        start + timedelta(hours=8),
        SiteConfig(),
        SatelliteConfig(
            raw_uri=str(raw),
            processed_uri=str(tmp_path / "processed"),
            processing_workers=1,
        ),
        client=client,
        decoder=lambda archive, site: pytest.fail("no processing expected"),
        chunk_hours=4,
    )

    assert client.searches == [
        (start, start + timedelta(hours=4)),
        (start + timedelta(hours=4), start + timedelta(hours=8)),
    ]
    assert result["discovery"]["covered_chunks"] == 2
    assert result["discovery"]["remaining_uncovered_chunks"] == 0
    saved = json.loads((raw / "manifest.json").read_text())["collection_run"]
    assert saved["discovery_coverage_utc"] == [
        {"start": start.isoformat(), "end": (start + timedelta(hours=8)).isoformat()}
    ]


def test_full_year_chunk_selection_includes_large_and_small_historical_gaps():
    year_start = datetime(2025, 1, 1, tzinfo=UTC)
    year_end = datetime(2026, 1, 1, tzinfo=UTC)
    april_gap_start = datetime(2025, 4, 11, tzinfo=UTC)
    june_boundary = datetime(2025, 6, 1, tzinfo=UTC)
    coverage = satellite_batch._merge_discovery_coverage(
        [
            (year_start, april_gap_start),
            (june_boundary, datetime(2025, 8, 3, tzinfo=UTC)),
            (datetime(2025, 8, 3, 4, tzinfo=UTC), year_end),
        ]
    )
    uncovered = [
        chunk
        for chunk in satellite_batch._chunk_intervals(
            year_start, year_end, timedelta(hours=4)
        )
        if not satellite_batch._interval_is_covered(chunk[0], chunk[1], coverage)
    ]

    assert uncovered[0] == (april_gap_start, april_gap_start + timedelta(hours=4))
    assert (june_boundary - timedelta(hours=4), june_boundary) in uncovered
    assert (
        datetime(2025, 8, 3, tzinfo=UTC),
        datetime(2025, 8, 3, 4, tzinfo=UTC),
    ) in uncovered


def test_legacy_reaudit_does_not_redownload_verified_or_unavailable_products(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    products = [
        {
            "product_id": "verified",
            "sensing_time": start.isoformat(),
            "archive_uri": str(raw / "verified.zip"),
            "local_uri": str(raw / "verified.zip"),
            "processing_status": "processed",
            "backup_status": "verified",
            "deletion_status": "deleted",
        },
        {
            "product_id": "unavailable",
            "sensing_time": (start + timedelta(minutes=15)).isoformat(),
            "archive_uri": str(raw / "unavailable.zip"),
            "local_uri": str(raw / "unavailable.zip"),
            "processing_status": "unavailable_source",
            "unavailable_reason": "all_channels_non_finite",
        },
    ]
    (raw / "manifest.json").write_text(
        json.dumps(
            {
                "site": SiteConfig().__dict__,
                "collection_run": {
                    "discovery_checkpoint_utc": (
                        start + timedelta(minutes=30)
                    ).isoformat()
                },
                "products": products,
            }
        ),
        encoding="utf-8",
    )

    class ReauditCatalogue:
        downloads = 0

        def search(self, *args, **kwargs):
            return [
                {
                    "id": record["product_id"],
                    "sensing_time": datetime.fromisoformat(record["sensing_time"]),
                    "satellite": "MSG",
                }
                for record in products
            ]

        def download(self, *args, **kwargs):
            self.downloads += 1

    client = ReauditCatalogue()
    result = satellite_batch.run_satellite_batch(
        start,
        start + timedelta(minutes=30),
        SiteConfig(),
        SatelliteConfig(
            raw_uri=str(raw),
            processed_uri=str(tmp_path / "processed"),
            processing_workers=1,
        ),
        client=client,
        decoder=lambda archive, site: pytest.fail("terminal products must not be decoded"),
    )

    assert client.downloads == 0
    assert {record["processing_status"] for record in result["products"]} == {
        "processed",
        "unavailable_source",
    }


def test_interrupted_chunk_remains_uncovered(tmp_path):
    raw = tmp_path / "raw"
    start = datetime(2025, 1, 1, tzinfo=UTC)

    class InterruptedCatalogue:
        def __init__(self):
            self.calls = 0

        def search(self, collection, chunk_start, chunk_end, bbox=None):
            self.calls += 1
            if self.calls == 2:
                raise ConnectionError("interrupted second chunk")
            return []

    with pytest.raises(ConnectionError, match="interrupted second chunk"):
        satellite_batch.run_satellite_batch(
            start,
            start + timedelta(hours=8),
            SiteConfig(),
            SatelliteConfig(
                raw_uri=str(raw),
                processed_uri=str(tmp_path / "processed"),
                processing_workers=1,
            ),
            client=InterruptedCatalogue(),
            decoder=lambda archive, site: channels(),
            chunk_hours=4,
        )

    assert satellite_batch._load_discovery_coverage(raw / "manifest.json") == [
        (start, start + timedelta(hours=4))
    ]


def test_empty_catalogue_chunk_is_covered_and_reports_missing_slots(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=UTC)

    class EmptyCatalogue:
        def search(self, *args, **kwargs):
            return []

    result = satellite_batch.run_satellite_batch(
        start,
        start + timedelta(hours=1),
        SiteConfig(),
        SatelliteConfig(
            raw_uri=str(tmp_path / "raw"),
            processed_uri=str(tmp_path / "processed"),
            processing_workers=1,
        ),
        client=EmptyCatalogue(),
        decoder=lambda archive, site: channels(),
        chunk_hours=1,
    )

    assert result["discovery"]["covered_chunks"] == 1
    assert result["discovery"]["remaining_uncovered_chunks"] == 0
    assert result["discovery"]["confirmed_missing_15_minute_source_slots"] == 4
    assert result["counts"]["confirmed_missing_source_slots"] == 4


def test_covered_chunk_resumes_processing_without_discovery(tmp_path):
    raw = tmp_path / "raw"
    start = datetime(2025, 1, 1, tzinfo=UTC)
    archive = raw / "2025/01/01/p1.zip"
    _write_native_archive(archive)
    manifest = {
        "site": SiteConfig().__dict__,
        "collection_run": {
            "discovery_coverage_utc": [
                {
                    "start": start.isoformat(),
                    "end": (start + timedelta(minutes=15)).isoformat(),
                }
            ]
        },
        "products": [
            {
                "product_id": "p1",
                "sensing_time": start.isoformat(),
                "archive_uri": str(archive),
                "local_uri": str(archive),
                "processing_status": "downloaded",
            }
        ],
    }
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    class NoProviderCalls:
        def search(self, *args, **kwargs):
            raise AssertionError("covered discovery must not be repeated")

        def download(self, *args, **kwargs):
            raise AssertionError("retained archive must be reused")

    result = satellite_batch.run_satellite_batch(
        start,
        start + timedelta(minutes=15),
        SiteConfig(),
        SatelliteConfig(
            raw_uri=str(raw),
            processed_uri=str(tmp_path / "processed"),
            processing_workers=1,
        ),
        client=NoProviderCalls(),
        decoder=lambda archive, site: channels(),
    )

    assert result["products"][0]["processing_status"] == "processed"
    assert result["counts"]["submitted"] == 1
    assert result["discovery"]["covered_chunks"] == 1
