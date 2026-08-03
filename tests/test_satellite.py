from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from aurora.satellite.acquisition import coverage_report, download_products, validate_native_archive
from aurora.satellite.config import CHANNEL_NAMES, SatelliteConfig, SiteConfig
from aurora.satellite.embedding import FrozenCNNEncoder
from aurora.satellite.features import align_satellite_features, derive_patch_features
from aurora.satellite.preprocess import (
    extract_center_patch,
    fit_normalization_stats,
    validate_channels,
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
