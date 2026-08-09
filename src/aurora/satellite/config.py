from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PRODUCT_COLLECTION = "EO:EUM:DAT:MSG:HRSEVIRI"
CHANNEL_NAMES = (
    "VIS006",
    "VIS008",
    "IR_016",
    "IR_039",
    "WV_062",
    "WV_073",
    "IR_087",
    "IR_097",
    "IR_108",
    "IR_120",
    "IR_134",
    "HRV",
)


@dataclass(frozen=True)
class SiteConfig:
    site_id: str = "hirvensalmi"
    latitude: float = 61.64
    longitude: float = 26.78
    projection: str = "latlon"
    resolution_m: float = 3000.0
    patch_size: int = 64
    radius_km: float | None = None

    def __post_init__(self) -> None:
        if self.patch_size <= 0 or self.patch_size % 2:
            raise ValueError("patch_size must be a positive even number")
        if not -90 <= self.latitude <= 90 or not -180 <= self.longitude <= 180:
            raise ValueError("site latitude/longitude are out of range")


@dataclass(frozen=True)
class SatelliteConfig:
    api_key: str | None = None
    api_secret: str | None = None
    raw_uri: str = "data/raw/satellite/native"
    processed_uri: str = "data/processed/satellite"
    collection: str = PRODUCT_COLLECTION
    channels: tuple[str, ...] = CHANNEL_NAMES
    normalization_version: str = "seviri-global-v1"
    max_alignment_minutes: float = 7.5
    retry_attempts: int = 4
    timeout_seconds: float = 60.0
    token_validity_seconds: int = 86400
    discovery_retry_attempts: int = 8
    discovery_retry_max_backoff_seconds: float = 60.0
    collection_chunk_hours: int = 24
    backup_uri: str | None = None
    normalization_stats_uri: str | None = None
    start_utc: str = "2025-01-01T00:00:00Z"
    end_utc: str = "2026-01-01T00:00:00Z"
    download_workers: int = 4
    processing_workers: int = 6
    pipeline_queue_size: int = 12

    def __post_init__(self) -> None:
        for name in ("token_validity_seconds", "discovery_retry_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        backoff = self.discovery_retry_max_backoff_seconds
        if isinstance(backoff, bool) or not isinstance(backoff, (int, float)) or backoff <= 0:
            raise ValueError("discovery_retry_max_backoff_seconds must be positive")

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> SatelliteConfig:
        """Load satellite settings from a TOML file.

        Credentials intentionally come only from the file; they are never
        inferred from process environment variables or printed in diagnostics.
        """
        with open(path, "rb") as handle:
            values = tomllib.load(handle)
        values = values.get("satellite", values)
        if "channels" in values:
            values["channels"] = tuple(values["channels"])
        return cls(**values)

    def metadata(self, site: SiteConfig) -> dict[str, object]:
        """Return auditable run metadata without exposing credentials."""
        public = {
            "collection": self.collection,
            "channels": list(self.channels),
            "normalization_version": self.normalization_version,
            "max_alignment_minutes": self.max_alignment_minutes,
            "collection_period": {"start": self.start_utc, "end": self.end_utc},
            "site": {
                "site_id": site.site_id,
                "latitude": site.latitude,
                "longitude": site.longitude,
                "resolution_m": site.resolution_m,
                "patch_size": site.patch_size,
            },
        }
        encoded = json.dumps(public, sort_keys=True, separators=(",", ":")).encode()
        return {**public, "configuration_hash": hashlib.sha256(encoded).hexdigest()}

    def validate_collection(self) -> None:
        if self.collection_chunk_hours <= 0:
            raise ValueError("collection_chunk_hours must be positive")
        if tuple(self.channels) != tuple(CHANNEL_NAMES):
            raise ValueError("HRSEVIRI preprocessing requires exactly the configured 12 channels")
        for name in ("download_workers", "processing_workers", "pipeline_queue_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.pipeline_queue_size < max(self.download_workers, self.processing_workers):
            raise ValueError("pipeline_queue_size must be at least the worker count")

    def validate_credentials(self) -> None:
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "api_key and api_secret are required in the satellite TOML config"
            )

    @staticmethod
    def ensure_utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("satellite timestamps must be timezone-aware UTC")
        utc = value.astimezone(__import__("datetime").UTC)
        return utc


def load_site_config(path: Path | str) -> SiteConfig:
    import json

    values = json.loads(Path(path).read_text(encoding="utf-8"))
    values = values.get("site", values)
    return SiteConfig(**values)
