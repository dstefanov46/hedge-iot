"""EUMETSAT HRSEVIRI acquisition, decoding, and feature extraction."""

from aurora.satellite.batch import finalize_satellite_folds, run_satellite_batch
from aurora.satellite.config import SatelliteConfig, SiteConfig
from aurora.satellite.embedding import FrozenCNNEncoder, fit_fold_encoder
from aurora.satellite.slovenian import (
    discover_archives,
    iter_site_patches,
    read_archive,
    sample_site_patches,
    site_patch,
)
from aurora.satellite.transfer import materialize_satellite_transfer, prepare_satellite_transfer

__all__ = [
    "FrozenCNNEncoder",
    "SatelliteConfig",
    "SiteConfig",
    "fit_fold_encoder",
    "discover_archives",
    "iter_site_patches",
    "read_archive",
    "sample_site_patches",
    "site_patch",
    "prepare_satellite_transfer",
    "materialize_satellite_transfer",
    "run_satellite_batch",
    "finalize_satellite_folds",
]
