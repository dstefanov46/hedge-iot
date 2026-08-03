"""EUMETSAT HRSEVIRI acquisition, decoding, and feature extraction."""

from aurora.satellite.batch import finalize_satellite_folds, run_satellite_batch
from aurora.satellite.config import SatelliteConfig, SiteConfig
from aurora.satellite.embedding import FrozenCNNEncoder, fit_fold_encoder

__all__ = [
    "FrozenCNNEncoder",
    "SatelliteConfig",
    "SiteConfig",
    "fit_fold_encoder",
    "run_satellite_batch",
    "finalize_satellite_folds",
]
