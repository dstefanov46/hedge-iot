from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aurora.data.canonical import SiteSpec, canonicalize_site_frame
from aurora.evaluation.cross_validation import generate_expanding_finnish_folds
from aurora.evaluation.transfer import (
    assert_skill_gate,
    common_daylight_mask,
    evaluate_long_forecasts,
)
from aurora.forecasting.baselines import (
    HarmonicSmartPersistence,
    PhysicalSmartPersistence,
    enforce_monotonic_quantiles,
)
from aurora.forecasting.tft_transfer import (
    PHYSICAL_CONTEXT_REALS,
    SATELLITE_REALS,
    TFTTransferConfig,
    create_dataset,
    fit_tft,
    prepare_tft_frame,
    predict_long,
    save_model_bundle,
)


def test_expanding_finnish_fold_boundaries() -> None:
    folds = generate_expanding_finnish_folds(2025)
    assert len(folds) == 8
    assert folds[0].train_start.month == 1
    assert folds[0].train_end.month == 3
    assert folds[0].val_start.month == 4
    assert folds[0].test_start.month == 5
    assert folds[-1].train_end.month == 10
    assert folds[-1].val_start.month == 11
    assert folds[-1].test_start.month == 12


def test_smart_persistence_is_train_only_and_finite() -> None:
    timestamps = pd.date_range("2025-06-01", periods=96, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "capacity_factor": np.clip(np.sin(np.linspace(-2, 4, 96)), 0, 1),
            "solar_elevation": np.where(np.arange(96) < 24, -1, 20),
        }
    )
    train = frame.iloc[:80]
    model = HarmonicSmartPersistence().fit(train)
    forecasts = model.predict_long(frame, pd.DatetimeIndex([timestamps[79]]), 8)

    assert model.fitted_through == timestamps[79]
    assert np.isfinite(forecasts["smart_persistence"]).all()
    assert len(forecasts) == 8


def test_physical_context_is_observed_only_and_clipped() -> None:
    frame = _site_frame("physical", 46.0)
    frame["capacity_factor"] = [0.2, np.nan, 1.4, 0.5] + [0.0] * (len(frame) - 4)
    frame["observation_available"] = [True, False, True, True] + [True] * (len(frame) - 4)
    frame["clear_sky_norm"] = 0.5
    prepared = prepare_tft_frame(frame, TFTTransferConfig())
    assert set(PHYSICAL_CONTEXT_REALS).issubset(prepared.columns)
    assert prepared.loc[0, "observed_capacity_factor"] == pytest.approx(0.2)
    assert prepared.loc[0, "clear_sky_index"] == pytest.approx(0.4)
    assert prepared.loc[1, "clear_sky_index_available"] == 0.0
    assert prepared.loc[1, "clear_sky_index"] == 0.0
    assert prepared.loc[2, "clear_sky_index"] == 1.0


def test_common_mask_fails_on_missing_eligible_prediction() -> None:
    frame = _forecast_fixture()
    frame.loc[1, "point_forecast"] = np.nan
    with pytest.raises(ValueError, match="eligible daylight forecasts are missing"):
        common_daylight_mask(frame)


def test_metrics_use_one_common_daylight_mask() -> None:
    frame = _forecast_fixture()
    metrics, pooled = evaluate_long_forecasts(frame, n_bootstrap=4, block_length=2)
    assert pooled["count"] == 2
    assert set(metrics["model"]) == {
        "transfer",
        "transfer_q50",
        "scratch",
        "smart_persistence",
    }
    q50 = metrics.query("model == 'transfer_q50' and grouping == 'overall'").iloc[0]
    assert q50["mae"] == pytest.approx(0.05)
    assert pooled["q50"]["mae"] == pytest.approx(0.05)


def test_optional_legacy_baseline_does_not_change_common_mask() -> None:
    frame = _forecast_fixture()
    expected = common_daylight_mask(frame)
    frame["legacy_harmonic_smart_persistence"] = [np.nan, 0.2, 0.0]
    assert common_daylight_mask(frame).equals(expected)


def test_report_only_acceptance_never_raises() -> None:
    result = assert_skill_gate({"skill": 0.09, "skill_ci_low": 0.01}, threshold=0.10)
    assert result == {
        "skill_threshold": 0.10,
        "point_estimate_meets_threshold": False,
        "ci_lower_bound_meets_threshold": False,
    }


def test_physical_smart_persistence_propagates_issue_index_and_horizons() -> None:
    timestamps = pd.date_range("2025-06-01 10:00", periods=4, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "capacity_factor": [0.25, 0.99, 0.01, 0.75],
            "clear_sky_norm": [0.5, 0.6, 0.8, 1.0],
        }
    )
    result = PhysicalSmartPersistence().predict_long(frame, timestamps[:1], 3)

    assert result["target_timestamp_utc"].tolist() == timestamps[1:].tolist()
    assert result["horizon_step"].tolist() == [1, 2, 3]
    assert result["smart_persistence"].tolist() == pytest.approx([0.3, 0.4, 0.5])


def test_physical_smart_persistence_fallback_clipping_and_no_future_production() -> None:
    timestamps = pd.date_range("2025-06-01", periods=5, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "capacity_factor": [0.9, np.nan, 0.0, 0.2, 0.9],
            "clear_sky_norm": [0.2, 0.0, 0.5, 0.6, 0.8],
        }
    )
    model = PhysicalSmartPersistence()
    clipped = model.predict_long(frame, timestamps[:1], 2)
    night_fallback = model.predict_long(frame, timestamps[1:2], 2)
    missing_issue = frame.copy()
    missing_issue.loc[2, "capacity_factor"] = np.nan
    missing_fallback = model.predict_long(missing_issue, timestamps[2:3], 2)
    future_changed = frame.copy()
    future_changed.loc[1:, "capacity_factor"] = [0.0, 1.0, 0.0, 1.0]

    assert clipped["smart_persistence"].tolist() == pytest.approx([0.0, 1.0])
    assert night_fallback["smart_persistence"].tolist() == pytest.approx([0.5, 0.6])
    assert missing_fallback["smart_persistence"].tolist() == pytest.approx([0.6, 0.8])
    assert model.predict_long(frame, timestamps[:1], 4)["smart_persistence"].equals(
        model.predict_long(future_changed, timestamps[:1], 4)["smart_persistence"]
    )


def test_positive_physical_daylight_does_not_collapse_to_zero() -> None:
    timestamps = pd.date_range("2025-03-01 08:00", periods=12, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "capacity_factor": np.linspace(0.1, 0.5, len(timestamps)),
            "clear_sky_norm": np.linspace(0.2, 0.8, len(timestamps)),
        }
    )
    forecasts = PhysicalSmartPersistence().predict_long(frame, timestamps[:4], 8)
    assert (forecasts["smart_persistence"] > 0).all()


def test_quantiles_are_clamped_and_ordered() -> None:
    values = np.array([[[0.8, -0.1, 0.3, 1.2, 0.5, 0.2, 0.9]]])
    ordered = enforce_monotonic_quantiles(values)
    assert np.diff(ordered, axis=-1).min() >= 0
    assert ordered.min() == 0
    assert ordered.max() == 1


@pytest.mark.filterwarnings("ignore:X does not have valid feature names")
def test_synthetic_multisite_pretrain_reload_and_unseen_site_finetune(tmp_path: Path) -> None:
    pytest.importorskip("pytorch_forecasting")
    config = TFTTransferConfig(
        accelerator="cpu",
        hidden_size=4,
        attention_head_size=1,
        hidden_continuous_size=2,
        batch_size=16,
        max_epochs=1,
        patience=1,
        max_sequences_per_epoch=32,
    )
    source = pd.concat([_site_frame("si-1", 46.0), _site_frame("si-2", 46.5)])
    cutoff = int(source["time_idx"].min() + 47)
    source_train = source[source["time_idx"] <= cutoff]
    source_dataset = create_dataset(source_train, config)
    source_validation = create_dataset(
        source,
        config,
        template=source_dataset,
        min_prediction_idx=cutoff + 1,
    )
    _, source_metadata = fit_tft(
        source_dataset, source_validation, config, tmp_path / "source"
    )

    unseen = _site_frame("fi-unseen", 61.64, longitude=26.78, timezone="Europe/Helsinki")
    unseen_cutoff = int(unseen["time_idx"].min() + 47)
    unseen_train = unseen[unseen["time_idx"] <= unseen_cutoff]
    unseen_dataset = create_dataset(unseen_train, config)
    unseen_validation = create_dataset(
        unseen,
        config,
        template=unseen_dataset,
        min_prediction_idx=unseen_cutoff + 1,
    )
    model, _ = fit_tft(
        unseen_dataset,
        unseen_validation,
        config,
        tmp_path / "unseen",
        pretrained_checkpoint=Path(source_metadata["best_model_path"]),
    )
    predictions = predict_long(model, unseen_validation, unseen, config.batch_size)
    save_model_bundle(
        tmp_path / "bundle",
        model,
        unseen_dataset,
        config,
        {"best_epoch": 1},
        {"fixture": "hash"},
        {"fi-unseen": 1.0},
    )

    assert not unseen_dataset.static_categoricals
    assert set(predictions["horizon_step"]) == set(range(1, 9))
    quantile_columns = ["q02", "q10", "q25", "q50", "q75", "q90", "q98"]
    assert (np.diff(predictions[quantile_columns], axis=1) >= 0).all()
    assert (tmp_path / "bundle" / "manifest.json").exists()
    assert (tmp_path / "bundle" / "dataset_parameters.pt").exists()


def test_satellite_features_are_encoder_only_unknown_reals() -> None:
    pytest.importorskip("pytorch_forecasting")
    frame = _site_frame("si-satellite", 46.0)
    for column in SATELLITE_REALS:
        frame[column] = 0.0
    config = TFTTransferConfig(
        accelerator="cpu", satellite_enabled=True, satellite_encoder_only=True
    )

    dataset = create_dataset(frame, config)

    assert set(SATELLITE_REALS) <= set(dataset.time_varying_unknown_reals)
    assert set(SATELLITE_REALS).isdisjoint(dataset.time_varying_known_reals)


def _site_frame(
    site_id: str,
    latitude: float,
    longitude: float = 14.0,
    timezone: str = "Europe/Ljubljana",
) -> pd.DataFrame:
    timestamps = pd.date_range("2025-06-01", periods=64, freq="15min", tz="UTC")
    energy = np.maximum(0, np.sin((np.arange(64) - 8) / 12)) * 0.25
    frame, _ = canonicalize_site_frame(
        pd.DataFrame({"timestamp_utc": timestamps, "interval_energy_mwh": energy}),
        SiteSpec(site_id, 1.0, latitude, longitude, timezone),
        "synthetic",
    )
    return frame


def _forecast_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_timestamp_utc": pd.date_range(
                "2025-01-01", periods=3, freq="15min", tz="UTC"
            ),
            "horizon_step": [1, 2, 3],
            "fold": [0, 0, 0],
            "solar_elevation": [10.0, 20.0, -1.0],
            "actual_value": [0.2, 0.4, 0.0],
            "point_forecast": [0.3, 0.5, 0.1],
            "q50": [0.25, 0.45, 0.1],
            "scratch_forecast": [0.2, 0.6, 0.1],
            "smart_persistence": [0.4, 0.6, 0.0],
        }
    )
