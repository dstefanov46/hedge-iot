import numpy as np
import pandas as pd

from aurora.experiments.tft_plan_comparison import (
    assign_splits,
    forecast_diagnostics,
    point_metrics,
    quantile_loss_from_frame,
)


def test_assign_splits_uses_chronological_fractions() -> None:
    frame = pd.DataFrame({"time_idx": range(10)})

    result = assign_splits(frame)

    assert result["split"].tolist() == [
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
    ]


def test_point_metrics_skill_against_reference() -> None:
    actual = pd.Series([1.0, 2.0, 3.0])
    forecast = pd.Series([1.0, 2.0, 4.0])
    reference = pd.Series([1.0, 3.0, 5.0])

    metrics = point_metrics(actual, forecast, reference)

    assert metrics["mse"] == 1 / 3
    assert metrics["mae"] == 1 / 3
    assert round(metrics["skill_score"], 6) == 0.8


def test_quantile_loss_from_frame_uses_tft_quantiles() -> None:
    frame = pd.DataFrame({"actual_mwh": [1.0]})
    for q in [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]:
        frame[f"tft_q{int(q * 100):02d}_mwh"] = [1.0]

    assert quantile_loss_from_frame(frame) == 0.0


def test_forecast_diagnostics_flags_night_positive_forecasts() -> None:
    stats = forecast_diagnostics(
        actual=pd.Series([0.0, 1.0]),
        forecast=pd.Series([0.1, 1.2]),
        clear_sky=pd.Series([0.0, 100.0]),
    )

    assert stats["count"] == 2
    assert stats["night_positive_share"] == 1.0
    assert np.isclose(stats["bias_mean"], 0.15)
