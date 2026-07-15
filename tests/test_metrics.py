import pandas as pd

from aurora.evaluation.metrics import mae, mbe, rmse


def test_forecast_metrics() -> None:
    actual = pd.Series([1.0, 2.0, 3.0])
    forecast = pd.Series([1.0, 3.0, 5.0])

    assert mae(actual, forecast) == 1.0
    assert round(rmse(actual, forecast), 6) == 1.290994
    assert mbe(actual, forecast) == 1.0
