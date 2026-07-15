import pandas as pd

from aurora.forecasting.baselines import multi_horizon_smart_persistence, persistence_forecast


def test_persistence_forecast_shifts_series() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="15min", tz="UTC")
    series = pd.Series([1.0, 2.0, 3.0], index=index)

    forecast = persistence_forecast(series)

    assert pd.isna(forecast.iloc[0])
    assert forecast.iloc[1:].tolist() == [1.0, 2.0]


def test_multi_horizon_smart_persistence_columns() -> None:
    index = pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC")
    series = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0], index=index)
    clear_sky = pd.Series([1.0, 2.0, 4.0, 6.0, 8.0], index=index)

    forecast = multi_horizon_smart_persistence(series, clear_sky, horizon_steps=2)

    assert list(forecast.columns) == ["horizon_1", "horizon_2"]
    assert len(forecast) == 5
