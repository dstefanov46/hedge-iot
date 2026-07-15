import numpy as np
import pandas as pd


def persistence_forecast(series: pd.Series, lag_steps: int = 1) -> pd.Series:
    """Forecast by repeating the latest observed value."""
    if lag_steps <= 0:
        raise ValueError("lag_steps must be positive.")
    return series.sort_index().shift(lag_steps).reindex(series.index)


def climatology_forecast(series: pd.Series, window: int = 30) -> pd.Series:
    """Rolling-mean climatology baseline."""
    if window <= 0:
        raise ValueError("window must be positive.")
    return series.sort_index().rolling(window=window, min_periods=1).mean()


def smart_persistence_forecast(
    series: pd.Series,
    clear_sky_series: pd.Series,
    lag_steps: int = 1,
    clip_nonnegative: bool = True,
) -> pd.Series:
    """One-step smart persistence using clear-sky index."""
    y = series.sort_index()
    clear_sky = clear_sky_series.sort_index().reindex(y.index)

    with np.errstate(divide="ignore", invalid="ignore"):
        clear_sky_index = y.shift(lag_steps) / clear_sky.shift(lag_steps)
        forecast = clear_sky_index * clear_sky

    if clip_nonnegative:
        forecast = forecast.clip(lower=0)
    return forecast.reindex(y.index)


def multi_horizon_smart_persistence(
    series: pd.Series,
    clear_sky_series: pd.Series,
    horizon_steps: int,
    lag_steps: int = 1,
    clip_nonnegative: bool = True,
) -> pd.DataFrame:
    """Multi-horizon smart persistence with columns `horizon_1..horizon_n`."""
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive.")

    y = series.sort_index()
    clear_sky = clear_sky_series.sort_index().reindex(y.index.union(clear_sky_series.index))
    y = y.reindex(clear_sky.index)

    with np.errstate(divide="ignore", invalid="ignore"):
        clear_sky_index = y.shift(lag_steps) / clear_sky.shift(lag_steps)

    columns: dict[str, pd.Series] = {}
    for horizon in range(1, horizon_steps + 1):
        forecast = (clear_sky_index * clear_sky.shift(-(horizon - 1))).astype(float)
        if clip_nonnegative:
            forecast = forecast.clip(lower=0)
        columns[f"horizon_{horizon}"] = forecast

    return pd.DataFrame(columns).reindex(series.index)
