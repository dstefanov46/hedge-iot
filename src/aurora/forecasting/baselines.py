import numpy as np
import pandas as pd

from aurora.features.time import harmonic_features


class PhysicalSmartPersistence:
    """Leakage-safe smart persistence based on the physical clear-sky feature.

    The only production value used for an issue is the observation at that exact
    issue timestamp. Future values come exclusively from ``clear_sky_norm``.
    """

    def __init__(self, epsilon: float = 1e-4, frequency: str = "15min") -> None:
        self.epsilon = epsilon
        self.frequency = frequency

    def predict_long(
        self,
        history: pd.DataFrame,
        issue_timestamps: pd.DatetimeIndex,
        horizon_steps: int = 8,
    ) -> pd.DataFrame:
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive.")
        required = {"timestamp_utc", "capacity_factor", "clear_sky_norm"}
        missing = required.difference(history.columns)
        if missing:
            raise ValueError(f"Missing physical smart-persistence columns: {sorted(missing)}")

        deterministic = history[list(required)].copy()
        deterministic["timestamp_utc"] = pd.to_datetime(
            deterministic["timestamp_utc"], utc=True
        )
        if deterministic["timestamp_utc"].duplicated().any():
            raise ValueError("Physical smart persistence requires unique timestamps.")
        indexed = deterministic.set_index("timestamp_utc")
        issues = pd.DatetimeIndex(pd.to_datetime(issue_timestamps, utc=True)).drop_duplicates()

        rows: list[dict[str, object]] = []
        for issue in issues:
            if issue in indexed.index:
                issue_actual = indexed.at[issue, "capacity_factor"]
                issue_clear_sky = indexed.at[issue, "clear_sky_norm"]
            else:
                issue_actual = np.nan
                issue_clear_sky = np.nan
            if (
                not np.isfinite(issue_actual)
                or not np.isfinite(issue_clear_sky)
                or float(issue_clear_sky) <= self.epsilon
            ):
                clear_sky_index = 1.0
            else:
                clear_sky_index = float(issue_actual) / float(issue_clear_sky)
            clear_sky_index = float(np.clip(clear_sky_index, 0, 2))

            targets = pd.date_range(
                issue + pd.Timedelta(self.frequency),
                periods=horizon_steps,
                freq=self.frequency,
            )
            target_clear_sky = indexed["clear_sky_norm"].reindex(targets).to_numpy(dtype=float)
            forecasts = np.clip(clear_sky_index * target_clear_sky, 0, 1)
            for horizon, (target, forecast) in enumerate(
                zip(targets, forecasts, strict=True), start=1
            ):
                rows.append(
                    {
                        "issue_timestamp_utc": issue,
                        "target_timestamp_utc": target,
                        "horizon_step": horizon,
                        "smart_persistence": float(forecast),
                    }
                )
        return pd.DataFrame(rows)


class HarmonicSmartPersistence:
    """Legacy q=0.99 harmonic baseline retained for v1 reproduction and audit."""

    def __init__(
        self,
        quantile: float = 0.99,
        epsilon: float = 1e-4,
        timezone: str = "Europe/Helsinki",
    ) -> None:
        self.quantile = quantile
        self.epsilon = epsilon
        self.timezone = timezone
        self.model = None
        self.fitted_through: pd.Timestamp | None = None

    def fit(self, frame: pd.DataFrame) -> "HarmonicSmartPersistence":
        from sklearn.linear_model import QuantileRegressor

        required = {"timestamp_utc", "capacity_factor", "solar_elevation"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing smart-persistence columns: {sorted(missing)}")
        train = frame.loc[
            frame["capacity_factor"].notna() & (frame["solar_elevation"] > 0)
        ].copy()
        if train.empty:
            raise ValueError("Cannot fit clear-sky curve without daylight observations.")
        x = self._features(train["timestamp_utc"])
        self.model = QuantileRegressor(quantile=self.quantile, alpha=0, solver="highs")
        self.model.fit(x, train["capacity_factor"].to_numpy())
        self.fitted_through = pd.to_datetime(train["timestamp_utc"], utc=True).max()
        return self

    def clear_sky_curve(self, timestamps: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Fit HarmonicSmartPersistence before prediction.")
        return np.clip(self.model.predict(self._features(timestamps)), 0, 1)

    def predict_long(
        self,
        history: pd.DataFrame,
        issue_timestamps: pd.DatetimeIndex,
        horizon_steps: int = 8,
    ) -> pd.DataFrame:
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive.")
        indexed = history.copy()
        indexed["timestamp_utc"] = pd.to_datetime(indexed["timestamp_utc"], utc=True)
        indexed = indexed.set_index("timestamp_utc")
        rows: list[dict[str, object]] = []
        for issue in pd.DatetimeIndex(issue_timestamps):
            issue = issue.tz_localize("UTC") if issue.tz is None else issue.tz_convert("UTC")
            actual_issue = indexed["capacity_factor"].get(issue, np.nan)
            issue_curve = float(self.clear_sky_curve(pd.DatetimeIndex([issue]))[0])
            if not np.isfinite(actual_issue):
                sky_index = 1.0
            elif issue_curve > self.epsilon:
                sky_index = float(actual_issue) / issue_curve
            else:
                # At night there is no stable denominator. A neutral clear-sky index
                # gives finite sunrise forecasts and zero forecasts while the curve is zero.
                sky_index = 1.0
            sky_index = float(np.clip(sky_index, 0, 2))
            targets = pd.date_range(
                issue + pd.Timedelta(minutes=15), periods=horizon_steps, freq="15min"
            )
            target_curve = self.clear_sky_curve(targets)
            pairs = zip(targets, target_curve, strict=True)
            for horizon, (target, curve) in enumerate(pairs, start=1):
                rows.append(
                    {
                        "issue_timestamp_utc": issue,
                        "target_timestamp_utc": target,
                        "horizon_step": horizon,
                        "smart_persistence": float(np.clip(sky_index * curve, 0, 1)),
                    }
                )
        return pd.DataFrame(rows)

    def _features(self, timestamps: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
        index = index.tz_convert(self.timezone)
        return harmonic_features(index).to_numpy(dtype=float)


def enforce_monotonic_quantiles(values: np.ndarray) -> np.ndarray:
    """Clamp capacity factors and sort the final quantile axis."""
    return np.sort(np.clip(np.asarray(values, dtype=float), 0, 1), axis=-1)


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
