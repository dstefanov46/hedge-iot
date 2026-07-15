import pandas as pd


class PersistenceForecaster:
    """Naive baseline that repeats the latest observed target value."""

    def __init__(self) -> None:
        self._last_value: float | None = None

    def fit(self, frame: pd.DataFrame, target: str) -> None:
        if target not in frame.columns:
            raise KeyError(f"Missing target column: {target}")
        series = frame[target].dropna()
        if series.empty:
            raise ValueError(f"Target column has no usable values: {target}")
        self._last_value = float(series.iloc[-1])

    def predict(self, frame: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
        if self._last_value is None:
            raise RuntimeError("Forecaster must be fitted before predict().")
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive.")

        index = (
            frame.index[:horizon_steps]
            if len(frame.index) >= horizon_steps
            else range(horizon_steps)
        )
        return pd.DataFrame({"forecast": [self._last_value] * horizon_steps}, index=index)
