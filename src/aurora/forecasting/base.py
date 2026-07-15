from typing import Protocol

import pandas as pd


class Forecaster(Protocol):
    """Common interface for deterministic and probabilistic forecasters."""

    def fit(self, frame: pd.DataFrame, target: str) -> None:
        """Fit model on historical data."""

    def predict(self, frame: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
        """Return forecast values for a future horizon."""
