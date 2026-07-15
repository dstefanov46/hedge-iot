import numpy as np
import pandas as pd


def mae(actual: pd.Series, forecast: pd.Series) -> float:
    return float(np.mean(np.abs(actual.to_numpy() - forecast.to_numpy())))


def rmse(actual: pd.Series, forecast: pd.Series) -> float:
    return float(np.sqrt(np.mean((actual.to_numpy() - forecast.to_numpy()) ** 2)))


def mbe(actual: pd.Series, forecast: pd.Series) -> float:
    return float(np.mean(forecast.to_numpy() - actual.to_numpy()))
