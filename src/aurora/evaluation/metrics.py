import numpy as np
import pandas as pd


def mae(actual: pd.Series, forecast: pd.Series) -> float:
    return float(np.mean(np.abs(actual.to_numpy() - forecast.to_numpy())))


def rmse(actual: pd.Series, forecast: pd.Series) -> float:
    return float(np.sqrt(np.mean((actual.to_numpy() - forecast.to_numpy()) ** 2)))


def mbe(actual: pd.Series, forecast: pd.Series) -> float:
    return float(np.mean(forecast.to_numpy() - actual.to_numpy()))


def mape(actual: pd.Series, forecast: pd.Series, eps: float = 1e-6) -> float:
    denominator = np.maximum(np.abs(actual.to_numpy()), eps)
    return float(np.mean(np.abs((actual.to_numpy() - forecast.to_numpy()) / denominator)))


def mse(actual: pd.Series, forecast: pd.Series) -> float:
    return float(np.mean((actual.to_numpy() - forecast.to_numpy()) ** 2))


def nrmse(actual: pd.Series, forecast: pd.Series, normalization_factor: float) -> float:
    if normalization_factor == 0:
        raise ValueError("normalization_factor cannot be zero.")
    return rmse(actual, forecast) / normalization_factor


def skill_score(
    actual: pd.Series,
    forecast: pd.Series,
    reference_forecast: pd.Series,
    metric: str = "mse",
) -> float:
    if metric == "mse":
        model_error = mse(actual, forecast)
        reference_error = mse(actual, reference_forecast)
    elif metric == "mae":
        model_error = mae(actual, forecast)
        reference_error = mae(actual, reference_forecast)
    elif metric == "rmse":
        model_error = rmse(actual, forecast)
        reference_error = rmse(actual, reference_forecast)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if reference_error == 0:
        return 1.0 if model_error == 0 else float("-inf")
    return float(1.0 - model_error / reference_error)
