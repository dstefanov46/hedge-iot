from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

POINT_COLUMNS = ("point_forecast", "q50", "scratch_forecast", "smart_persistence")
BASELINE_IDENTITY_PHYSICAL = {
    "name": "physical_clear_sky_smart_persistence",
    "column": "smart_persistence",
    "clear_sky_feature": "clear_sky_norm",
    "legacy": False,
}
BASELINE_IDENTITY_LEGACY = {
    "name": "harmonic_smart_persistence",
    "column": "smart_persistence",
    "legacy": True,
}


def common_daylight_mask(
    frame: pd.DataFrame,
    prediction_columns: Iterable[str] = POINT_COLUMNS,
) -> pd.Series:
    prediction_columns = tuple(prediction_columns)
    columns = ["actual_value", "solar_elevation", *prediction_columns]
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {sorted(missing)}")
    eligible_target = (frame["solar_elevation"] > 0) & frame["actual_value"].notna()
    incomplete = eligible_target & frame[list(prediction_columns)].isna().any(axis=1)
    if incomplete.any():
        sample = frame.loc[incomplete, ["target_timestamp_utc", "horizon_step"]].head()
        raise ValueError(
            f"{int(incomplete.sum())} eligible daylight forecasts are missing. "
            f"First rows: {sample.to_dict(orient='records')}"
        )
    return eligible_target & frame[list(prediction_columns)].notna().all(axis=1)


def metric_summary(frame: pd.DataFrame, model_column: str) -> dict[str, float]:
    actual = frame["actual_value"].to_numpy(dtype=float)
    forecast = frame[model_column].to_numpy(dtype=float)
    reference = frame["smart_persistence"].to_numpy(dtype=float)
    errors = forecast - actual
    model_mse = float(np.mean(errors**2))
    reference_mse = float(np.mean((reference - actual) ** 2))
    return {
        "count": int(len(frame)),
        "mse": model_mse,
        "rmse": float(np.sqrt(model_mse)),
        "mae": float(np.mean(np.abs(errors))),
        "mbe": float(np.mean(errors)),
        "skill": 1 - model_mse / reference_mse if reference_mse else float("nan"),
    }


def evaluate_long_forecasts(
    forecasts: pd.DataFrame,
    n_bootstrap: int = 100,
    block_length: int = 15,
    seed: int = 42,
    skill_threshold: float = 0.10,
    baseline_identity: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mask = common_daylight_mask(forecasts)
    eligible = forecasts.loc[mask].copy()
    eligible["month"] = (
        pd.to_datetime(eligible["target_timestamp_utc"], utc=True)
        .dt.tz_convert("Europe/Helsinki")
        .dt.month
    )
    models = [
        ("transfer", "point_forecast", True),
        ("transfer_q50", "q50", True),
        ("scratch", "scratch_forecast", True),
        ("smart_persistence", "smart_persistence", True),
        ("legacy_harmonic_smart_persistence", "legacy_harmonic_smart_persistence", False),
        ("existing_plan", "existing_plan", False),
    ]
    rows: list[dict[str, object]] = []
    for model_name, column, required in models:
        if column not in eligible:
            if required:
                raise ValueError(f"Missing evaluation column: {column}")
            continue
        model_eligible = eligible if required else eligible.loc[eligible[column].notna()]
        if model_eligible.empty:
            continue
        for grouping, keys in [
            ("overall", []),
            ("horizon", ["horizon_step"]),
            ("month", ["month"]),
            ("fold", ["fold"]),
        ]:
            groups = [((), model_eligible)] if not keys else model_eligible.groupby(keys, sort=True)
            for key, group in groups:
                values = metric_summary(group, column)
                key_values = key if isinstance(key, tuple) else (key,)
                row = {"model": model_name, "grouping": grouping, **values}
                row.update(dict(zip(keys, key_values, strict=True)))
                rows.append(row)
    report = pd.DataFrame(rows)

    pooled: dict[str, Any] = metric_summary(eligible, "point_forecast")
    pooled.update(
        moving_block_intervals(
            eligible,
            model_column="point_forecast",
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            seed=seed,
        )
    )
    q50 = metric_summary(eligible, "q50")
    q50.update(
        moving_block_intervals(
            eligible,
            model_column="q50",
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            seed=seed,
        )
    )
    pooled["q50"] = q50
    pooled["baseline"] = baseline_identity or BASELINE_IDENTITY_PHYSICAL.copy()
    pooled["diagnostics"] = baseline_diagnostics(eligible)
    pooled["acceptance"] = acceptance_results(pooled, skill_threshold)
    return report, pooled


def acceptance_results(summary: dict[str, Any], threshold: float = 0.10) -> dict[str, Any]:
    """Return both interpretations of the skill threshold without enforcing either."""
    skill = float(summary["skill"])
    ci_low = float(summary.get("skill_ci_low", float("nan")))
    return {
        "skill_threshold": float(threshold),
        "point_estimate_meets_threshold": bool(np.isfinite(skill) and skill >= threshold),
        "ci_lower_bound_meets_threshold": bool(np.isfinite(ci_low) and ci_low >= threshold),
    }


def assert_skill_gate(summary: dict[str, Any], threshold: float = 0.10) -> dict[str, Any]:
    """Compatibility alias for the former gate; acceptance is report-only."""
    return acceptance_results(summary, threshold)


def baseline_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize suspicious zero forecasts overall and by evaluation fold."""
    overall = _baseline_diagnostic_summary(frame)
    folds: dict[str, dict[str, float | int]] = {}
    if "fold" in frame:
        folds = {
            str(fold): _baseline_diagnostic_summary(group)
            for fold, group in frame.groupby("fold", sort=True)
        }
    for label, values in [("overall", overall), *folds.items()]:
        suspicious = (
            values["daylight_zero_forecast_rate"] > 0.50
            or values["zero_forecast_actual_above_0_1_count"] > 0
            or values["mse_contribution_from_zero_forecasts"] > 0.50
        )
        if suspicious:
            warnings.warn(
                "Suspicious smart-persistence zero forecasts for "
                f"{label}: {values}",
                RuntimeWarning,
                stacklevel=2,
            )
    return {"overall": overall, "folds": folds}


def _baseline_diagnostic_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    count = len(frame)
    forecast = frame["smart_persistence"].to_numpy(dtype=float)
    actual = frame["actual_value"].to_numpy(dtype=float)
    zero = np.isclose(forecast, 0.0, atol=1e-12)
    large_actual = zero & (actual > 0.1)
    squared_error = (forecast - actual) ** 2
    total_squared_error = float(squared_error.sum())
    zero_squared_error = float(squared_error[zero].sum())
    return {
        "daylight_count": int(count),
        "daylight_zero_forecast_count": int(zero.sum()),
        "daylight_zero_forecast_rate": float(zero.mean()) if count else float("nan"),
        "zero_forecast_actual_above_0_1_count": int(large_actual.sum()),
        "zero_forecast_actual_above_0_1_rate": float(large_actual.mean())
        if count
        else float("nan"),
        "zero_forecast_mse": float(np.mean(squared_error * zero))
        if count
        else float("nan"),
        "mse_contribution_from_zero_forecasts": zero_squared_error / total_squared_error
        if total_squared_error
        else 0.0,
    }


def moving_block_intervals(
    frame: pd.DataFrame,
    model_column: str,
    n_bootstrap: int = 100,
    block_length: int = 15,
    seed: int = 42,
) -> dict[str, float]:
    if n_bootstrap <= 0 or block_length <= 0:
        raise ValueError("n_bootstrap and block_length must be positive.")
    n = len(frame)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty forecast frame.")
    candidate_blocks = _temporal_blocks(frame, block_length)
    rng = np.random.default_rng(seed)
    samples: list[dict[str, float]] = []
    for _ in range(n_bootstrap):
        selected: list[np.ndarray] = []
        selected_rows = 0
        while selected_rows < n:
            block = candidate_blocks[int(rng.integers(0, len(candidate_blocks)))]
            selected.append(block)
            selected_rows += len(block)
        indices = np.concatenate(selected)[:n]
        samples.append(metric_summary(frame.iloc[indices], model_column))
    out: dict[str, float] = {}
    for metric in ("mse", "rmse", "mae", "mbe", "skill"):
        values = np.asarray([sample[metric] for sample in samples], dtype=float)
        out[f"{metric}_ci_low"] = float(np.nanpercentile(values, 2.5))
        out[f"{metric}_ci_high"] = float(np.nanpercentile(values, 97.5))
    return out


def _temporal_blocks(frame: pd.DataFrame, block_length: int) -> list[np.ndarray]:
    """Build contiguous 15-issue blocks, retaining every horizon for each issue."""
    if "issue_timestamp_utc" not in frame:
        length = min(block_length, len(frame))
        return [np.arange(start, start + length) for start in range(len(frame) - length + 1)]
    ordered = frame.reset_index(drop=True).sort_values(
        [column for column in ("fold", "issue_timestamp_utc", "horizon_step") if column in frame]
    )
    issue_groups = [
        group.index.to_numpy(dtype=int)
        for _, group in ordered.groupby(
            [column for column in ("fold", "issue_timestamp_utc") if column in ordered],
            sort=False,
        )
    ]
    length = min(block_length, len(issue_groups))
    return [
        np.concatenate(issue_groups[start : start + length])
        for start in range(len(issue_groups) - length + 1)
    ]
