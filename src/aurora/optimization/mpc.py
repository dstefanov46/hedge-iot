from dataclasses import dataclass

import pandas as pd

from aurora.config import BatteryConfig, OptimizationWeights


@dataclass(frozen=True)
class ScheduleResult:
    schedule: pd.DataFrame
    objective_value: float | None = None


def build_bess_schedule(
    forecast: pd.DataFrame,
    battery: BatteryConfig,
    weights: OptimizationWeights | None = None,
) -> ScheduleResult:
    """Placeholder for receding-horizon BESS scheduling.

    The spec calls for configurable objectives: grid exchange minimization,
    self-consumption maximization, and battery degradation penalties.
    """
    if "forecast" not in forecast.columns:
        raise KeyError("Forecast frame must contain a 'forecast' column.")
    if battery.capacity_mwh <= 0:
        raise ValueError("Battery capacity must be positive.")

    _ = weights or OptimizationWeights()
    schedule = pd.DataFrame(
        {
            "forecast": forecast["forecast"],
            "charge_mw": 0.0,
            "discharge_mw": 0.0,
            "state_of_charge_mwh": battery.capacity_mwh * battery.min_state_of_charge,
        },
        index=forecast.index,
    )
    return ScheduleResult(schedule=schedule)
