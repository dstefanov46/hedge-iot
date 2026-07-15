from pathlib import Path

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProjectPaths(BaseModel):
    root: Path = PROJECT_ROOT
    raw_data: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_data: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    models: Path = Field(default_factory=lambda: PROJECT_ROOT / "models")
    outputs: Path = Field(default_factory=lambda: PROJECT_ROOT / "outputs")


class ForecastConfig(BaseModel):
    horizon_steps: int = 96
    frequency: str = "15min"
    target_column: str = "actual_mwh"


class BatteryConfig(BaseModel):
    capacity_mwh: float
    max_charge_mw: float
    max_discharge_mw: float
    round_trip_efficiency: float = 0.9
    min_state_of_charge: float = 0.1
    max_state_of_charge: float = 0.9


class OptimizationWeights(BaseModel):
    grid_exchange: float = 1.0
    self_consumption: float = 1.0
    battery_degradation: float = 0.1


class Settings(BaseModel):
    paths: ProjectPaths = Field(default_factory=ProjectPaths)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)


settings = Settings()
