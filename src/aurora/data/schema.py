from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

QualityFlag = Literal[
    "ok",
    "missing",
    "estimated",
    "linked_workbook_unresolved",
    "malformed_timestamp",
]

CANONICAL_PV_SCHEMA_VERSION = "pv-15min-v1"
CANONICAL_PV_COLUMNS = (
    "site_id",
    "timestamp_utc",
    "interval_energy_mwh",
    "installed_capacity_mw",
    "capacity_factor",
    "latitude",
    "longitude",
    "timezone",
    "quality_status",
    "source",
    "source_sheet",
    "source_row",
    "planned_mwh",
    "observation_available",
    "time_idx",
)


class SiteMetadata(BaseModel):
    site_id: str
    name: str
    latitude: float | None = None
    longitude: float | None = None
    timezone: str = "UTC"
    installed_capacity_mw: float | None = Field(default=None, gt=0)
    grid_connection_id: str | None = None
    bess_id: str | None = None


class ProductionObservation(BaseModel):
    site_id: str
    timestamp_utc: datetime
    interval_minutes: int = Field(default=15, gt=0)
    planned_mwh: float | None = None
    actual_mwh: float | None = None
    source: str
    source_sheet: str | None = None
    source_row: int | None = Field(default=None, gt=0)
    quality_flag: QualityFlag = "ok"


class WeatherObservation(BaseModel):
    site_id: str
    timestamp_utc: datetime
    provider: str
    forecast_reference_utc: datetime | None = None
    horizon_hours: float | None = Field(default=None, ge=0)
    ghi_w_m2: float | None = Field(default=None, ge=0)
    dni_w_m2: float | None = Field(default=None, ge=0)
    cloud_cover_pct: float | None = Field(default=None, ge=0, le=100)
    temperature_c: float | None = None
    wind_speed_m_s: float | None = Field(default=None, ge=0)
    humidity_pct: float | None = Field(default=None, ge=0, le=100)


class SatelliteFeatureObservation(BaseModel):
    site_id: str
    timestamp_utc: datetime
    provider: str
    product: str
    tile_id: str | None = None
    cloud_index: float | None = None
    irradiance_estimate_w_m2: float | None = Field(default=None, ge=0)
    feature_vector_uri: str | None = None


class SatellitePatchObservation(BaseModel):
    site_id: str
    timestamp_utc: datetime
    product_id: str
    provider: str = "EUMETSAT"
    product: str
    patch_uri: str
    channel_names: list[str]
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    missing_fraction: float = Field(ge=0, le=1)
    quality_status: str
    normalization_version: str


class ForecastRecord(BaseModel):
    site_id: str
    created_at_utc: datetime
    target_timestamp_utc: datetime
    horizon_steps: int = Field(gt=0)
    forecast_mwh: float
    p10_mwh: float | None = None
    p50_mwh: float | None = None
    p90_mwh: float | None = None
    model_name: str
    model_version: str | None = None

    @field_validator("p90_mwh")
    @classmethod
    def validate_quantiles(cls, value: float | None, info):
        data = info.data
        p10 = data.get("p10_mwh")
        p50 = data.get("p50_mwh")
        if value is not None and p10 is not None and value < p10:
            raise ValueError("p90_mwh must be greater than or equal to p10_mwh")
        if value is not None and p50 is not None and value < p50:
            raise ValueError("p90_mwh must be greater than or equal to p50_mwh")
        return value


class BessConstraints(BaseModel):
    bess_id: str
    site_id: str
    capacity_mwh: float = Field(gt=0)
    max_charge_mw: float = Field(gt=0)
    max_discharge_mw: float = Field(gt=0)
    round_trip_efficiency: float = Field(default=0.9, gt=0, le=1)
    min_state_of_charge: float = Field(default=0.1, ge=0, le=1)
    max_state_of_charge: float = Field(default=0.9, ge=0, le=1)
    initial_state_of_charge_mwh: float | None = Field(default=None, ge=0)

    @field_validator("max_state_of_charge")
    @classmethod
    def validate_soc_bounds(cls, value: float, info):
        min_soc = info.data.get("min_state_of_charge")
        if min_soc is not None and value <= min_soc:
            raise ValueError("max_state_of_charge must be greater than min_state_of_charge")
        return value


class BessScheduleRecord(BaseModel):
    bess_id: str
    site_id: str
    timestamp_utc: datetime
    interval_minutes: int = Field(default=15, gt=0)
    charge_mw: float = Field(ge=0)
    discharge_mw: float = Field(ge=0)
    state_of_charge_mwh: float = Field(ge=0)
    grid_import_mwh: float | None = Field(default=None, ge=0)
    grid_export_mwh: float | None = Field(default=None, ge=0)
    objective_value: float | None = None
