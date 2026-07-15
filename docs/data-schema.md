# AURORA Canonical Data Schema

All timestamps use timezone-aware UTC datetimes at storage/API boundaries. Site-local time can be retained as an optional derived field for reporting.

## Site Metadata

| Column | Type | Unit | Required | Description |
| --- | --- | --- | --- | --- |
| `site_id` | string | none | yes | Stable unique site identifier, for example `hirvensalmi`. |
| `name` | string | none | yes | Human-readable site name. |
| `latitude` | float | degrees | no | WGS84 latitude. |
| `longitude` | float | degrees | no | WGS84 longitude. |
| `timezone` | string | IANA timezone | yes | Site-local timezone, for example `Europe/Helsinki`. |
| `installed_capacity_mw` | float | MW | no | PV installed AC/DC capacity used for normalization. |
| `grid_connection_id` | string | none | no | Optional HEDGE-IoT/grid integration identifier. |
| `bess_id` | string | none | no | Optional linked BESS identifier. |

## Production Observation

| Column | Type | Unit | Required | Description |
| --- | --- | --- | --- | --- |
| `site_id` | string | none | yes | Site identifier. |
| `timestamp_utc` | datetime | UTC | yes | Start timestamp for the interval. |
| `interval_minutes` | integer | minutes | yes | Interval length; workbook source uses 15. |
| `planned_mwh` | float | MWh | no | Production plan / `Tuotantosuunnitelma`. |
| `actual_mwh` | float | MWh | no | Actual production / `Toteutunut tuotanto`. |
| `source` | string | none | yes | Source artifact or system name. |
| `source_sheet` | string | none | no | Workbook sheet or upstream table. |
| `quality_flag` | string | none | no | Optional quality status such as `ok`, `missing`, `estimated`, `linked_workbook_unresolved`. |

Workbook mapping:

| Sheet Pattern | Header Rows | Timestamp Source | Planned Column | Actual Column |
| --- | --- | --- | --- | --- |
| `Q1_25` | rows 1-2 | `A` Excel date serial + `B` interval start | `C` | `D` |
| `Q2_25`-`Q4_25` | rows 1-3 | `A` timestamp/cache, with quarter sequence fallback | `F` | `G` |

## Weather / NWP Observation

| Column | Type | Unit | Required | Description |
| --- | --- | --- | --- | --- |
| `site_id` | string | none | yes | Site identifier. |
| `timestamp_utc` | datetime | UTC | yes | Forecast-valid or observation timestamp. |
| `provider` | string | none | yes | NWP/weather provider. |
| `forecast_reference_utc` | datetime | UTC | no | Forecast initialization time for NWP data. |
| `horizon_hours` | float | hours | no | Forecast lead time. |
| `ghi_w_m2` | float | W/m2 | no | Global horizontal irradiance. |
| `dni_w_m2` | float | W/m2 | no | Direct normal irradiance. |
| `cloud_cover_pct` | float | percent | no | Cloud cover. |
| `temperature_c` | float | C | no | Air temperature. |
| `wind_speed_m_s` | float | m/s | no | Wind speed. |
| `humidity_pct` | float | percent | no | Relative humidity. |

## Satellite Feature Observation

| Column | Type | Unit | Required | Description |
| --- | --- | --- | --- | --- |
| `site_id` | string | none | yes | Site identifier or nearest site for tile-derived features. |
| `timestamp_utc` | datetime | UTC | yes | Satellite acquisition or derived-feature timestamp. |
| `provider` | string | none | yes | Satellite/source provider. |
| `product` | string | none | yes | Product name or processing pipeline. |
| `tile_id` | string | none | no | Geospatial tile identifier. |
| `cloud_index` | float | index | no | Cloud feature derived from imagery. |
| `irradiance_estimate_w_m2` | float | W/m2 | no | Satellite-derived irradiance estimate. |
| `feature_vector_uri` | string | URI/path | no | Pointer to dense image or embedding features. |

## Forecast Record

| Column | Type | Unit | Required | Description |
| --- | --- | --- | --- | --- |
| `site_id` | string | none | yes | Site identifier. |
| `created_at_utc` | datetime | UTC | yes | Forecast generation timestamp. |
| `target_timestamp_utc` | datetime | UTC | yes | Timestamp being forecast. |
| `horizon_steps` | integer | steps | yes | Number of 15-minute steps from creation time. |
| `forecast_mwh` | float | MWh | yes | Point forecast for interval production. |
| `p10_mwh` | float | MWh | no | Lower probabilistic quantile. |
| `p50_mwh` | float | MWh | no | Median probabilistic quantile. |
| `p90_mwh` | float | MWh | no | Upper probabilistic quantile. |
| `model_name` | string | none | yes | Model identifier, for example `persistence` or `tft`. |
| `model_version` | string | none | no | Model artifact/version identifier. |

## BESS Constraints

| Column | Type | Unit | Required | Description |
| --- | --- | --- | --- | --- |
| `bess_id` | string | none | yes | Battery identifier. |
| `site_id` | string | none | yes | Linked site. |
| `capacity_mwh` | float | MWh | yes | Usable or nominal energy capacity. |
| `max_charge_mw` | float | MW | yes | Charge power limit. |
| `max_discharge_mw` | float | MW | yes | Discharge power limit. |
| `round_trip_efficiency` | float | fraction | yes | Round-trip efficiency, 0-1. |
| `min_state_of_charge` | float | fraction | yes | Minimum allowed SoC, 0-1. |
| `max_state_of_charge` | float | fraction | yes | Maximum allowed SoC, 0-1. |
| `initial_state_of_charge_mwh` | float | MWh | no | Initial SoC at optimization start. |

## BESS Schedule Record

| Column | Type | Unit | Required | Description |
| --- | --- | --- | --- | --- |
| `bess_id` | string | none | yes | Battery identifier. |
| `site_id` | string | none | yes | Site identifier. |
| `timestamp_utc` | datetime | UTC | yes | Interval start timestamp. |
| `interval_minutes` | integer | minutes | yes | Scheduling interval length. |
| `charge_mw` | float | MW | yes | Charge command. |
| `discharge_mw` | float | MW | yes | Discharge command. |
| `state_of_charge_mwh` | float | MWh | yes | End-of-interval or modeled SoC. |
| `grid_import_mwh` | float | MWh | no | Modeled grid import. |
| `grid_export_mwh` | float | MWh | no | Modeled grid export. |
| `objective_value` | float | none | no | Total or per-step optimization objective value. |
