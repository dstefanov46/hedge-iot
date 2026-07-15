# AURORA Requirements

This document extracts formal project requirements from `D1.1_AURORA_System_Specification_v4.docx` and the pilot workbook `Hirvensalmi_tuotanto2025.xlsx`.

## Functional Requirements

| ID | Requirement | Source Evidence |
| --- | --- | --- |
| FR-001 | Ingest multimodal data for pilot solar forecasting. | The system specification defines a data layer for ingesting, preprocessing, aligning, and storing multimodal inputs. |
| FR-002 | Ingest historical and near-real-time PV production data at the pilot-site level. | The Hirvensalmi workbook contains quarter-level 15-minute production plan and actual production records. |
| FR-003 | Ingest numerical weather prediction and meteorological variables used for solar forecasting. | The specification lists NWP/weather inputs as part of the AURORA data sources. |
| FR-004 | Ingest satellite-derived irradiance or imagery features for geographically transferable forecasting. | The specification emphasizes satellite images as pan-European inputs and as key IP from the prior Sunesis work. |
| FR-005 | Align all source streams to a canonical time grid before modeling. | The architecture assigns preprocessing and alignment to the data layer; workbook observations use 15-minute intervals. |
| FR-006 | Normalize and filter model inputs according to the training protocol. | The specification calls for z-score normalization per variable, target normalization by installed capacity, and daylight-only filtering. |
| FR-007 | Train forecasting models across available PV sites without temporal data leakage. | The training protocol calls for temporal cross-validation and cross-site training. |
| FR-008 | Produce multi-horizon solar power forecasts. | The intelligence layer is a Temporal Fusion Transformer forecasting engine that generates multi-horizon forecasts. |
| FR-009 | Support probabilistic forecast outputs. | The forecasting layer is described as probabilistic multi-horizon forecasting. |
| FR-010 | Evaluate forecasts with standard error and bias metrics. | The specification names metrics such as MAE, RMSE, and MBE. |
| FR-011 | Optimize battery energy storage operation from forecast outputs. | The optimization layer consumes forecast outputs and produces BESS charge-discharge schedules. |
| FR-012 | Implement MPC/receding-horizon optimization for battery scheduling. | The specification selects MPC because schedules can be updated as forecasts and system state change. |
| FR-013 | Balance configurable objectives: grid exchange, self-consumption, and battery degradation. | The optimization objectives list grid import/export minimization, local self-consumption maximization, and battery lifetime extension. |
| FR-014 | Enforce BESS operational constraints. | The MPC section references battery charge-discharge scheduling with constraints. |
| FR-015 | Exchange data and outputs through standard interfaces. | The architecture references standardized interfaces and HEDGE-IoT integration pathways. |
| FR-016 | Produce pilot KPI outputs for milestone validation. | The specification defines KPI tracking aligned to project objectives and milestones MS1-MS3. |

## Nonfunctional Requirements

| ID | Requirement | Rationale |
| --- | --- | --- |
| NFR-001 | The system must support cloud or edge deployment. | The architecture is described as deployable in cloud or edge computing environments. |
| NFR-002 | The system must be modular across data, intelligence, and optimization layers. | The specification defines a three-layer modular architecture. |
| NFR-003 | The system must be geographically transferable to additional pilot locations. | Transferability to Greece, Italy, Slovenia, and other HEDGE-IoT pilots is described as an architectural goal. |
| NFR-004 | The system must use location-agnostic inputs where practical. | Satellite and public NWP data are highlighted because they reduce location-specific dependencies. |
| NFR-005 | The system must preserve clear model-evaluation reproducibility. | Temporal cross-validation and no data leakage are explicit training constraints. |
| NFR-006 | The system must support missing-data contingency behavior. | The specification includes data availability and contingency considerations. |
| NFR-007 | The system must keep user-editable configuration for pilot-specific constraints and objective weights. | Optimization objective weights are pilot-operator configurable. |
| NFR-008 | The system must protect sensitive data and follow GDPR constraints where applicable. | GDPR appears in the system glossary and project context. |

## Interfaces

| Interface | Direction | Canonical Payload |
| --- | --- | --- |
| Pilot production data | Input | `ProductionObservation` records at 15-minute or source-native resolution. |
| Weather/NWP data | Input | `WeatherObservation` records keyed by site and timestamp. |
| Satellite features | Input | `SatelliteFeatureObservation` records keyed by site or geospatial tile and timestamp. |
| Site metadata | Input | `SiteMetadata` records with location, capacity, timezone, and optional grid/battery references. |
| Forecast output | Output | `ForecastRecord` records with horizon timestamps and optional quantiles. |
| BESS constraints | Input | `BessConstraints` records defining capacity, power limits, efficiency, and SoC bounds. |
| Optimization schedule | Output | `BessScheduleRecord` records with charge, discharge, SoC, grid import/export, and objective terms. |
| HEDGE-IoT exchange | Input/Output | REST/MQTT-compatible serialized records using canonical timestamps and units. |

## Milestones and KPIs

| Area | KPI / Check | Target Type |
| --- | --- | --- |
| Data readiness | Production, weather/NWP, satellite, and site metadata streams can be loaded and aligned. | MS1-style integration readiness. |
| Forecast accuracy | MAE and RMSE reported for holdout periods and pilot sites. | Quantitative forecast quality. |
| Forecast bias | MBE reported to detect systematic over- or under-forecasting. | Quantitative bias control. |
| Forecast horizon | Multi-horizon forecasts generated on the configured 15-minute grid. | Functional acceptance. |
| Optimization | BESS schedule generated from forecasts and constraints. | Functional acceptance. |
| Pilot value | Grid exchange reduction and self-consumption improvement calculated from schedule simulation. | Operational KPI. |
| Battery operation | Cycle/degradation proxy reported when scheduling. | Asset-lifetime KPI. |
| Transferability | New site can be configured through metadata and input mappings without model-code rewrite. | Scalability KPI. |

## Open Requirements

- Exact satellite feature source and raster/tile schema.
- Exact NWP provider, forecast horizon, and variable set.
- Finnish pilot battery capacity, power limits, SoC bounds, and efficiency.
- Final KPI numeric targets from project governance documents, if stricter than the system specification text available here.
- External workbooks referenced by `Hirvensalmi_tuotanto2025.xlsx` for source-level traceability.
