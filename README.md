# AURORA HEDGE-IoT

AURORA is a solar production forecasting and battery optimization project for the HEDGE-IoT pilot context.

This scaffold is organized around the system specification in `D1.1_AURORA_System_Specification_v4.docx` and the Hirvensalmi production workbook `Hirvensalmi_tuotanto2025.xlsx`.

## Scope

- Ingest pilot production data, weather/NWP data, satellite-derived features, and site metadata.
- Align multimodal inputs into model-ready time series.
- Train and evaluate solar production forecasts.
- Produce battery energy storage schedules with a receding-horizon MPC layer.
- Expose repeatable CLI workflows for data preparation, training, evaluation, and optimization.

## Layout

```text
src/aurora/
  cli.py                 Command-line entry point
  config.py              Project settings
  data/                  Workbook and source-data ingestion
  features/              Time-series alignment and feature engineering
  forecasting/           Forecast model interfaces and baselines
  optimization/          Battery/MPC scheduling interfaces
  evaluation/            Forecast and schedule metrics
tests/                   Unit tests
data/raw/                Local raw inputs, gitignored
data/processed/          Generated datasets, gitignored
models/                  Trained model artifacts, gitignored
outputs/                 Reports, forecasts, schedules, gitignored
docs/                    Project notes and extracted requirements
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m aurora --help
pytest
```

## First Milestones

1. Extract canonical data schema from the Hirvensalmi workbook.
2. Build a reliable 15-minute production loader.
3. Add baseline forecasts before TFT work.
4. Define BESS constraints and objective weights for the MPC layer.
5. Add end-to-end evaluation reports.
