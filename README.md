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

## Slovenian-to-Finnish TFT transfer

The versioned transfer workflow uses 15-minute capacity factor, an eight-step
encoder/decoder, deterministic solar/calendar features, a global Slovenian TFT,
and full-layer Finnish fine-tuning. The primary acceptance metric is pooled
daylight point-head MSE skill against physical clear-sky smart persistence.
The q50 head is reported separately for median/MAE interpretation. Acceptance
is report-only: both the 10% point-estimate result and its 95% interval are shown.

```powershell
# Correct Finnish workbook parsing and canonical Parquet + manifest
aurora prepare-finnish Hirvensalmi_tuotanto2025.xlsx

# Restored source uses data/processed/data.p (34944 x 233 wide power-kW data).
# Supply its matching plant metadata with IDs, coordinates, and installed capacity.
aurora prepare-slovenian F:\nitrov15_backup\dimis\sol-forecast `
  --metadata F:\path\to\plants.csv

# Full configured CUDA experiment (isolated v2 model/output paths)
aurora run-transfer-experiment configs/transfer_experiment.v2.json

# Correct an existing v1 evaluation without retraining or changing v1 artifacts
aurora reevaluate-transfer outputs/transfer_experiment `
  configs/transfer_experiment.v2.json --output outputs/transfer_experiment_v2
```

Individual stages are also exposed as `pretrain-slovenian`,
`fine-tune-finnish`, and `fit-final-finnish`. `compare-tft-plan` remains a
legacy one-step compatibility command; its checkpoints cannot be loaded by the
`tft-transfer-v1` schema.

`transfer-experiment-v1` remains readable for exact legacy reproduction and
keeps harmonic persistence canonical. New work should use
`transfer-experiment-v2`, which retains that forecast under
`legacy_harmonic_smart_persistence` for audit.

The default experiment requires CUDA. Set `"accelerator": "cpu"` with a small
epoch/sample budget only for smoke testing.

## First Milestones

1. Extract canonical data schema from the Hirvensalmi workbook.
2. Build a reliable 15-minute production loader.
3. Add baseline forecasts before TFT work.
4. Define BESS constraints and objective weights for the MPC layer.
5. Add end-to-end evaluation reports.
