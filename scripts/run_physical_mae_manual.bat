@echo off
setlocal EnableExtensions

cd /d C:\Users\dimit\hedge-iot\code
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate the virtual environment.
    exit /b 1
)

echo Checking Torch and CUDA...
python -c "import torch; print('Torch ' + torch.__version__); print('CUDA available: ' + str(torch.cuda.is_available()))"
if errorlevel 1 (
    echo Torch cannot be imported. Resolve the Windows DLL policy error first.
    exit /b 1
)

echo Starting Slovenian pretraining...
aurora pretrain-slovenian ^
  outputs\transfer_experiment_satellite_pilot\prepared_satellite\slovenia.parquet ^
  --output models\tft_slovenia_satellite_physical_mae_retry ^
  --config configs\tft_satellite_physical_mae.json
if errorlevel 1 (
    echo Slovenian pretraining failed.
    exit /b 1
)

set "SOURCE="
for /f "delims=" %%F in ('dir /b /a-d /o-d "models\tft_slovenia_satellite_physical_mae_retry\*.ckpt" 2^>nul') do if not defined SOURCE set "SOURCE=models\tft_slovenia_satellite_physical_mae_retry\%%F"
if not defined SOURCE (
    echo No Slovenian checkpoint was produced.
    exit /b 1
)
echo Using checkpoint: %SOURCE%

echo Starting Finnish retraining and evaluation...
aurora fine-tune-finnish ^
  outputs\transfer_experiment_satellite_pilot\prepared_satellite\finnish.parquet ^
  --pretrained-checkpoint "%SOURCE%" ^
  --output outputs\transfer_experiment_satellite_physical_mae_retry ^
  --model-root models\tft_finnish_folds_satellite_physical_mae_retry ^
  --config configs\tft_satellite_physical_mae.json ^
  --no-enforce-gate
if errorlevel 1 (
    echo Finnish retraining/evaluation failed.
    exit /b 1
)

echo.
echo Experiment complete.
echo Metrics: outputs\transfer_experiment_satellite_physical_mae_retry\metrics.csv
echo Pooled results: outputs\transfer_experiment_satellite_physical_mae_retry\pooled_metrics.json
endlocal
