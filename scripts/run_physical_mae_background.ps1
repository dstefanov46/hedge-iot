$ErrorActionPreference = "Continue"
Set-Location "C:\Users\dimit\hedge-iot\code"

& .\.venv\Scripts\python.exe -c 'import torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.cuda.is_available()))'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& .\.venv\Scripts\aurora.exe pretrain-slovenian `
  outputs\transfer_experiment_satellite_pilot\prepared_satellite\slovenia.parquet `
  --output models\tft_slovenia_satellite_physical_mae_retry `
  --config configs\tft_satellite_physical_mae.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$source = Get-ChildItem models\tft_slovenia_satellite_physical_mae_retry -File -Filter *.ckpt |
  Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $source) { Write-Error "No Slovenian checkpoint was produced."; exit 1 }

& .\.venv\Scripts\aurora.exe fine-tune-finnish `
  outputs\transfer_experiment_satellite_pilot\prepared_satellite\finnish.parquet `
  --pretrained-checkpoint $source.FullName `
  --output outputs\transfer_experiment_satellite_physical_mae_retry `
  --model-root models\tft_finnish_folds_satellite_physical_mae_retry `
  --config configs\tft_satellite_physical_mae.json `
  --no-enforce-gate
exit $LASTEXITCODE
