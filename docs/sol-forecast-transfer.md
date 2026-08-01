# sol-forecast Transfer Notes

Reviewed source: `F:\nitrov15_backup\dimis\sol-forecast`.

Transferred reusable parts:

- Solar position and clear-sky irradiance helpers into `src/aurora/features/solar.py`.
- Harmonic daily/yearly time features and step inference into `src/aurora/features/time.py`.
- Persistence, climatology, and smart-persistence baselines into `src/aurora/forecasting/baselines.py`.
- Rolling month-based cross-validation folds into `src/aurora/evaluation/cross_validation.py`.
- Extra evaluation metrics into `src/aurora/evaluation/metrics.py`.

Transfer pipeline now implemented:

- Canonical UTC-grid Parquet datasets and hash manifests for Slovenia and Hirvensalmi.
- Correct Finnish Q1/DST/malformed-row handling with masked gaps.
- Global eight-horizon TFT pretraining, unseen-site full-layer transfer, scratch control,
  fold isolation, long-form predictions, final checkpoint packaging, and CLI stages.
- Physical clear-sky smart persistence from issue-time production and canonical
  `clear_sky_norm`, with the fold-trained q=0.99 harmonic result retained only for audit.
- Separate TFT point-head and q50 reporting, common-mask daylight evaluation,
  baseline zero-forecast diagnostics, moving-block confidence intervals, and
  report-only 10% acceptance statuses.
- A v2 re-evaluation path that reuses saved model predictions and leaves all v1
  outputs and checkpoints untouched.

The restored `data/processed/data.p` was validated as a 34,944-row by 233-site wide
power table spanning 2022-02-02 through 2023-01-31. Its matching plant metadata is not
present under the restored project root, so canonical Slovenian preparation requires the
metadata path explicitly.

Intentionally not used:

- Old TFT checkpoints and one-step scripts, because they are schema-incompatible legacy artifacts.
- Satellite extraction module, because it assumes fixed latitude/longitude image bounds and raw numpy/zarr image layouts not yet defined for the Finnish pilot.
- Paper reproduction scripts and notebooks, because they are experiment-specific.
- Environment files and shell scripts, because this repo already has a Python package scaffold and should converge on one dependency path.
