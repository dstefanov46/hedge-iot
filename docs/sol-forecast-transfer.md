# sol-forecast Transfer Notes

Reviewed source: `F:\nitrov15_backup\dimis\sol-forecast`.

Transferred reusable parts:

- Solar position and clear-sky irradiance helpers into `src/aurora/features/solar.py`.
- Harmonic daily/yearly time features and step inference into `src/aurora/features/time.py`.
- Persistence, climatology, and smart-persistence baselines into `src/aurora/forecasting/baselines.py`.
- Rolling month-based cross-validation folds into `src/aurora/evaluation/cross_validation.py`.
- Extra evaluation metrics into `src/aurora/evaluation/metrics.py`.

Intentionally not transferred yet:

- Old TFT training scripts, because they rely on the previous project layout and should be adapted after the AURORA canonical dataset is finalized.
- Satellite extraction module, because it assumes fixed latitude/longitude image bounds and raw numpy/zarr image layouts not yet defined for the Finnish pilot.
- Paper reproduction scripts and notebooks, because they are experiment-specific.
- Environment files and shell scripts, because this repo already has a Python package scaffold and should converge on one dependency path.
