from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from math import floor
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from aurora.data.canonical import read_processed_dataset, sha256_file
from aurora.data.grib import read_grib_weather_archive
from aurora.data.openmeteo import (
    OPEN_METEO_MODEL,
    OpenMeteoClient,
    attach_weather,
    fetch_single_run_weather_archive,
)
from aurora.evaluation.cross_validation import CVFold, generate_expanding_finnish_folds
from aurora.evaluation.transfer import (
    BASELINE_IDENTITY_LEGACY,
    BASELINE_IDENTITY_PHYSICAL,
    evaluate_long_forecasts,
    metric_summary,
)
from aurora.forecasting.baselines import HarmonicSmartPersistence, PhysicalSmartPersistence
from aurora.forecasting.tft_transfer import (
    TFTTransferConfig,
    assert_usable_weather,
    create_dataset,
    fit_tft,
    predict_long,
    save_model_bundle,
)

LOSS_SWEEP_PROFILES = ("mse-heavy", "balanced-huber-mse", "mae-heavy")


@dataclass(frozen=True)
class TransferExperimentConfig:
    schema_version: str
    slovenian_dataset: Path
    finnish_dataset: Path
    output_root: Path
    slovenian_checkpoint_root: Path
    fold_model_root: Path
    final_model_root: Path
    source_hashes: dict[str, str]
    tft: TFTTransferConfig
    bootstrap_replicates: int = 100
    bootstrap_block_length: int = 15
    skill_threshold: float = 0.10
    satellite: dict[str, object] | None = None
    weather: dict[str, object] | None = None

    @classmethod
    def from_json(cls, path: Path) -> TransferExperimentConfig:
        values = json.loads(path.read_text(encoding="utf-8"))
        supported = {"transfer-experiment-v1", "transfer-experiment-v2"}
        if values.get("schema_version") not in supported:
            raise ValueError(
                "Expected transfer experiment schema_version 'transfer-experiment-v1' "
                "or 'transfer-experiment-v2'."
            )
        paths = values["paths"]
        return cls(
            schema_version=values["schema_version"],
            slovenian_dataset=Path(paths["slovenian_dataset"]),
            finnish_dataset=Path(paths["finnish_dataset"]),
            output_root=Path(paths["output_root"]),
            slovenian_checkpoint_root=Path(paths["slovenian_checkpoint_root"]),
            fold_model_root=Path(paths["fold_model_root"]),
            final_model_root=Path(paths["final_model_root"]),
            source_hashes=values.get("source_hashes", {}),
            tft=TFTTransferConfig(
                **{
                    **values.get("tft", {}),
                    **_satellite_tft_values(values.get("satellite")),
                    **_weather_tft_values(values.get("weather")),
                }
            ),
            bootstrap_replicates=int(values.get("bootstrap_replicates", 100)),
            bootstrap_block_length=int(values.get("bootstrap_block_length", 15)),
            skill_threshold=float(values.get("skill_threshold", 0.10)),
            satellite=values.get("satellite"),
            weather=values.get("weather"),
        )


def _satellite_tft_values(values: dict[str, object] | None) -> dict[str, object]:
    if not values or not bool(values.get("enabled", False)):
        return {}
    return {
        "satellite_enabled": True,
        "satellite_encoder_only": bool(values.get("encoder_only", True)),
        "satellite_embedding_dim": int(values.get("embedding_dim", 32)),
        "satellite_missing_value": float(values.get("missing_value", 0.0)),
        "satellite_max_alignment_minutes": float(values.get("max_alignment_minutes", 7.5)),
    }


def _weather_tft_values(values: dict[str, object] | None) -> dict[str, object]:
    if not values or not bool(values.get("enabled", False)):
        return {}
    return {"weather_enabled": True}


def pretrain_slovenian_tft(
    dataset_path: Path,
    checkpoint_root: Path,
    config: TFTTransferConfig,
    source_hashes: dict[str, str] | None = None,
    validation_days: int = 30,
    resume_checkpoint: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    frame = read_processed_dataset(dataset_path)
    if config.weather_enabled:
        assert_usable_weather(frame)
    source_hashes = source_hashes or _manifest_source_hashes(dataset_path)
    validation_start = frame["timestamp_utc"].max() - pd.Timedelta(days=validation_days)
    train_frame = frame[frame["timestamp_utc"] < validation_start]
    validation_idx = int(frame.loc[frame["timestamp_utc"] >= validation_start, "time_idx"].min())
    training = create_dataset(train_frame, config)
    validation = create_dataset(
        frame,
        config,
        template=training,
        min_prediction_idx=validation_idx,
    )
    model, metadata = fit_tft(
        training,
        validation,
        config,
        checkpoint_root,
        resume_checkpoint=resume_checkpoint,
    )
    checkpoint_path = Path(metadata["best_model_path"])
    save_model_bundle(
        checkpoint_root / "bundle",
        model,
        training,
        config,
        metadata,
        source_hashes,
        _capacity_metadata(frame),
        dataset_path=dataset_path,
    )
    return checkpoint_path, metadata


def run_finnish_rolling_evaluation(
    dataset_path: Path,
    pretrained_checkpoint: Path,
    output_root: Path,
    model_root: Path,
    config: TFTTransferConfig,
    bootstrap_replicates: int = 100,
    bootstrap_block_length: int = 15,
    skill_threshold: float = 0.10,
    enforce_gate: bool = True,
    folds: list[CVFold] | None = None,
    experiment_schema: str = "transfer-experiment-v2",
    resume: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[int]]:
    frame = read_processed_dataset(dataset_path)
    folds = folds or generate_expanding_finnish_folds(
        int(frame["timestamp_utc"].dt.year.mode().iloc[0]), timezone="Europe/Helsinki"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "run_state.json"
    state = _load_or_create_run_state(
        state_path,
        resume=resume,
        dataset_path=dataset_path,
        pretrained_checkpoint=pretrained_checkpoint,
        config=config,
        experiment_schema=experiment_schema,
    )
    all_predictions: list[pd.DataFrame] = []
    best_epochs: list[int] = [int(epoch) for epoch in state.get("best_epochs", [])]
    for fold in folds:
        prediction_path = output_root / f"predictions_fold_{fold.fold_id}.csv"
        if resume and fold.fold_id in state["completed_folds"] and prediction_path.is_file():
            fold_predictions = pd.read_csv(
                prediction_path,
                parse_dates=["issue_timestamp_utc", "target_timestamp_utc"],
            )
            all_predictions.append(fold_predictions)
            continue
        fold_predictions, transfer_metadata = _run_fold(
            frame=frame,
            fold=fold,
            pretrained_checkpoint=pretrained_checkpoint,
            model_root=model_root / f"fold_{fold.fold_id}",
            config=config,
            experiment_schema=experiment_schema,
            resume=resume,
        )
        best_epoch = transfer_metadata.get("best_epoch")
        if best_epoch:
            best_epochs.append(int(best_epoch))
        all_predictions.append(fold_predictions)
        fold_predictions.to_csv(prediction_path, index=False)
        state["completed_folds"] = sorted(set(state["completed_folds"]) | {int(fold.fold_id)})
        state["best_epochs"] = best_epochs
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics, pooled = evaluate_long_forecasts(
        predictions,
        n_bootstrap=bootstrap_replicates,
        block_length=bootstrap_block_length,
        seed=config.random_seed,
        skill_threshold=skill_threshold,
        baseline_identity=_baseline_identity(experiment_schema),
    )
    predictions.to_csv(output_root / "predictions_long.csv", index=False)
    metrics.to_csv(output_root / "metrics.csv", index=False)
    (output_root / "pooled_metrics.json").write_text(
        json.dumps(pooled, indent=2, sort_keys=True), encoding="utf-8"
    )
    return predictions, metrics, pooled, best_epochs


def report_finnish_weather_ablation(
    baseline_output: Path,
    weather_output: Path,
    output: Path,
) -> dict[str, Any]:
    """Compare existing satellite, valid-weather transfer, and weather scratch.

    Only prediction rows carrying both issue-time and weather availability are
    included.  The report is therefore not allowed to silently fall back to
    the masked historical-weather rows used by the earlier experiment.
    """
    baseline_path = baseline_output / "predictions_long.csv"
    weather_path = weather_output / "predictions_long.csv"
    if not baseline_path.is_file() or not weather_path.is_file():
        raise FileNotFoundError("Both baseline and valid-weather predictions_long.csv are required")
    baseline = pd.read_csv(
        baseline_path,
        parse_dates=["issue_timestamp_utc", "target_timestamp_utc"],
    )
    weather = pd.read_csv(
        weather_path,
        parse_dates=["issue_timestamp_utc", "target_timestamp_utc"],
    )
    required_weather = {"weather_issue_timestamp_available", "weather_available"}
    missing = required_weather - set(weather)
    if missing:
        raise AssertionError(f"Weather predictions lack availability columns: {sorted(missing)}")
    valid = (pd.to_numeric(weather["weather_issue_timestamp_available"], errors="coerce") > 0) & (
        pd.to_numeric(weather["weather_available"], errors="coerce") > 0
    )
    assert valid.mean() > 0, "No valid issue-time-aware weather prediction rows"
    keys = ["site_id", "issue_timestamp_utc", "target_timestamp_utc", "horizon_step", "fold"]
    weather_columns = keys + [
        "actual_value",
        "solar_elevation",
        "smart_persistence",
        "point_forecast",
        "q50",
        "scratch_forecast",
        "weather_issue_timestamp_available",
        "weather_available",
        "weather_lead_hours",
    ]
    weather = weather[weather_columns].copy()
    baseline = baseline[keys + ["point_forecast"]].rename(
        columns={"point_forecast": "satellite_baseline_forecast"}
    )
    merged = weather.merge(baseline, on=keys, how="inner", validate="one_to_one")
    merged = merged.loc[
        (merged["weather_issue_timestamp_available"] > 0) & (merged["weather_available"] > 0)
    ].copy()
    if merged.empty:
        raise AssertionError("No baseline/weather rows remain after metadata filtering")
    model_columns = {
        "satellite_baseline": "satellite_baseline_forecast",
        "weather_transfer": "point_forecast",
        "weather_transfer_q50": "q50",
        "weather_scratch": "scratch_forecast",
        "smart_persistence": "smart_persistence",
    }
    metric_rows: list[dict[str, Any]] = []
    for model, column in model_columns.items():
        for grouping, group_keys in (
            ("overall", []),
            ("horizon", ["horizon_step"]),
            ("fold", ["fold"]),
        ):
            groups = [((), merged)] if not group_keys else merged.groupby(group_keys, sort=True)
            for key, group in groups:
                values = group[["actual_value", "smart_persistence", column]].rename(
                    columns={column: "point_forecast"}
                )
                row = {
                    "model": model,
                    "grouping": grouping,
                    **metric_summary(values, "point_forecast"),
                }
                key_values = key if isinstance(key, tuple) else (key,)
                row.update(dict(zip(group_keys, key_values, strict=True)))
                metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    availability = {
        "rows_total": int(len(weather)),
        "rows_valid": int(len(merged)),
        "issue_timestamp_available_pct": float(
            100 * weather["weather_issue_timestamp_available"].mean()
        ),
        "weather_available_pct": float(100 * weather["weather_available"].mean()),
        "valid_rows_by_fold": {
            str(fold): int(len(group)) for fold, group in merged.groupby("fold", sort=True)
        },
    }
    pooled = {
        model: metrics.query("model == @model and grouping == 'overall'").iloc[0].to_dict()
        for model in model_columns
    }
    output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / "predictions_valid_weather.csv", index=False)
    metrics.to_csv(output / "ablation_metrics.csv", index=False)
    summary = {"availability": availability, "pooled": pooled}
    (output / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def run_tft_loss_sweep(
    dataset_path: Path,
    pretrained_checkpoint: Path,
    current_output: Path,
    output_root: Path,
    model_root: Path,
    config: TFTTransferConfig,
    bootstrap_replicates: int = 100,
    bootstrap_block_length: int = 15,
    skill_threshold: float = 0.10,
    profiles: tuple[str, ...] = LOSS_SWEEP_PROFILES,
) -> pd.DataFrame:
    """Train and evaluate the three point-loss profiles against the current run.

    The current run is read from its saved metrics.csv; candidate runs reuse the
    same source checkpoint, folds, daylight mask, and evaluation contract.
    """
    current_metrics_path = current_output / "metrics.csv"
    if not current_metrics_path.is_file():
        raise FileNotFoundError(f"Current metrics are missing: {current_metrics_path}")
    current = pd.read_csv(current_metrics_path)
    current_row = current[(current["model"] == "transfer") & (current["grouping"] == "overall")]
    if len(current_row) != 1:
        raise ValueError("Current metrics.csv must contain exactly one transfer/overall row.")

    rows: list[dict[str, Any]] = [
        {
            "profile": "current-checkpoint",
            "point_loss": "mse",
            "source": str(current_output),
            **current_row.iloc[0].to_dict(),
        }
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    unknown_profiles = sorted(set(profiles).difference(LOSS_SWEEP_PROFILES + ("pure-mae",)))
    if unknown_profiles:
        raise ValueError(f"Unknown loss sweep profile(s): {', '.join(unknown_profiles)}")
    for profile in profiles:
        candidate_config = replace(config, point_loss=profile)
        candidate_output = output_root / profile
        _, metrics, _, _ = run_finnish_rolling_evaluation(
            dataset_path,
            pretrained_checkpoint,
            candidate_output,
            model_root / profile,
            candidate_config,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_block_length=bootstrap_block_length,
            skill_threshold=skill_threshold,
            experiment_schema="transfer-experiment-v2",
        )
        candidate_row = metrics[
            (metrics["model"] == "transfer") & (metrics["grouping"] == "overall")
        ]
        if len(candidate_row) != 1:
            raise ValueError(f"Candidate {profile} did not produce one transfer/overall row.")
        rows.append(
            {
                "profile": profile,
                "point_loss": profile,
                "source": str(candidate_output),
                **candidate_row.iloc[0].to_dict(),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison["mae_delta_vs_current"] = comparison["mae"] - float(rows[0]["mae"])
    comparison["mse_delta_vs_current"] = comparison["mse"] - float(rows[0]["mse"])
    comparison.to_csv(output_root / "loss_sweep_comparison.csv", index=False)
    (output_root / "loss_sweep_comparison.json").write_text(
        json.dumps(comparison.to_dict(orient="records"), indent=2, default=str), encoding="utf-8"
    )
    return comparison


def _run_fold(
    frame: pd.DataFrame,
    fold: CVFold,
    pretrained_checkpoint: Path,
    model_root: Path,
    config: TFTTransferConfig,
    experiment_schema: str = "transfer-experiment-v2",
    resume: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    timestamps = frame["timestamp_utc"]
    train_frame = frame[(timestamps >= fold.train_start) & (timestamps <= fold.train_end)]
    through_validation = frame[(timestamps >= fold.train_start) & (timestamps <= fold.val_end)]
    through_test = frame[(timestamps >= fold.train_start) & (timestamps <= fold.test_end)]
    val_idx = int(through_validation.loc[timestamps >= fold.val_start, "time_idx"].min())
    test_idx = int(through_test.loc[timestamps >= fold.test_start, "time_idx"].min())
    training = create_dataset(train_frame, config)
    validation = create_dataset(
        through_validation,
        config,
        template=training,
        min_prediction_idx=val_idx,
    )
    test = create_dataset(
        through_test,
        config,
        template=training,
        min_prediction_idx=test_idx,
    )
    transfer_model, transfer_metadata = fit_tft(
        training,
        validation,
        replace(config, learning_rate=1e-4, max_epochs=min(config.max_epochs, 50)),
        model_root / "transfer",
        pretrained_checkpoint=pretrained_checkpoint,
        resume_checkpoint=_latest_checkpoint(model_root / "transfer") if resume else None,
    )
    scratch_model, scratch_metadata = fit_tft(
        training,
        validation,
        config,
        model_root / "scratch",
        resume_checkpoint=_latest_checkpoint(model_root / "scratch") if resume else None,
    )
    transfer = predict_long(transfer_model, test, through_test, config.batch_size)
    scratch = predict_long(scratch_model, test, through_test, config.batch_size)
    keys = ["site_id", "issue_timestamp_utc", "target_timestamp_utc", "horizon_step"]
    scratch = scratch[keys + ["point_forecast"]].rename(
        columns={"point_forecast": "scratch_forecast"}
    )
    predictions = transfer.merge(scratch, on=keys, how="left", validate="one_to_one")
    predictions = predictions[
        (predictions["target_timestamp_utc"] >= fold.test_start)
        & (predictions["target_timestamp_utc"] <= fold.test_end)
    ].copy()

    issues = pd.DatetimeIndex(predictions["issue_timestamp_utc"].drop_duplicates())
    predictions = _attach_fold_baselines(
        predictions,
        train_frame,
        through_test,
        issues,
        config.horizon_steps,
        experiment_schema,
    )
    target_lookup = through_test.drop_duplicates("timestamp_utc").set_index("timestamp_utc")
    predictions["existing_plan"] = predictions["target_timestamp_utc"].map(
        target_lookup["planned_mwh"] / (target_lookup["installed_capacity_mw"] * 0.25)
    )
    predictions["split"] = "test"
    predictions["fold"] = fold.fold_id
    predictions.attrs["scratch_training"] = scratch_metadata
    return predictions, transfer_metadata


def fit_final_finnish_checkpoint(
    dataset_path: Path,
    pretrained_checkpoint: Path,
    output_root: Path,
    config: TFTTransferConfig,
    fold_best_epochs: list[int],
    source_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not fold_best_epochs:
        raise ValueError("At least one fold best epoch is required for final fitting.")
    epochs = max(1, floor(median(fold_best_epochs) + 0.5))
    frame = read_processed_dataset(dataset_path)
    source_hashes = source_hashes or _manifest_source_hashes(dataset_path)
    training = create_dataset(frame, config)
    # Final epoch count is fixed from isolated fold validation. Reusing the loader
    # as a monitor does not select or tune the deliverable checkpoint.
    model, metadata = fit_tft(
        training,
        training,
        config,
        output_root / "checkpoints",
        pretrained_checkpoint=pretrained_checkpoint,
        fixed_epochs=epochs,
    )
    metadata["fixed_epoch_count"] = epochs
    metadata["fold_best_epochs"] = fold_best_epochs
    save_model_bundle(
        output_root,
        model,
        training,
        config,
        metadata,
        source_hashes,
        _capacity_metadata(frame),
    )
    return metadata


def run_transfer_experiment(config_path: Path) -> dict[str, Any]:
    experiment = TransferExperimentConfig.from_json(config_path)
    if experiment.weather and bool(experiment.weather.get("enabled", False)):
        experiment = _prepare_weather_frames(experiment)
    if experiment.satellite and bool(experiment.satellite.get("enabled", False)):
        experiment = _prepare_satellite_frames(experiment)
    reusable = _reusable_pretraining_checkpoint(
        experiment.slovenian_dataset,
        experiment.slovenian_checkpoint_root,
        experiment.tft,
        experiment.source_hashes,
    )
    if reusable is not None:
        checkpoint, pretraining = reusable
        print(f"[TFT] reusing Slovenian checkpoint {checkpoint}", flush=True)
    else:
        checkpoint, pretraining = pretrain_slovenian_tft(
            experiment.slovenian_dataset,
            experiment.slovenian_checkpoint_root,
            experiment.tft,
            experiment.source_hashes,
        )
    _, _, pooled, epochs = run_finnish_rolling_evaluation(
        experiment.finnish_dataset,
        checkpoint,
        experiment.output_root,
        experiment.fold_model_root,
        experiment.tft,
        experiment.bootstrap_replicates,
        experiment.bootstrap_block_length,
        experiment.skill_threshold,
        experiment_schema=experiment.schema_version,
    )
    final = fit_final_finnish_checkpoint(
        experiment.finnish_dataset,
        checkpoint,
        experiment.final_model_root,
        experiment.tft,
        epochs,
        experiment.source_hashes,
    )
    summary = {
        "schema_version": experiment.schema_version,
        "legacy": experiment.schema_version == "transfer-experiment-v1",
        "config": asdict(experiment.tft),
        "pretraining": pretraining,
        "pooled_metrics": pooled,
        "final_training": final,
    }
    (experiment.output_root / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _prepare_weather_frames(experiment: TransferExperimentConfig) -> TransferExperimentConfig:
    """Download/cache weather once and materialize issue-aware columns beside inputs."""
    values = experiment.weather or {}
    provider = str(values.get("provider", "open-meteo")).lower()
    cache_dir = Path(str(values.get("cache_dir", "data/cache/openmeteo")))
    mode = str(values.get("mode", "historical"))
    model = str(values.get("model", OPEN_METEO_MODEL))
    client = OpenMeteoClient(cache_dir)
    prepared_root = experiment.output_root / "prepared_weather"
    prepared_root.mkdir(parents=True, exist_ok=True)
    new_paths: list[Path] = []
    weather_manifest: dict[str, Any] = {
        "provider": "ECMWF GRIB" if provider in {"ecmwf-grib", "grib", "nwp"} else "Open-Meteo",
        "api_mode": mode,
        "model": model,
        "license": "CC BY 4.0",
        "cache_dir": str(cache_dir),
        "sites": {},
    }
    if provider in {"ecmwf-grib", "grib", "nwp"}:
        slovenia_frame = read_processed_dataset(experiment.slovenian_dataset)
        site_rows = slovenia_frame.groupby("site_id", as_index=False).first()[
            ["site_id", "latitude", "longitude"]
        ]
        grib_root = Path(str(values.get("archive_root", r"F:\MAG\nwp_data")))
        grib_weather, grib_manifest = read_grib_weather_archive(
            grib_root,
            site_rows,
            slovenia_frame["timestamp_utc"],
            checkpoint_root=prepared_root / "grib_checkpoints",
        )
        weather_manifest.update(grib_manifest)
        weather_manifest["sites"] = {
            str(site.site_id): {
                "latitude": float(site.latitude),
                "longitude": float(site.longitude),
            }
            for site in site_rows.itertuples(index=False)
        }
        slovenia_output = prepared_root / "slovenia.parquet"
        attach_weather(slovenia_frame, grib_weather, already_aligned=True).to_parquet(
            slovenia_output, index=False
        )
        new_paths = [slovenia_output]
        # Finland deliberately remains exact-run ECMWF single-run Open-Meteo.
        finland_label = "finland"
    else:
        finland_label = "finland"
    for label, input_path in (
        ("slovenia", experiment.slovenian_dataset),
        ("finland", experiment.finnish_dataset),
    ):
        if provider in {"ecmwf-grib", "grib", "nwp"} and label == "slovenia":
            continue
        frame = read_processed_dataset(input_path)
        weather_parts = []
        for site_id, group in frame.groupby("site_id", sort=False):
            latitude = float(group["latitude"].iloc[0])
            longitude = float(group["longitude"].iloc[0])
            if (
                mode == "single_run" or provider in {"ecmwf-grib", "grib", "nwp"}
            ) and label == finland_label:
                parsed = fetch_single_run_weather_archive(
                    client,
                    site_id=str(site_id),
                    latitude=latitude,
                    longitude=longitude,
                    target_timestamps=group["timestamp_utc"],
                    model=model,
                    forecast_days=int(values.get("forecast_days", 10)),
                )
                weather_manifest["sites"][str(site_id)] = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": group["timestamp_utc"].min().date().isoformat(),
                    "end_date": group["timestamp_utc"].max().date().isoformat(),
                    "api_mode": "single_run",
                    "run_cycles_utc": ["00:00", "06:00", "12:00", "18:00"],
                    "issue_timestamp_available_rows": int(
                        parsed["weather_issue_timestamp_available"].sum()
                    ),
                    "weather_available_rows": int(parsed["weather_available"].sum()),
                    "row_count": int(len(parsed)),
                    "run_count": int(parsed.attrs.get("run_count", 0)),
                    "cache_hashes": parsed.attrs.get("cache_hashes", []),
                }
            else:
                # The 2022 Slovenian period predates the exact-run archive. Keep
                # this source weather explicitly masked; never infer an issue.
                parsed = pd.DataFrame(
                    {"site_id": str(site_id), "timestamp_utc": group["timestamp_utc"]}
                )
                for column in (
                    "weather_temperature_2m",
                    "weather_relative_humidity_2m",
                    "weather_dew_point_2m",
                    "weather_wind_speed_10m",
                    "weather_wind_direction_10m",
                    "weather_cloud_cover",
                    "weather_low_cloud_cover",
                    "weather_precipitation",
                    "weather_shortwave_radiation",
                    "weather_surface_pressure",
                    "weather_available",
                    "weather_issue_timestamp_available",
                    "weather_lead_hours",
                ):
                    parsed[column] = 0.0
                parsed["weather_provider"] = "open-meteo"
                parsed["weather_model"] = model
                parsed["weather_forecast_reference_utc"] = pd.NaT
                weather_manifest["sites"][str(site_id)] = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "api_mode": "masked_no_exact_runs",
                    "issue_timestamp_available_rows": 0,
                    "weather_available_rows": 0,
                    "row_count": int(len(parsed)),
                }
            weather_parts.append(parsed)
        weather = pd.concat(weather_parts, ignore_index=True) if weather_parts else pd.DataFrame()
        output = prepared_root / f"{label}.parquet"
        attached = attach_weather(frame, weather, already_aligned=True)
        attached.to_parquet(output, index=False)
        new_paths.append(output)
    manifest_path = prepared_root / "weather_manifest.json"
    manifest_path.write_text(
        json.dumps(weather_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return replace(
        experiment,
        slovenian_dataset=new_paths[0],
        finnish_dataset=new_paths[1],
        weather=weather_manifest,
    )


def _prepare_satellite_frames(experiment: TransferExperimentConfig) -> TransferExperimentConfig:
    """Materialize the shared satellite contract once before all three TFT stages."""
    from aurora.satellite.transfer import materialize_satellite_transfer

    values = experiment.satellite or {}
    required = ("slovenian_zarr_root", "finnish_patch_root")
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError(
            "Satellite transfer is enabled but the configuration is missing: " + ", ".join(missing)
        )
    checkpoint = Path(
        values.get(
            "encoder_checkpoint", experiment.slovenian_checkpoint_root / "satellite_encoder.pt"
        )
    )
    prepared_root = experiment.output_root / "prepared_satellite"
    source_path, target_path, _, source_hashes = materialize_satellite_transfer(
        experiment.slovenian_dataset,
        experiment.finnish_dataset,
        slovenian_zarr_root=values["slovenian_zarr_root"],
        finnish_patch_root=values["finnish_patch_root"],
        checkpoint_path=checkpoint,
        output_root=prepared_root,
        training_start=values.get("training_start"),
        training_end=values.get("training_end"),
        max_training_patches=int(values.get("max_training_patches", 4096)),
        max_alignment_minutes=float(values.get("max_alignment_minutes", 15.0)),
    )
    return replace(
        experiment,
        slovenian_dataset=source_path,
        finnish_dataset=target_path,
        source_hashes={**experiment.source_hashes, **source_hashes},
    )


def reevaluate_transfer_predictions(
    source_output: Path,
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Re-evaluate saved TFT predictions under the v2 physical-baseline contract."""
    experiment = TransferExperimentConfig.from_json(config_path)
    if experiment.schema_version != "transfer-experiment-v2":
        raise ValueError("Re-evaluation requires a transfer-experiment-v2 configuration.")
    source_predictions = (
        source_output / "predictions_long.csv" if source_output.is_dir() else source_output
    )
    if not source_predictions.is_file():
        raise FileNotFoundError(f"Missing source predictions: {source_predictions}")
    source_directory = source_predictions.parent.resolve()
    destination = output_root.resolve()
    if destination == source_directory:
        raise ValueError("Re-evaluation output must be distinct from the source output directory.")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"Re-evaluation output directory is not empty: {output_root}")

    predictions = pd.read_csv(
        source_predictions,
        parse_dates=["issue_timestamp_utc", "target_timestamp_utc"],
    )
    if "smart_persistence" not in predictions:
        raise ValueError("Source predictions do not contain the v1 smart_persistence baseline.")
    predictions["legacy_harmonic_smart_persistence"] = predictions["smart_persistence"]
    predictions = predictions.drop(columns=["smart_persistence"])

    finnish = read_processed_dataset(experiment.finnish_dataset)
    issues = pd.DatetimeIndex(predictions["issue_timestamp_utc"].drop_duplicates())
    physical = PhysicalSmartPersistence().predict_long(
        finnish, issues, experiment.tft.horizon_steps
    )
    predictions = predictions.merge(
        physical,
        on=["issue_timestamp_utc", "target_timestamp_utc", "horizon_step"],
        how="left",
        validate="many_to_one",
    )
    metrics, pooled = evaluate_long_forecasts(
        predictions,
        n_bootstrap=experiment.bootstrap_replicates,
        block_length=experiment.bootstrap_block_length,
        seed=experiment.tft.random_seed,
        skill_threshold=experiment.skill_threshold,
        baseline_identity=BASELINE_IDENTITY_PHYSICAL.copy(),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_root / "predictions_long.csv", index=False)
    if "fold" in predictions:
        for fold, fold_predictions in predictions.groupby("fold", sort=True):
            fold_label = str(int(fold)) if float(fold).is_integer() else str(fold)
            fold_predictions.to_csv(output_root / f"predictions_fold_{fold_label}.csv", index=False)
    metrics.to_csv(output_root / "metrics.csv", index=False)
    (output_root / "pooled_metrics.json").write_text(
        json.dumps(pooled, indent=2, sort_keys=True), encoding="utf-8"
    )
    source_hash = sha256_file(source_predictions)
    provenance = {
        "source_predictions_path": str(source_predictions.resolve()),
        "source_predictions_sha256": source_hash,
        "reuses_model_predictions": True,
    }
    summary = {
        "schema_version": experiment.schema_version,
        "legacy": False,
        "config": asdict(experiment.tft),
        "pooled_metrics": pooled,
        "source_predictions_path": provenance["source_predictions_path"],
        "source_predictions_sha256": source_hash,
        "reuses_model_predictions": True,
        "provenance": provenance,
    }
    (output_root / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _attach_fold_baselines(
    predictions: pd.DataFrame,
    train_frame: pd.DataFrame,
    through_test: pd.DataFrame,
    issues: pd.DatetimeIndex,
    horizon_steps: int,
    experiment_schema: str,
) -> pd.DataFrame:
    keys = ["issue_timestamp_utc", "target_timestamp_utc", "horizon_step"]
    harmonic = (
        HarmonicSmartPersistence()
        .fit(train_frame)
        .predict_long(through_test, issues, horizon_steps)
    )
    if experiment_schema == "transfer-experiment-v1":
        return predictions.merge(harmonic, on=keys, how="left", validate="many_to_one")
    if experiment_schema != "transfer-experiment-v2":
        raise ValueError(f"Unsupported transfer experiment schema: {experiment_schema}")
    harmonic = harmonic.rename(columns={"smart_persistence": "legacy_harmonic_smart_persistence"})
    physical = PhysicalSmartPersistence().predict_long(through_test, issues, horizon_steps)
    return predictions.merge(harmonic, on=keys, how="left", validate="many_to_one").merge(
        physical, on=keys, how="left", validate="many_to_one"
    )


def _baseline_identity(experiment_schema: str) -> dict[str, Any]:
    if experiment_schema == "transfer-experiment-v1":
        return BASELINE_IDENTITY_LEGACY.copy()
    if experiment_schema == "transfer-experiment-v2":
        return BASELINE_IDENTITY_PHYSICAL.copy()
    raise ValueError(f"Unsupported transfer experiment schema: {experiment_schema}")


def _load_or_create_run_state(
    path: Path,
    *,
    resume: bool,
    dataset_path: Path,
    pretrained_checkpoint: Path,
    config: TFTTransferConfig,
    experiment_schema: str,
) -> dict[str, Any]:
    expected = {
        "dataset_path": str(dataset_path.resolve()),
        "pretrained_checkpoint": str(pretrained_checkpoint.resolve()),
        "config": asdict(config),
        "experiment_schema": experiment_schema,
    }
    if resume:
        if not path.is_file():
            raise FileNotFoundError(
                f"Cannot resume: run state is missing at {path}. "
                "Start a fresh run or use the original output directory."
            )
        state = json.loads(path.read_text(encoding="utf-8"))
        for key in ("dataset_path", "pretrained_checkpoint", "config", "experiment_schema"):
            if state.get(key) != expected[key]:
                raise ValueError(
                    f"Cannot resume: run state mismatch for {key}. "
                    "Use the same dataset, checkpoint, configuration, and schema."
                )
        state.setdefault("completed_folds", [])
        state.setdefault("best_epochs", [])
        return state
    state = {
        "state_version": 1,
        **expected,
        "completed_folds": [],
        "best_epochs": [],
    }
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return state


def _latest_checkpoint(directory: Path) -> Path | None:
    checkpoints = sorted(directory.glob("*.ckpt"), key=lambda path: path.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def _reusable_pretraining_checkpoint(
    dataset_path: Path,
    checkpoint_root: Path,
    config: TFTTransferConfig,
    source_hashes: dict[str, str],
) -> tuple[Path, dict[str, Any]] | None:
    """Return a validated completed pretraining checkpoint, if available."""
    manifest_path = checkpoint_root / "bundle" / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("config") != asdict(config):
        return None
    if {
        str(key): str(value) for key, value in manifest.get("source_hashes", {}).items()
    } != {str(key): str(value) for key, value in source_hashes.items()}:
        return None
    recorded_dataset = manifest.get("dataset_path")
    if recorded_dataset is not None and Path(recorded_dataset).resolve() != dataset_path.resolve():
        return None
    metadata = manifest.get("training", {})
    checkpoint_value = metadata.get("best_model_path")
    if not checkpoint_value:
        return None
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_file():
        return None
    return checkpoint, metadata


def _manifest_source_hashes(dataset_path: Path) -> dict[str, str]:
    manifest_path = dataset_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in manifest.get("source_hashes", {}).items()}


def _capacity_metadata(frame: pd.DataFrame) -> dict[str, float]:
    sites = frame.drop_duplicates("site_id").set_index("site_id")["installed_capacity_mw"]
    return {str(site_id): float(capacity) for site_id, capacity in sites.items()}
