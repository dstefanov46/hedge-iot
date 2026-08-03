from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from math import floor
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from aurora.data.canonical import read_processed_dataset, sha256_file
from aurora.evaluation.cross_validation import CVFold, generate_expanding_finnish_folds
from aurora.evaluation.transfer import (
    BASELINE_IDENTITY_LEGACY,
    BASELINE_IDENTITY_PHYSICAL,
    evaluate_long_forecasts,
)
from aurora.forecasting.baselines import HarmonicSmartPersistence, PhysicalSmartPersistence
from aurora.forecasting.tft_transfer import (
    TFTTransferConfig,
    create_dataset,
    fit_tft,
    predict_long,
    save_model_bundle,
)


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
                **{**values.get("tft", {}), **_satellite_tft_values(values.get("satellite"))}
            ),
            bootstrap_replicates=int(values.get("bootstrap_replicates", 100)),
            bootstrap_block_length=int(values.get("bootstrap_block_length", 15)),
            skill_threshold=float(values.get("skill_threshold", 0.10)),
            satellite=values.get("satellite"),
        )


def _satellite_tft_values(values: dict[str, object] | None) -> dict[str, object]:
    if not values or not bool(values.get("enabled", False)):
        return {}
    return {
        "satellite_enabled": True,
        "satellite_embedding_dim": int(values.get("embedding_dim", 32)),
        "satellite_missing_value": float(values.get("missing_value", 0.0)),
        "satellite_max_alignment_minutes": float(values.get("max_alignment_minutes", 7.5)),
    }


def pretrain_slovenian_tft(
    dataset_path: Path,
    checkpoint_root: Path,
    config: TFTTransferConfig,
    source_hashes: dict[str, str] | None = None,
    validation_days: int = 30,
) -> tuple[Path, dict[str, Any]]:
    frame = read_processed_dataset(dataset_path)
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
    model, metadata = fit_tft(training, validation, config, checkpoint_root)
    checkpoint_path = Path(metadata["best_model_path"])
    save_model_bundle(
        checkpoint_root / "bundle",
        model,
        training,
        config,
        metadata,
        source_hashes,
        _capacity_metadata(frame),
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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[int]]:
    frame = read_processed_dataset(dataset_path)
    folds = folds or generate_expanding_finnish_folds(
        int(frame["timestamp_utc"].dt.year.mode().iloc[0]), timezone="Europe/Helsinki"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    all_predictions: list[pd.DataFrame] = []
    best_epochs: list[int] = []
    for fold in folds:
        fold_predictions, transfer_metadata = _run_fold(
            frame=frame,
            fold=fold,
            pretrained_checkpoint=pretrained_checkpoint,
            model_root=model_root / f"fold_{fold.fold_id}",
            config=config,
            experiment_schema=experiment_schema,
        )
        best_epoch = transfer_metadata.get("best_epoch")
        if best_epoch:
            best_epochs.append(int(best_epoch))
        all_predictions.append(fold_predictions)
        fold_predictions.to_csv(output_root / f"predictions_fold_{fold.fold_id}.csv", index=False)
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


def _run_fold(
    frame: pd.DataFrame,
    fold: CVFold,
    pretrained_checkpoint: Path,
    model_root: Path,
    config: TFTTransferConfig,
    experiment_schema: str = "transfer-experiment-v2",
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
    )
    scratch_model, scratch_metadata = fit_tft(
        training,
        validation,
        config,
        model_root / "scratch",
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


def _manifest_source_hashes(dataset_path: Path) -> dict[str, str]:
    manifest_path = dataset_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in manifest.get("source_hashes", {}).items()}


def _capacity_metadata(frame: pd.DataFrame) -> dict[str, float]:
    sites = frame.drop_duplicates("site_id").set_index("site_id")["installed_capacity_mw"]
    return {str(site_id): float(capacity) for site_id, capacity in sites.items()}
