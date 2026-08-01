from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurora.data.canonical import prepare_hirvensalmi_dataset
from aurora.evaluation.metrics import mae, mse, skill_score
from aurora.forecasting.baselines import smart_persistence_forecast

DEFAULT_QUANTILES = (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names",
)


@dataclass(frozen=True)
class ComparisonConfig:
    workbook_path: Path
    output_root: Path = Path("outputs/tft_plan_comparison")
    model_root: Path = Path("models/tft_hirvensalmi")
    site_id: str = "hirvensalmi"
    source_timezone: str = "Europe/Helsinki"
    latitude: float = 61.64
    longitude: float = 26.78
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    encoder_length: int = 96
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 4096
    hidden_size: int = 16
    attention_head_size: int = 2
    dropout: float = 0.1
    learning_rate: float = 0.001
    random_seed: int = 42
    checkpoint_path: Path | None = None


@dataclass(frozen=True)
class ComparisonOutputs:
    run_dir: Path
    model_dir: Path
    report_path: Path
    metrics_path: Path
    stats_path: Path
    predictions_path: Path
    data_quality_path: Path


def run_tft_plan_comparison(config: ComparisonConfig) -> ComparisonOutputs:
    """Train TFT and compare it with the workbook forecast plan."""
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_root / run_id
    model_dir = config.model_root / run_id
    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frame, data_quality = load_and_preprocess(config)
    split_frame = assign_splits(frame, config.train_fraction, config.validation_fraction)
    predictions, training_metadata = train_and_predict_tft(split_frame, config, model_dir)
    predictions = add_baseline_and_plan_predictions(predictions)

    metrics = compute_metrics_table(predictions, training_metadata)
    stats = compute_forecast_stats(predictions)

    metrics_path = run_dir / "metrics_summary.csv"
    stats_path = run_dir / "forecast_stats.csv"
    predictions_path = run_dir / "predictions.csv"
    data_quality_path = run_dir / "data_quality.json"
    report_path = run_dir / "comparison_report.md"

    metrics.to_csv(metrics_path, index=False)
    stats.to_csv(stats_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    data_quality_path.write_text(json.dumps(data_quality, indent=2, default=str), encoding="utf-8")
    write_plots(predictions, figures_dir)
    write_report(
        report_path=report_path,
        metrics=metrics,
        stats=stats,
        data_quality=data_quality,
        training_metadata=training_metadata,
    )

    return ComparisonOutputs(
        run_dir=run_dir,
        model_dir=model_dir,
        report_path=report_path,
        metrics_path=metrics_path,
        stats_path=stats_path,
        predictions_path=predictions_path,
        data_quality_path=data_quality_path,
    )


def load_and_preprocess(config: ComparisonConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    canonical, manifest = prepare_hirvensalmi_dataset(config.workbook_path)
    cleaned = canonical.loc[canonical["observation_available"]].copy().reset_index(drop=True)
    cleaned = cleaned.rename(columns={"interval_energy_mwh": "actual_mwh"})
    cleaned["actual_mwh"] = cleaned["actual_mwh"].clip(lower=0)
    cleaned["planned_mwh"] = cleaned["planned_mwh"].clip(lower=0)
    cleaned["series_id"] = config.site_id
    cleaned["hour_sin"] = cleaned["day_sin_1"]
    cleaned["hour_cos"] = cleaned["day_cos_1"]
    cleaned["day_sin"] = cleaned["year_sin_1"]
    cleaned["day_cos"] = cleaned["year_cos_1"]
    cleaned["clear_sky_w_m2"] = cleaned["clear_sky_norm"] * 1361.0
    gap_minutes = cleaned["timestamp_utc"].diff().dt.total_seconds().div(60).dropna()

    data_quality = {
        "source": str(config.workbook_path),
        "schema_version": manifest["schema_version"],
        "raw_records": manifest["observed_row_count"],
        "records_after_duplicate_removal": int(len(cleaned)),
        "duplicate_timestamps_removed": 0,
        "masked_gaps": manifest["gap_count"],
        "exclusions": manifest["exclusions"],
        "quality_flag_counts": canonical["quality_status"].value_counts().to_dict(),
        "source_sheet_counts": cleaned["source_sheet"].value_counts(dropna=False).to_dict(),
        "timestamp_min_utc": cleaned["timestamp_utc"].min(),
        "timestamp_max_utc": cleaned["timestamp_utc"].max(),
        "gap_minutes_counts": gap_minutes.round(6).value_counts().head(20).to_dict(),
        "actual_mwh_sum": float(cleaned["actual_mwh"].sum()),
        "planned_mwh_sum": float(cleaned["planned_mwh"].sum()),
        "actual_nonzero_count": int((cleaned["actual_mwh"] > 0).sum()),
        "planned_nonzero_count": int((cleaned["planned_mwh"] > 0).sum()),
    }
    return cleaned, data_quality


def assign_splits(
    frame: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> pd.DataFrame:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1.")

    out = frame.copy()
    train_end = int(len(out) * train_fraction)
    validation_end = int(len(out) * (train_fraction + validation_fraction))
    out["split"] = "test"
    out.loc[: train_end - 1, "split"] = "train"
    out.loc[train_end : validation_end - 1, "split"] = "validation"
    return out


def train_and_predict_tft(
    frame: pd.DataFrame,
    config: ComparisonConfig,
    model_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        import torch
        from lightning.pytorch import Trainer, seed_everything
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.metrics import QuantileLoss
    except ImportError as exc:
        raise ImportError(
            "TFT comparison requires ML dependencies. Install with: "
            'python -m pip install -e ".[ml]"'
        ) from exc

    seed_everything(config.random_seed, workers=True)
    train_cutoff = int(frame.loc[frame["split"] == "train", "time_idx"].max())
    validation_start = int(frame.loc[frame["split"] == "validation", "time_idx"].min())
    test_start = int(frame.loc[frame["split"] == "test", "time_idx"].min())

    target_scale = max(float(frame.loc[frame["split"] == "train", "actual_mwh"].max()), 1.0)
    model_frame = frame.copy()
    model_frame["target_norm"] = model_frame["actual_mwh"] / target_scale

    known_reals = [
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "solar_azimuth",
        "solar_elevation",
        "clear_sky_norm",
    ]
    training = TimeSeriesDataSet(
        model_frame[model_frame["time_idx"] <= train_cutoff],
        time_idx="time_idx",
        target="target_norm",
        group_ids=["series_id"],
        min_encoder_length=min(24, config.encoder_length),
        max_encoder_length=config.encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=["series_id"],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=["target_norm"],
        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training,
        model_frame[
            model_frame["time_idx"]
            <= model_frame.loc[frame["split"] == "validation", "time_idx"].max()
        ],
        min_prediction_idx=validation_start,
        stop_randomization=True,
    )
    test = TimeSeriesDataSet.from_dataset(
        training,
        model_frame,
        min_prediction_idx=test_start,
        stop_randomization=True,
    )

    if config.checkpoint_path is not None:
        best_model = TemporalFusionTransformer.load_from_checkpoint(str(config.checkpoint_path))
        checkpoint_metadata = parse_checkpoint_metadata(config.checkpoint_path)
        prediction_frame = _predict_frame_from_model(
            model=best_model,
            validation=validation,
            test=test,
            frame=frame,
            batch_size=config.batch_size,
            target_scale=target_scale,
        )
        metadata = {
            "target_scale_mwh": target_scale,
            "best_model_path": str(config.checkpoint_path),
            "best_validation_loss": checkpoint_metadata.get("validation_loss"),
            "current_epoch": checkpoint_metadata.get("epoch"),
            "quantiles": list(DEFAULT_QUANTILES),
            "torch_version": torch.__version__,
            "loaded_from_checkpoint": True,
        }
        return prediction_frame, metadata

    train_loader = training.to_dataloader(train=True, batch_size=config.batch_size, num_workers=0)
    validation_loader = validation.to_dataloader(
        train=False,
        batch_size=config.batch_size * 2,
        num_workers=0,
    )

    checkpoint = ModelCheckpoint(
        dirpath=str(model_dir),
        filename="tft-{epoch:02d}-{val_loss:.5f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-5,
        patience=config.patience,
        mode="min",
    )
    trainer = Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[checkpoint, early_stop],
        enable_model_summary=False,
        enable_progress_bar=True,
        logger=False,
    )
    loss = QuantileLoss(quantiles=list(DEFAULT_QUANTILES))
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=min(config.hidden_size, 8),
        output_size=len(DEFAULT_QUANTILES),
        loss=loss,
        reduce_on_plateau_patience=2,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    best_model = model
    if checkpoint.best_model_path:
        best_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)

    prediction_frame = _predict_frame_from_model(
        model=best_model,
        validation=validation,
        test=test,
        frame=frame,
        batch_size=config.batch_size,
        target_scale=target_scale,
    )

    metadata = {
        "target_scale_mwh": target_scale,
        "best_model_path": checkpoint.best_model_path,
        "best_validation_loss": float(checkpoint.best_model_score)
        if checkpoint.best_model_score is not None
        else None,
        "current_epoch": int(trainer.current_epoch),
        "quantiles": list(DEFAULT_QUANTILES),
        "torch_version": torch.__version__,
    }
    return prediction_frame, metadata


def _predict_frame_from_model(
    model: Any,
    validation: Any,
    test: Any,
    frame: pd.DataFrame,
    batch_size: int,
    target_scale: float,
) -> pd.DataFrame:
    prediction_frame = frame.copy()
    for split, dataset in [("validation", validation), ("test", test)]:
        split_predictions = _predict_quantiles(model, dataset, batch_size)
        for column in [f"tft_q{int(q * 100):02d}_mwh" for q in DEFAULT_QUANTILES]:
            prediction_frame.loc[prediction_frame["split"] == split, column] = np.nan
        for _, row in split_predictions.iterrows():
            mask = prediction_frame["time_idx"] == int(row["time_idx"])
            for q in DEFAULT_QUANTILES:
                prediction_frame.loc[mask, f"tft_q{int(q * 100):02d}_mwh"] = (
                    row[f"q{int(q * 100):02d}"] * target_scale
                )

    prediction_frame["tft_q50_mwh"] = prediction_frame["tft_q50_mwh"].clip(lower=0)
    quantile_columns = [f"tft_q{int(q * 100):02d}_mwh" for q in DEFAULT_QUANTILES]
    prediction_frame[quantile_columns] = prediction_frame[quantile_columns].clip(lower=0)
    return prediction_frame


def parse_checkpoint_metadata(checkpoint_path: Path) -> dict[str, float | int | None]:
    match = re.search(r"epoch=(\d+)-val_loss=([0-9]+(?:\.[0-9]+)?)", checkpoint_path.name)
    if match is None:
        return {"epoch": None, "validation_loss": None}
    return {"epoch": int(match.group(1)), "validation_loss": float(match.group(2))}


def _predict_quantiles(model: Any, dataset: Any, batch_size: int) -> pd.DataFrame:
    dataloader = dataset.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)
    raw = model.predict(dataloader, mode="raw", return_x=True)
    output = raw.output
    if hasattr(output, "prediction"):
        values = output.prediction
    elif isinstance(output, dict):
        values = output["prediction"]
    else:
        values = output
    predictions = values.detach().cpu().numpy()
    decoder_time_idx = raw.x["decoder_time_idx"][:, 0].detach().cpu().numpy()

    rows = {"time_idx": decoder_time_idx.astype(int)}
    for i, q in enumerate(DEFAULT_QUANTILES):
        rows[f"q{int(q * 100):02d}"] = predictions[:, 0, i]
    return pd.DataFrame(rows)


def add_baseline_and_plan_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    clear_sky_proxy = pd.Series(out["clear_sky_w_m2"].to_numpy(), index=out["time_idx"])
    actual = pd.Series(out["actual_mwh"].to_numpy(), index=out["time_idx"])
    out["smart_persistence_mwh"] = smart_persistence_forecast(actual, clear_sky_proxy).to_numpy()
    out["existing_plan_mwh"] = out["planned_mwh"]
    return out


def compute_metrics_table(frame: pd.DataFrame, training_metadata: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["validation", "test"]:
        split_frame = frame[frame["split"] == split]
        for model_name, forecast_column in [
            ("TFT q50", "tft_q50_mwh"),
            ("Existing plan", "existing_plan_mwh"),
        ]:
            metric_values = point_metrics(
                split_frame["actual_mwh"],
                split_frame[forecast_column],
                split_frame["smart_persistence_mwh"],
            )
            row = {"split": split, "model": model_name, **metric_values}
            if model_name == "TFT q50":
                row["quantile_loss"] = quantile_loss_from_frame(split_frame)
            else:
                row["quantile_loss"] = np.nan
            rows.append(row)

    train_loss = training_metadata.get("best_validation_loss")
    rows.append(
        {
            "split": "training",
            "model": "TFT q50",
            "mse": np.nan,
            "mae": np.nan,
            "skill_score": np.nan,
            "quantile_loss": train_loss,
        }
    )
    return pd.DataFrame(rows)


def point_metrics(
    actual: pd.Series,
    forecast: pd.Series,
    reference: pd.Series,
) -> dict[str, float]:
    aligned = pd.concat(
        [actual.rename("actual"), forecast.rename("forecast"), reference.rename("reference")],
        axis=1,
    ).dropna()
    return {
        "mse": mse(aligned["actual"], aligned["forecast"]),
        "mae": mae(aligned["actual"], aligned["forecast"]),
        "skill_score": skill_score(
            aligned["actual"],
            aligned["forecast"],
            aligned["reference"],
            metric="mse",
        ),
    }


def quantile_loss_from_frame(frame: pd.DataFrame) -> float:
    actual = frame["actual_mwh"].to_numpy()
    losses = []
    for q in DEFAULT_QUANTILES:
        column = f"tft_q{int(q * 100):02d}_mwh"
        prediction = frame[column].to_numpy()
        mask = ~np.isnan(prediction)
        error = actual[mask] - prediction[mask]
        losses.append(np.maximum(q * error, (q - 1) * error))
    if not losses:
        return float("nan")
    return float(np.mean(np.column_stack(losses)))


def compute_forecast_stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ["validation", "test"]:
        split_frame = frame[frame["split"] == split]
        for model_name, forecast_column in [
            ("TFT q50", "tft_q50_mwh"),
            ("Existing plan", "existing_plan_mwh"),
        ]:
            rows.append(
                {
                    "split": split,
                    "model": model_name,
                    **forecast_diagnostics(
                        actual=split_frame["actual_mwh"],
                        forecast=split_frame[forecast_column],
                        clear_sky=split_frame["clear_sky_w_m2"],
                    ),
                }
            )
        rows.append(
            {
                "split": split,
                "model": "TFT interval",
                **tft_interval_diagnostics(split_frame),
            }
        )
    return pd.DataFrame(rows)


def forecast_diagnostics(
    actual: pd.Series,
    forecast: pd.Series,
    clear_sky: pd.Series,
) -> dict[str, float]:
    aligned = pd.concat(
        [actual.rename("actual"), forecast.rename("forecast"), clear_sky.rename("clear_sky")],
        axis=1,
    ).dropna()
    errors = aligned["forecast"] - aligned["actual"]
    ramp = aligned["forecast"].diff().abs().dropna()
    night = aligned["clear_sky"] <= 1
    return {
        "count": float(len(aligned)),
        "min": float(aligned["forecast"].min()),
        "max": float(aligned["forecast"].max()),
        "mean": float(aligned["forecast"].mean()),
        "std": float(aligned["forecast"].std(ddof=0)),
        "p01": float(aligned["forecast"].quantile(0.01)),
        "p05": float(aligned["forecast"].quantile(0.05)),
        "p50": float(aligned["forecast"].quantile(0.5)),
        "p95": float(aligned["forecast"].quantile(0.95)),
        "p99": float(aligned["forecast"].quantile(0.99)),
        "sum": float(aligned["forecast"].sum()),
        "zero_share": float((aligned["forecast"] <= 1e-9).mean()),
        "nonzero_share": float((aligned["forecast"] > 1e-9).mean()),
        "bias_mean": float(errors.mean()),
        "bias_sum": float(errors.sum()),
        "correlation": float(aligned["actual"].corr(aligned["forecast"])),
        "negative_count": float((aligned["forecast"] < 0).sum()),
        "ramp_abs_mean": float(ramp.mean()) if len(ramp) else 0.0,
        "ramp_abs_p95": float(ramp.quantile(0.95)) if len(ramp) else 0.0,
        "night_positive_share": float((aligned.loc[night, "forecast"] > 1e-6).mean())
        if night.any()
        else 0.0,
    }


def tft_interval_diagnostics(frame: pd.DataFrame) -> dict[str, float]:
    needed = ["actual_mwh", "tft_q10_mwh", "tft_q90_mwh"]
    aligned = frame[needed].dropna()
    if aligned.empty:
        return {"count": 0.0, "q10_q90_coverage": np.nan, "q10_q90_width_mean": np.nan}
    return {
        "count": float(len(aligned)),
        "q10_q90_coverage": float(
            (
                (aligned["actual_mwh"] >= aligned["tft_q10_mwh"])
                & (aligned["actual_mwh"] <= aligned["tft_q90_mwh"])
            ).mean()
        ),
        "q10_q90_width_mean": float((aligned["tft_q90_mwh"] - aligned["tft_q10_mwh"]).mean()),
    }


def write_plots(frame: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    test = frame[frame["split"] == "test"].dropna(subset=["tft_q50_mwh"]).copy()
    if test.empty:
        return
    plot_data = test.tail(min(len(test), 96 * 14))
    x = pd.to_datetime(plot_data["timestamp_utc"])

    plt.figure(figsize=(14, 6))
    plt.plot(x, plot_data["actual_mwh"], label="Actual", linewidth=1.5)
    plt.plot(x, plot_data["existing_plan_mwh"], label="Existing plan", linewidth=1)
    plt.plot(x, plot_data["tft_q50_mwh"], label="TFT q50", linewidth=1)
    plt.legend()
    plt.title("Actual vs Forecasts (test tail)")
    plt.tight_layout()
    plt.savefig(figures_dir / "actual_vs_forecasts.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(test["tft_q50_mwh"] - test["actual_mwh"], bins=60, alpha=0.6, label="TFT q50")
    plt.hist(
        test["existing_plan_mwh"] - test["actual_mwh"],
        bins=60,
        alpha=0.6,
        label="Existing plan",
    )
    plt.legend()
    plt.title("Forecast Error Distribution (test)")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(x, plot_data["actual_mwh"], label="Actual", linewidth=1.5)
    plt.plot(x, plot_data["tft_q50_mwh"], label="TFT q50", linewidth=1)
    plt.fill_between(
        x,
        plot_data["tft_q10_mwh"].to_numpy(),
        plot_data["tft_q90_mwh"].to_numpy(),
        alpha=0.25,
        label="TFT q10-q90",
    )
    plt.legend()
    plt.title("TFT Quantile Band (test tail)")
    plt.tight_layout()
    plt.savefig(figures_dir / "tft_quantile_band.png", dpi=160)
    plt.close()


def write_report(
    report_path: Path,
    metrics: pd.DataFrame,
    stats: pd.DataFrame,
    data_quality: dict[str, Any],
    training_metadata: dict[str, Any],
) -> None:
    lines = [
        "# TFT vs Existing Forecast Plan",
        "",
        "## Data Quality",
        "",
        f"- Source: `{data_quality['source']}`",
        f"- Records after cleaning: {data_quality['records_after_duplicate_removal']}",
        f"- Duplicate timestamps removed: {data_quality['duplicate_timestamps_removed']}",
        f"- Quality flags: `{data_quality['quality_flag_counts']}`",
        "",
        "## Training",
        "",
        f"- Best validation quantile loss: {training_metadata.get('best_validation_loss')}",
        f"- Best model path: `{training_metadata.get('best_model_path')}`",
        f"- Epochs completed: {training_metadata.get('current_epoch')}",
        "",
        "## Metrics",
        "",
        dataframe_to_markdown(metrics),
        "",
        "## Forecast Diagnostics",
        "",
        dataframe_to_markdown(stats),
        "",
        "## Notes",
        "",
        "- Existing plan has no quantile outputs, so quantile loss is reported as N/A.",
        "- Skill score is computed against smart persistence using MSE.",
        "- TFT metrics use the q50 median forecast as the point forecast.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without optional pandas tabulate dependency."""
    if frame.empty:
        return "_No rows._"

    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6g}"
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )

    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)
