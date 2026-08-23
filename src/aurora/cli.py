import json
import warnings
from pathlib import Path

import typer

from aurora.data.canonical import prepare_hirvensalmi_dataset
from aurora.data.hirvensalmi import (
    hirvensalmi_records_to_frame,
    load_hirvensalmi_workbook,
    parse_hirvensalmi_workbook,
)
from aurora.data.slovenia import prepare_slovenian_dataset
from aurora.experiments.tft_plan_comparison import ComparisonConfig, run_tft_plan_comparison
from aurora.experiments.transfer import (
    fit_final_finnish_checkpoint,
    pretrain_slovenian_tft,
    reevaluate_transfer_predictions,
    report_finnish_weather_ablation,
    run_finnish_rolling_evaluation,
    run_tft_loss_sweep,
    run_transfer_experiment,
)
from aurora.forecasting.tft_transfer import TFTTransferConfig
from aurora.satellite.acquisition import collect_in_chunks, download_products
from aurora.satellite.batch import run_satellite_batch
from aurora.satellite.config import SatelliteConfig, load_site_config

app = typer.Typer(help="AURORA forecasting and optimization workflows.")


@app.command("satellite-batch")
def satellite_batch(
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    site_config: Path = typer.Option(..., exists=True, dir_okay=False),
    config: Path = typer.Option(Path("configs/satellite.toml"), "--config"),
    delete_native: bool = typer.Option(False, "--delete-native"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    chunk_hours: int | None = typer.Option(None, min=1),
    download_workers: int | None = typer.Option(None, min=1, help="Concurrent download workers."),
    processing_workers: int | None = typer.Option(None, min=1, help="Concurrent Satpy workers."),
    pipeline_queue_size: int | None = typer.Option(
        None, min=1, help="Bounded pipeline queue size."
    ),
) -> None:
    """Collect, process, verify backup, and optionally delete native archives."""
    from datetime import datetime

    satellite_config = _satellite_config(config)
    overrides = {
        k: v
        for k, v in {
            "download_workers": download_workers,
            "processing_workers": processing_workers,
            "pipeline_queue_size": pipeline_queue_size,
        }.items()
        if v is not None
    }
    if overrides:
        from dataclasses import replace

        satellite_config = replace(satellite_config, **overrides)
    result = run_satellite_batch(
        datetime.fromisoformat(start.replace("Z", "+00:00")),
        datetime.fromisoformat(end.replace("Z", "+00:00")),
        load_site_config(site_config),
        satellite_config,
        delete_native=delete_native,
        dry_run=dry_run,
        chunk_hours=chunk_hours,
    )
    typer.echo(
        f"Batch {result['batch_id']}: {len(result['products'])} products; "
        f"{len(result['failed_products'])} failed; "
        f"{len(result['unavailable_products'])} unavailable; "
        f"interval {result['interval']['start']} to {result['interval']['end']}"
    )
    discovery = result["discovery"]
    typer.echo(
        "Discovery: "
        f"covered chunks={discovery['covered_chunks']}/{discovery['total_chunks']}; "
        f"remaining uncovered chunks={discovery['remaining_uncovered_chunks']}; "
        "confirmed missing 15-minute source slots="
        f"{discovery['confirmed_missing_15_minute_source_slots']}"
    )
    typer.echo(
        "Workers: "
        f"download={result['workers']['download']}, "
        f"processing={result['workers']['processing']}"
    )
    executor = result["processing_executor"]
    typer.echo(
        f"Processing executor: {executor['type']}; observed workers="
        f"{executor['observed_workers']}/{executor['configured_workers']}"
    )
    for worker in executor["workers"]:
        typer.echo(
            f"  PID {worker['pid']}: tasks={worker['tasks']}, "
            f"wall={worker['wall_seconds']:.3f}s, cpu={worker['cpu_seconds']:.3f}s, "
            f"affinity={worker['affinity_cpus']}, "
            f"native_threads={worker['native_threads_before']}->"
            f"{worker['native_threads_during_decode']}->"
            f"{worker['native_threads_after_decode']}, "
            f"dask={worker['dask_scheduler']}/{worker['dask_workers']}"
        )
        if worker["threadpools_during_decode"]:
            pools = ",".join(
                f"{pool['prefix']}={pool['num_threads']}"
                for pool in worker["threadpools_during_decode"]
            )
            typer.echo(f"    native pools during decode: {pools}")
    typer.echo("Timing summary (seconds):")
    for stage, elapsed in result["timings"].items():
        typer.echo(f"  {stage}: {elapsed:.3f}")
    if result["failed_products"]:
        raise typer.Exit(1)


@app.command("satellite-collect")
def satellite_collect(
    start: str = typer.Option("2025-01-01T00:00:00Z"),
    end: str = typer.Option("2026-01-01T00:00:00Z"),
    site_config: Path = typer.Option(..., exists=True, dir_okay=False),
    config: Path = typer.Option(Path("configs/satellite.toml"), "--config"),
    chunk_hours: int | None = typer.Option(None, min=1),
) -> None:
    """Collect and validate HRSEVIRI products in resumable chunks."""
    from datetime import datetime

    satellite_config = _satellite_config(config)
    records = collect_in_chunks(
        datetime.fromisoformat(start.replace("Z", "+00:00")),
        datetime.fromisoformat(end.replace("Z", "+00:00")),
        load_site_config(site_config),
        satellite_config,
        chunk_hours=chunk_hours,
    )
    typer.echo(f"Collected or reused {len(records)} products; manifest is resumable")


@app.command("satellite-download")
def satellite_download(
    start: str = typer.Option(..., help="UTC ISO-8601 start time."),
    end: str = typer.Option(..., help="UTC ISO-8601 end time."),
    site_config: Path = typer.Option(..., exists=True, dir_okay=False),
    config: Path = typer.Option(Path("configs/satellite.toml"), "--config"),
) -> None:
    """Download HRSEVIRI native products and write an acquisition manifest."""
    from datetime import datetime

    satellite_config = _satellite_config(config)
    records = download_products(
        datetime.fromisoformat(start.replace("Z", "+00:00")),
        datetime.fromisoformat(end.replace("Z", "+00:00")),
        load_site_config(site_config),
        satellite_config,
    )
    typer.echo(f"Downloaded or reused {len(records)} HRSEVIRI products")


@app.command("satellite-preprocess")
def satellite_preprocess(
    manifest: Path = typer.Option(..., exists=True, dir_okay=False),
    site_config: Path = typer.Option(..., exists=True, dir_okay=False),
    config: Path = typer.Option(Path("configs/satellite.toml"), "--config"),
) -> None:
    """Decode native HRSEVIRI products into site-centered patches."""
    import json

    from aurora.satellite.preprocess import (
        decode_native,
        load_normalization_stats,
        preprocess_channels,
    )

    values = json.loads(manifest.read_text(encoding="utf-8"))
    site = load_site_config(site_config)
    satellite_config = _satellite_config(config)
    means = scales = None
    normalization_version = satellite_config.normalization_version
    if satellite_config.normalization_stats_uri:
        means, scales, normalization_version = load_normalization_stats(
            satellite_config.normalization_stats_uri
        )
    records = []
    for product in values.get("products", []):
        timestamp = datetime_from_record(product["sensing_time"])
        channels = decode_native(product["local_uri"], site)
        records.append(
            preprocess_channels(
                channels,
                site,
                product["product_id"],
                timestamp,
                f"{satellite_config.processed_uri}/patches/{product['product_id']}.zarr",
                means=means,
                scales=scales,
                normalization_version=normalization_version,
            )
        )
    output = Path(satellite_config.processed_uri) / "manifest.parquet"
    import pandas as pd

    pd.DataFrame(records).to_parquet(output, index=False)
    typer.echo(f"Wrote {len(records)} patch records to {output}")


def datetime_from_record(value: str):
    from datetime import datetime

    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _satellite_config(path: Path) -> SatelliteConfig:
    if not path.is_file():
        raise typer.BadParameter(
            f"Missing satellite config file: {path}. Copy configs/satellite.example.toml "
            "to configs/satellite.toml and fill in the credentials."
        )
    return SatelliteConfig.from_file(path)


@app.command("satellite-embed")
def satellite_embed(
    patch_store: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(...),
    checkpoint: Path | None = typer.Option(None, exists=True, dir_okay=False),
) -> None:
    """Generate provenance-preserving frozen CNN embeddings from patch files."""
    import numpy as np
    import pandas as pd

    from aurora.satellite.embedding import FrozenCNNEncoder
    from aurora.satellite.features import derive_patch_features

    encoder = FrozenCNNEncoder.from_checkpoint(checkpoint) if checkpoint else FrozenCNNEncoder()
    rows = []
    patch_paths = [*patch_store.rglob("*.npz"), *patch_store.rglob("*.zarr")]
    for path in sorted(patch_paths):
        if path.suffix == ".zarr":
            import zarr

            patch = np.asarray(zarr.open(path, mode="r"))
        else:
            patch = np.load(path)["patch"]
        metadata_path = path.with_suffix(path.suffix + ".json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        )
        row = derive_patch_features(
            patch,
            str(path),
            site_id=metadata.get("site_id") or metadata.get("site", {}).get("site_id"),
            timestamp_utc=metadata.get("timestamp_utc"),
            product_id=metadata.get("product_id"),
        )
        row.update(
            {
                f"satellite_embedding_{i:02d}": float(value)
                for i, value in enumerate(encoder.encode(patch))
            }
        )
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    typer.echo(f"Wrote {len(rows)} satellite embeddings to {output}")


@app.command()
def inspect_workbook(path: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Print basic sheet and row counts for a Hirvensalmi workbook."""
    workbook = load_hirvensalmi_workbook(path)
    for sheet in workbook.sheets:
        typer.echo(f"{sheet.name}: {sheet.rows} rows, {sheet.columns} columns")


@app.command()
def parse_workbook(
    path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output", "-o", help="Optional CSV output path."),
    site_id: str = typer.Option("hirvensalmi", help="Canonical site identifier."),
    source_timezone: str = typer.Option("Europe/Helsinki", help="Workbook source timezone."),
) -> None:
    """Parse Hirvensalmi workbook rows into canonical production observations."""
    result = parse_hirvensalmi_workbook(path, site_id=site_id, source_timezone=source_timezone)
    frame = hirvensalmi_records_to_frame(result.records)
    typer.echo(f"Parsed {len(result.records)} production observations")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        typer.echo(f"Wrote {output}")


@app.command()
def prepare_finnish(
    path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path = typer.Option(Path("data/processed/hirvensalmi.parquet")),
) -> None:
    """Prepare the corrected Hirvensalmi canonical 15-minute dataset."""
    frame, manifest = prepare_hirvensalmi_dataset(path, output_path=output)
    typer.echo(
        f"Wrote {len(frame)} rows to {output} "
        f"({manifest['gap_count']} masked gaps, {len(manifest['exclusions'])} exclusions)"
    )


@app.command()
def prepare_slovenian(
    project_root: Path = typer.Argument(..., exists=True, file_okay=False),
    metadata: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(Path("data/processed/slovenia.parquet")),
    pv_path: Path | None = typer.Option(None, exists=True, dir_okay=False),
    timestamp_timezone: str = typer.Option("UTC"),
    value_kind: str = typer.Option("power_kw"),
) -> None:
    """Adapt the restored Slovenian wide PV data and site metadata."""
    frame, manifest = prepare_slovenian_dataset(
        project_root=project_root,
        metadata_path=metadata,
        output_path=output,
        pv_path=pv_path,
        timestamp_timezone=timestamp_timezone,
        value_kind=value_kind,
    )
    typer.echo(f"Wrote {len(frame)} rows for {manifest['site_count']} sites to {output}")


@app.command()
def pretrain_slovenian(
    dataset: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path = typer.Option(Path("models/tft_slovenia")),
    config: Path | None = typer.Option(None, exists=True, dir_okay=False),
    resume_checkpoint: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Resume interrupted pretraining from a Lightning checkpoint.",
    ),
) -> None:
    """Pretrain the global eight-horizon TFT on Slovenian sites."""
    checkpoint, metadata = pretrain_slovenian_tft(
        dataset,
        output,
        _tft_config(config),
        resume_checkpoint=resume_checkpoint,
    )
    typer.echo(f"Best checkpoint: {checkpoint}")
    typer.echo(f"Best validation loss: {metadata['best_validation_loss']}")


@app.command()
def fine_tune_finnish(
    dataset: Path = typer.Argument(..., exists=True, dir_okay=False),
    pretrained_checkpoint: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(Path("outputs/transfer_evaluation")),
    model_root: Path = typer.Option(Path("models/tft_finnish_folds")),
    config: Path | None = typer.Option(None, exists=True, dir_okay=False),
    enforce_gate: bool = typer.Option(True, "--enforce-gate/--no-enforce-gate"),
    resume: bool = typer.Option(
        False,
        "--resume/--no-resume",
        help="Resume completed folds and interrupted fold checkpoints in the output directory.",
    ),
) -> None:
    """Fine-tune every Finnish fold and run rolling out-of-fold evaluation."""
    _, _, pooled, epochs = run_finnish_rolling_evaluation(
        dataset,
        pretrained_checkpoint,
        output,
        model_root,
        _tft_config(config),
        enforce_gate=enforce_gate,
        resume=resume,
    )
    _print_acceptance(pooled)
    typer.echo(f"Fold best epochs: {epochs}")


@app.command()
def fit_final_finnish(
    dataset: Path = typer.Argument(..., exists=True, dir_okay=False),
    pretrained_checkpoint: Path = typer.Option(..., exists=True, dir_okay=False),
    fold_epochs: str = typer.Option(..., help="Comma-separated best epoch counts."),
    output: Path = typer.Option(Path("models/tft_finnish_final")),
    config: Path | None = typer.Option(None, exists=True, dir_okay=False),
) -> None:
    """Fit the final Finnish checkpoint for the median validated epoch count."""
    epochs = [int(value.strip()) for value in fold_epochs.split(",") if value.strip()]
    metadata = fit_final_finnish_checkpoint(
        dataset,
        pretrained_checkpoint,
        output,
        _tft_config(config),
        epochs,
    )
    typer.echo(f"Final model fitted for {metadata['fixed_epoch_count']} epochs in {output}")


@app.command("run-transfer-experiment")
def run_transfer_experiment_command(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Run preprocessing-independent pretraining, transfer evaluation, and final fit."""
    summary = run_transfer_experiment(config)
    _print_acceptance(summary["pooled_metrics"])


@app.command("report-weather-ablation")
def report_weather_ablation_command(
    baseline_output: Path = typer.Argument(..., exists=True, file_okay=False),
    weather_output: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Option(Path("outputs/finnish_weather_ablation")),
) -> None:
    """Report valid-weather Finnish ablation against an existing satellite run."""
    summary = report_finnish_weather_ablation(baseline_output, weather_output, output)
    for model, values in summary["pooled"].items():
        typer.echo(f"{model}: MAE={values['mae']:.6f}, MSE={values['mse']:.6f}")
    availability = summary["availability"]
    typer.echo(
        "Valid weather rows: "
        f"{availability['rows_valid']}/{availability['rows_total']} "
        f"({availability['issue_timestamp_available_pct']:.2f}% issue metadata)"
    )


@app.command("tft-loss-sweep")
def tft_loss_sweep_command(
    dataset: Path = typer.Argument(..., exists=True, dir_okay=False),
    pretrained_checkpoint: Path = typer.Option(..., exists=True, dir_okay=False),
    current_output: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(Path("outputs/tft_loss_sweep")),
    model_root: Path = typer.Option(Path("models/tft_loss_sweep")),
    config: Path | None = typer.Option(None, exists=True, dir_okay=False),
    profiles: str = typer.Option(
        "mse-heavy,balanced-huber-mse,mae-heavy",
        help="Comma-separated loss profiles; use pure-mae for exactly 1.00 x MAE.",
    ),
) -> None:
    """Compare three point-loss TFT fine-tunes with the current checkpoint."""
    values = json.loads(config.read_text(encoding="utf-8")) if config else {}
    experiment_values = values if "paths" in values else None
    tft_config = _tft_config(config)
    comparison = run_tft_loss_sweep(
        dataset_path=dataset,
        pretrained_checkpoint=pretrained_checkpoint,
        current_output=current_output,
        output_root=output,
        model_root=model_root,
        config=tft_config,
        bootstrap_replicates=int(experiment_values.get("bootstrap_replicates", 100))
        if experiment_values
        else 100,
        bootstrap_block_length=int(experiment_values.get("bootstrap_block_length", 15))
        if experiment_values
        else 15,
        skill_threshold=float(experiment_values.get("skill_threshold", 0.10))
        if experiment_values
        else 0.10,
        profiles=tuple(profile.strip() for profile in profiles.split(",") if profile.strip()),
    )
    columns = ["profile", "mae", "mse", "mae_delta_vs_current", "mse_delta_vs_current"]
    typer.echo(comparison[columns].to_string(index=False))
    typer.echo(f"Comparison: {output / 'loss_sweep_comparison.csv'}")


@app.command("reevaluate-transfer")
def reevaluate_transfer_command(
    source_output: Path = typer.Argument(..., exists=True),
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Correct a saved v1 evaluation without retraining either TFT model."""
    summary = reevaluate_transfer_predictions(source_output, config, output)
    _print_acceptance(summary["pooled_metrics"])
    typer.echo(f"Re-evaluated predictions: {output / 'predictions_long.csv'}")


@app.command()
def prepare() -> None:
    """Prepare raw pilot data into model-ready datasets."""
    typer.echo("Not implemented yet: define canonical input schema first.")
    raise typer.Exit(1)


@app.command()
def train() -> None:
    """Train forecasting models."""
    typer.echo("Not implemented yet: add baseline model before TFT.")
    raise typer.Exit(1)


@app.command()
def compare_tft_plan(
    path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_root: Path = typer.Option(Path("outputs/tft_plan_comparison")),
    model_root: Path = typer.Option(Path("models/tft_hirvensalmi")),
    max_epochs: int = typer.Option(30, min=1),
    batch_size: int = typer.Option(4096, min=1),
    checkpoint_path: Path | None = typer.Option(None, exists=True, dir_okay=False),
) -> None:
    """Train TFT and compare it against the workbook forecast plan."""
    warnings.warn(
        "compare-tft-plan is a legacy one-step compatibility workflow. Its checkpoints "
        "are incompatible with tft-transfer-v1; use prepare-finnish and fine-tune-finnish "
        "for corrected eight-horizon evaluation.",
        stacklevel=2,
    )
    outputs = run_tft_plan_comparison(
        ComparisonConfig(
            workbook_path=path,
            output_root=output_root,
            model_root=model_root,
            max_epochs=max_epochs,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path,
        )
    )
    typer.echo(f"Report: {outputs.report_path}")
    typer.echo(f"Metrics: {outputs.metrics_path}")
    typer.echo(f"Predictions: {outputs.predictions_path}")


@app.command()
def optimize() -> None:
    """Generate BESS schedules from forecasts."""
    typer.echo("Not implemented yet: define BESS constraints.")
    raise typer.Exit(1)


def _tft_config(path: Path | None) -> TFTTransferConfig:
    if path is None:
        return TFTTransferConfig()
    document = json.loads(path.read_text(encoding="utf-8"))
    values = dict(document.get("tft", document))
    satellite = document.get("satellite", {})
    if satellite.get("enabled", False):
        values.update(
            {
                "satellite_enabled": True,
                "satellite_encoder_only": bool(satellite.get("encoder_only", True)),
                "satellite_embedding_dim": int(satellite.get("embedding_dim", 32)),
                "satellite_missing_value": float(satellite.get("missing_value", 0.0)),
                "satellite_max_alignment_minutes": float(
                    satellite.get("max_alignment_minutes", 7.5)
                ),
            }
        )
    return TFTTransferConfig(**values)


def _print_acceptance(pooled: dict[str, object]) -> None:
    acceptance = pooled["acceptance"]
    if not isinstance(acceptance, dict):
        raise ValueError("Pooled metrics do not contain valid acceptance results.")
    threshold = float(acceptance["skill_threshold"])
    typer.echo(f"Pooled daylight point-head MSE skill: {float(pooled['skill']):.6f}")
    typer.echo(
        f"95% interval: [{float(pooled['skill_ci_low']):.6f}, {float(pooled['skill_ci_high']):.6f}]"
    )
    typer.echo(
        f"Point estimate meets {threshold:.0%} threshold: "
        f"{bool(acceptance['point_estimate_meets_threshold'])}"
    )
    typer.echo(
        f"CI lower bound meets {threshold:.0%} threshold: "
        f"{bool(acceptance['ci_lower_bound_meets_threshold'])}"
    )


if __name__ == "__main__":
    app()
