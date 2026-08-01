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
    run_finnish_rolling_evaluation,
    run_transfer_experiment,
)
from aurora.forecasting.tft_transfer import TFTTransferConfig

app = typer.Typer(help="AURORA forecasting and optimization workflows.")


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
) -> None:
    """Pretrain the global eight-horizon TFT on Slovenian sites."""
    checkpoint, metadata = pretrain_slovenian_tft(dataset, output, _tft_config(config))
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
) -> None:
    """Fine-tune every Finnish fold and run rolling out-of-fold evaluation."""
    _, _, pooled, epochs = run_finnish_rolling_evaluation(
        dataset,
        pretrained_checkpoint,
        output,
        model_root,
        _tft_config(config),
        enforce_gate=enforce_gate,
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
    values = json.loads(path.read_text(encoding="utf-8"))
    if "tft" in values:
        values = values["tft"]
    return TFTTransferConfig(**values)


def _print_acceptance(pooled: dict[str, object]) -> None:
    acceptance = pooled["acceptance"]
    if not isinstance(acceptance, dict):
        raise ValueError("Pooled metrics do not contain valid acceptance results.")
    threshold = float(acceptance["skill_threshold"])
    typer.echo(f"Pooled daylight point-head MSE skill: {float(pooled['skill']):.6f}")
    typer.echo(
        "95% interval: "
        f"[{float(pooled['skill_ci_low']):.6f}, {float(pooled['skill_ci_high']):.6f}]"
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
