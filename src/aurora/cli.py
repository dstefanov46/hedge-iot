from pathlib import Path

import typer

from aurora.data.hirvensalmi import (
    hirvensalmi_records_to_frame,
    load_hirvensalmi_workbook,
    parse_hirvensalmi_workbook,
)
from aurora.experiments.tft_plan_comparison import ComparisonConfig, run_tft_plan_comparison

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


if __name__ == "__main__":
    app()
