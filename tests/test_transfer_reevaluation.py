import json
from pathlib import Path

import pandas as pd
import pytest

from aurora.data.canonical import sha256_file
from aurora.experiments import transfer
from aurora.experiments.transfer import reevaluate_transfer_predictions


def test_reevaluation_replaces_baseline_preserves_predictions_and_records_provenance(
    tmp_path: Path, monkeypatch,
) -> None:
    timestamps = pd.date_range("2025-06-01 10:00", periods=3, freq="15min", tz="UTC")
    dataset = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "capacity_factor": [0.18, 0.3, 0.4],
            "clear_sky_norm": [0.4, 0.6, 0.8],
        }
    )
    dataset_path = tmp_path / "finnish.parquet"
    dataset.to_parquet(dataset_path, index=False)

    source = tmp_path / "v1"
    source.mkdir()
    source_predictions = source / "predictions_long.csv"
    pd.DataFrame(
        {
            "site_id": ["hirvensalmi", "hirvensalmi"],
            "issue_timestamp_utc": [timestamps[0], timestamps[0]],
            "target_timestamp_utc": timestamps[1:],
            "horizon_step": [1, 2],
            "actual_value": [0.3, 0.4],
            "solar_elevation": [20.0, 25.0],
            "point_forecast": [0.28, 0.38],
            "q50": [0.29, 0.39],
            "scratch_forecast": [0.25, 0.35],
            "smart_persistence": [0.0, 0.0],
            "existing_plan": [0.2, 0.3],
            "split": ["test", "test"],
            "fold": [0, 0],
        }
    ).to_csv(source_predictions, index=False)
    original_hash = sha256_file(source_predictions)

    config_path = tmp_path / "v2.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "transfer-experiment-v2",
                "paths": {
                    "slovenian_dataset": str(tmp_path / "unused.parquet"),
                    "finnish_dataset": str(dataset_path),
                    "output_root": str(tmp_path / "configured-output"),
                    "slovenian_checkpoint_root": str(tmp_path / "unused-source-model"),
                    "fold_model_root": str(tmp_path / "unused-fold-models"),
                    "final_model_root": str(tmp_path / "unused-final-model"),
                },
                "bootstrap_replicates": 2,
                "bootstrap_block_length": 1,
                "tft": {"horizon_steps": 2, "accelerator": "cpu"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        transfer,
        "fit_tft",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("training invoked")),
    )

    output = tmp_path / "v2"
    summary = reevaluate_transfer_predictions(source, config_path, output)
    corrected = pd.read_csv(output / "predictions_long.csv")

    assert corrected["legacy_harmonic_smart_persistence"].tolist() == [0.0, 0.0]
    assert corrected["smart_persistence"].tolist() == pytest.approx([0.27, 0.36])
    assert corrected["point_forecast"].tolist() == [0.28, 0.38]
    assert summary["reuses_model_predictions"] is True
    assert summary["source_predictions_sha256"] == original_hash
    assert sha256_file(source_predictions) == original_hash
    assert (output / "predictions_fold_0.csv").exists()
    assert (output / "metrics.csv").exists()
    assert (output / "pooled_metrics.json").exists()
    assert (output / "experiment_summary.json").exists()

    with pytest.raises(ValueError, match="distinct"):
        reevaluate_transfer_predictions(source, config_path, source)
    with pytest.raises(ValueError, match="not empty"):
        reevaluate_transfer_predictions(source, config_path, output)
