from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from openpyxl import Workbook

from aurora.data.canonical import SiteSpec, canonicalize_site_frame, capacity_factor
from aurora.data.hirvensalmi import _local_to_utc, _timestamp_from_q1, parse_hirvensalmi_workbook
from aurora.data.slovenia import prepare_slovenian_dataset


def test_q1_date_is_preserved_before_timezone_conversion() -> None:
    result = _timestamp_from_q1(
        datetime(2025, 1, 1), "00.00.00-00.15.00", ZoneInfo("Europe/Helsinki")
    )
    assert result == datetime(2024, 12, 31, 22, tzinfo=UTC)


def test_dst_occurrences_map_to_distinct_utc_quarters() -> None:
    timezone = ZoneInfo("Europe/Helsinki")
    repeated = datetime(2025, 10, 26, 3)
    assert _local_to_utc(repeated, timezone, 0) == datetime(2025, 10, 26, 0, tzinfo=UTC)
    assert _local_to_utc(repeated, timezone, 1) == datetime(2025, 10, 26, 1, tzinfo=UTC)
    before = _local_to_utc(datetime(2025, 3, 30, 2, 45), timezone, 0)
    after = _local_to_utc(datetime(2025, 3, 30, 4), timezone, 0)
    assert after - before == pd.Timedelta(minutes=15)


def test_parser_rejects_malformed_tail_and_keeps_autumn_hour(tmp_path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Q4_25"
    sheet.append([None] * 8)
    sheet.append([None] * 8)
    sheet.append([None] * 8)
    sheet.append([datetime(2025, 10, 26, 3), None, None, None, None, 0, 0, None])
    sheet.append([datetime(2025, 10, 26, 3), None, None, None, None, 0, 0, None])
    sheet.append([time(0), None, None, None, None, 1, 1, None])
    path = tmp_path / "dst.xlsx"
    workbook.save(path)

    result = parse_hirvensalmi_workbook(path)

    assert len(result.records) == 2
    assert result.records[0].timestamp_utc != result.records[1].timestamp_utc
    assert result.exclusions[0]["reason"] == "malformed_timestamp"


def test_canonical_grid_masks_gaps_without_turning_them_into_zero() -> None:
    timestamps = pd.DatetimeIndex(
        ["2025-01-01T00:00Z", "2025-01-01T00:30Z"]
    )
    raw = pd.DataFrame(
        {"timestamp_utc": timestamps, "interval_energy_mwh": [0.0, 0.25]}
    )
    frame, manifest = canonicalize_site_frame(
        raw, SiteSpec("site", 1.0, 46.0, 14.0, "Europe/Ljubljana"), "fixture"
    )

    assert frame["time_idx"].diff().dropna().eq(1).all()
    assert frame["observation_available"].tolist() == [True, False, True]
    assert np.isnan(frame.loc[1, "capacity_factor"])
    assert manifest["gap_count"] == 1


def test_capacity_normalization_uses_interval_energy() -> None:
    result = capacity_factor(pd.Series([0.0, 0.5]), 2.0)
    assert result.tolist() == [0.0, 1.0]


def test_restored_wide_slovenian_adapter_fixture(tmp_path: Path) -> None:
    project = tmp_path / "sol-forecast"
    processed = project / "data" / "processed"
    processed.mkdir(parents=True)
    power = pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-01-01", periods=2, freq="15min"),
            "11": [0.0, 500.0],
            "12": [0.0, 250.0],
        }
    )
    power.to_csv(processed / "data.csv", index=False)
    metadata = pd.DataFrame(
        {
            "alt_id": [11, 12],
            "lat": [46.0, 46.5],
            "lon": [14.0, 14.5],
            "capacity_kw": [500, 250],
        }
    )
    metadata_path = project / "plants.csv"
    metadata.to_csv(metadata_path, index=False)

    frame, manifest = prepare_slovenian_dataset(
        project,
        metadata_path,
        pv_path=processed / "data.csv",
    )

    assert manifest["site_count"] == 2
    assert frame.groupby("site_id")["capacity_factor"].max().to_dict() == {
        "11": 1.0,
        "12": 1.0,
    }


def test_restored_metadata_index_and_pinst_column_are_supported(tmp_path: Path) -> None:
    project = tmp_path / "sol-forecast"
    processed = project / "data" / "processed"
    processed.mkdir(parents=True)
    timestamps = pd.date_range("2022-01-01", periods=2, freq="15min")
    pd.DataFrame({11: [0.0, 500.0]}, index=timestamps).to_pickle(processed / "data.p")
    metadata = pd.DataFrame(
        {"Pinst_kW": [500.0], "lat": [46.0], "lon": [14.0]},
        index=pd.Index([11], name="alt_id"),
    )
    metadata_path = project / "meta_clean.p"
    metadata.to_pickle(metadata_path)

    frame, manifest = prepare_slovenian_dataset(project, metadata_path)

    assert manifest["site_count"] == 1
    assert frame["site_id"].unique().tolist() == ["11"]
    assert frame["installed_capacity_mw"].unique().tolist() == [0.5]
    assert frame["capacity_factor"].tolist() == [0.0, 1.0]
