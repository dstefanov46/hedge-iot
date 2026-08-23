import pandas as pd
import pytest

from aurora.data.grib import (
    deaccumulate,
    derive_relative_humidity,
    derive_wind,
    nearest_grid_point,
    select_latest_eligible_issue,
    valid_time_from_issue_step,
)


def test_issue_step_and_nearest_grid_point():
    assert valid_time_from_issue_step("2022-01-01T06:00Z", 3) == pd.Timestamp("2022-01-01T09:00Z")
    assert nearest_grid_point([45, 46, 47], [13, 14], 46.1, 13.9) == (1, 1)


def test_weather_derivations():
    assert derive_relative_humidity([20], [20])[0] == pytest.approx(100)
    speed, direction = derive_wind([0, 1], [-1, 0])
    assert speed.tolist() == pytest.approx([1, 1])
    assert direction.tolist() == pytest.approx([0, 270])


def test_deaccumulation_resets_per_issue():
    assert deaccumulate([0, 2, 5, 0, 3], ["a", "a", "a", "b", "b"]).tolist() == [0, 2, 3, 0, 3]


def test_latest_issue_rejects_issue_and_horizon():
    candidates = pd.DataFrame(
        {
            "issue_timestamp_utc": pd.to_datetime(["2022-01-01 00:00Z", "2022-01-01 06:00Z"]),
            "forecast_horizon_hours": [12, 12],
            "value": [1, 2],
        }
    )
    result = select_latest_eligible_issue(
        candidates, pd.to_datetime(["2022-01-01 06:00Z", "2022-01-01 07:00Z"])
    )
    assert result["value"].tolist() == [1, 2]
    assert (
        not select_latest_eligible_issue(
            candidates.iloc[[0]], pd.to_datetime(["2022-01-01 00:00Z"])
        )
        .any()
        .any()
    )
