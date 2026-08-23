import json

import numpy as np
import pandas as pd
import pytest

from aurora.data.openmeteo import (
    OpenMeteoClient,
    OpenMeteoRequest,
    attach_weather,
    fetch_single_run_weather_archive,
    parse_openmeteo_response,
    resample_weather_to_grid,
    validate_weather_forecast_windows,
)


def _payload(issue=None):
    value = {
        "hourly": {
            "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
            "temperature_2m": [1, 2],
            "cloud_cover": [10, None],
        },
        "model": "ecmwf_ifs025",
    }
    if issue:
        value["forecast_reference_utc"] = issue
    return value


def test_parse_utc_and_missing_values_are_explicit():
    result = parse_openmeteo_response(_payload(), site_id="fi")
    assert str(result.loc[0, "timestamp_utc"]) == "2025-01-01 00:00:00+00:00"
    assert np.isnan(result.loc[1, "weather_cloud_cover"])
    assert result.loc[0, "weather_issue_timestamp_available"] == 0
    assert pd.isna(result.loc[0, "weather_forecast_reference_utc"])


def test_cache_key_prevents_second_download(tmp_path):
    calls = []
    client = OpenMeteoClient(tmp_path, transport=lambda url: (calls.append(url), _payload())[1])
    request = OpenMeteoRequest(61.64, 26.78, "2025-01-01", "2025-01-01")
    client.fetch(request)
    client.fetch(request)
    assert len(calls) == 1
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["model"] == "ecmwf_ifs025"


def test_single_run_request_uses_run_and_forecast_days_without_date_range():
    request = OpenMeteoRequest(
        61.64,
        26.78,
        "2025-01-01",
        "2025-01-10",
        mode="single_run",
        run_timestamp_utc=pd.Timestamp("2025-01-01T00:00Z"),
        forecast_days=10,
    )
    params = request.params()
    assert params["run"] == "2025-01-01T00:00"
    assert params["forecast_days"] == "10"
    assert params["models"] == "ecmwf_ifs"
    assert "start_date" not in params
    assert "end_date" not in params


def test_batch_coordinates_and_quarter_hour_resampling(tmp_path):
    calls = []
    client = OpenMeteoClient(
        tmp_path, transport=lambda url: (calls.append(url), [_payload(), _payload()])[1]
    )
    requests = [
        OpenMeteoRequest(61, 26, "2025-01-01", "2025-01-01"),
        OpenMeteoRequest(46, 14, "2025-01-01", "2025-01-01"),
    ]
    assert len(client.fetch_many(requests)) == 2
    assert len(calls) == 1 and "latitude=61.000000%2C46.000000" in calls[0]
    parsed = parse_openmeteo_response(_payload("2024-12-31T23:00Z"), site_id="fi")
    grid = pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC")
    aligned = resample_weather_to_grid(parsed, grid)
    assert aligned["timestamp_utc"].tolist() == list(grid)
    assert aligned.loc[2, "weather_temperature_2m"] == pytest.approx(1.5)


def test_attach_and_leakage_validation():
    grid = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    frame = pd.DataFrame({"site_id": "fi", "timestamp_utc": grid})
    weather = parse_openmeteo_response(_payload("2024-12-31T23:00Z"), site_id="fi")
    attached = attach_weather(frame, weather)
    assert attached["weather_forecast_reference_utc"].notna().all()
    validate_weather_forecast_windows(
        attached.rename(columns={"timestamp_utc": "target_timestamp_utc"})
    )
    bad = attached.rename(columns={"timestamp_utc": "target_timestamp_utc"}).copy()
    bad.loc[0, "target_timestamp_utc"] = pd.Timestamp("2024-12-31 22:00", tz="UTC")
    with pytest.raises(ValueError, match="at or before"):
        validate_weather_forecast_windows(bad)


def test_single_run_archive_selects_latest_eligible_issue(tmp_path):
    calls = []

    def transport(url):
        from urllib.parse import parse_qs, urlparse

        query = parse_qs(urlparse(url).query)
        run = query["run"][0]
        calls.append(run)
        times = pd.date_range("2025-01-01", periods=97, freq="15min", tz="UTC")
        return {
            "hourly": {
                "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                "temperature_2m": [float(len(calls))] * len(times),
            },
            "model": "ecmwf_ifs025",
        }

    target = pd.date_range("2025-01-01", periods=97, freq="15min", tz="UTC")
    result = fetch_single_run_weather_archive(
        OpenMeteoClient(tmp_path, transport=transport),
        site_id="fi",
        latitude=61.64,
        longitude=26.78,
        target_timestamps=target,
        forecast_days=1,
    )
    assert len(calls) > 0
    assert result["weather_issue_timestamp_available"].mean() == 1.0
    noon = result.loc[result["timestamp_utc"] == pd.Timestamp("2025-01-01 12:00", tz="UTC")].iloc[0]
    assert noon["weather_forecast_reference_utc"] == pd.Timestamp("2025-01-01 06:00", tz="UTC")
    assert (result["weather_lead_hours"] >= 0).all()
