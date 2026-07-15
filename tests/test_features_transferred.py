import pandas as pd

from aurora.features.solar import calculate_clear_sky_irradiance, calculate_solar_angles
from aurora.features.time import harmonic_features, infer_step_minutes


def test_harmonic_features_shape_and_step_inference() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")

    features = harmonic_features(index)

    assert features.shape == (4, 8)
    assert infer_step_minutes(index) == 15


def test_clear_sky_irradiance_positive_near_midday() -> None:
    timestamp = pd.Timestamp("2025-06-21 10:00:00", tz="UTC")

    irradiance = calculate_clear_sky_irradiance(timestamp, latitude=61.64, longitude=26.78)

    assert irradiance > 0


def test_solar_angles_return_expected_columns() -> None:
    index = pd.date_range("2025-06-21", periods=2, freq="1h", tz="UTC")

    angles = calculate_solar_angles(index, latitude=61.64, longitude=26.78)

    assert list(angles.columns) == ["solar_azimuth", "solar_elevation"]
    assert len(angles) == 2
