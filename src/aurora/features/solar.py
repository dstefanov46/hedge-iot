from typing import overload

import numpy as np
import pandas as pd


@overload
def calculate_solar_position(
    timestamp: pd.Timestamp,
    latitude: float,
    longitude: float,
    elevation_m: float = 0.0,
) -> tuple[float, float]: ...


@overload
def calculate_solar_position(
    timestamp: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    elevation_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]: ...


def calculate_solar_position(
    timestamp: pd.Timestamp | pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    elevation_m: float = 0.0,
) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """Calculate approximate solar azimuth and elevation angles in degrees."""
    _ = elevation_m
    scalar_input = isinstance(timestamp, pd.Timestamp)
    timestamps = pd.DatetimeIndex([timestamp]) if scalar_input else pd.DatetimeIndex(timestamp)
    timestamps = (
        timestamps.tz_localize("UTC") if timestamps.tz is None else timestamps.tz_convert("UTC")
    )

    year = timestamps.year.values
    month = timestamps.month.values
    day = timestamps.day.values
    hour = timestamps.hour.values
    minute = timestamps.minute.values
    second = timestamps.second.values

    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    julian_day = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    julian_day = julian_day + (hour - 12) / 24.0 + minute / 1440.0 + second / 86400.0
    julian_century = (julian_day - 2451545.0) / 36525.0

    mean_longitude = (
        280.46646 + 36000.76983 * julian_century + 0.0003032 * julian_century**2
    ) % 360
    mean_anomaly = (357.52911 + 35999.05029 * julian_century - 0.0001537 * julian_century**2) % 360
    mean_anomaly_rad = np.radians(mean_anomaly)

    equation_center = (
        1.914602 - 0.004817 * julian_century - 0.000014 * julian_century**2
    ) * np.sin(mean_anomaly_rad)
    equation_center += (0.019993 - 0.000101 * julian_century) * np.sin(2 * mean_anomaly_rad)
    equation_center += 0.000289 * np.sin(3 * mean_anomaly_rad)

    true_longitude = (mean_longitude + equation_center) % 360
    true_longitude_rad = np.radians(true_longitude)
    obliquity = (
        23.439291
        - 0.0130042 * julian_century
        - 0.00000016 * julian_century**2
        + 0.000000504 * julian_century**3
    )
    obliquity_rad = np.radians(obliquity)

    right_ascension = (
        np.degrees(
            np.arctan2(
                np.cos(obliquity_rad) * np.sin(true_longitude_rad), np.cos(true_longitude_rad)
            )
        )
        % 360
    )
    declination = np.degrees(np.arcsin(np.sin(obliquity_rad) * np.sin(true_longitude_rad)))

    gmst = (
        280.46061837
        + 360.98564736629 * (julian_day - 2451545.0)
        + 0.000387933 * julian_century**2
        - julian_century**3 / 38710000.0
    ) % 360
    local_hour_angle = (gmst + longitude - right_ascension) % 360
    local_hour_angle_rad = np.radians(local_hour_angle)

    lat_rad = np.radians(latitude)
    dec_rad = np.radians(declination)
    sin_elevation = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(
        local_hour_angle_rad
    )
    elevation = np.degrees(np.arcsin(np.clip(sin_elevation, -1, 1)))

    cos_azimuth = (np.sin(dec_rad) - np.sin(lat_rad) * sin_elevation) / (
        np.cos(lat_rad) * np.cos(np.radians(elevation))
    )
    azimuth = np.degrees(np.arccos(np.clip(cos_azimuth, -1, 1)))
    azimuth = np.where(local_hour_angle > 180, 360 - azimuth, azimuth)

    refraction = np.zeros_like(elevation)
    mask = elevation > -1
    refraction[mask] = 0.0167 / np.tan(
        np.radians(elevation[mask] + 10.3 / (elevation[mask] + 5.11))
    )
    elevation_corrected = elevation + refraction

    if scalar_input:
        return float(azimuth[0]), float(elevation_corrected[0])
    return azimuth, elevation_corrected


def calculate_solar_angles(
    index: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    elevation_m: float = 0.0,
) -> pd.DataFrame:
    """Calculate solar azimuth/elevation features for timestamps."""
    azimuth, elevation = calculate_solar_position(index, latitude, longitude, elevation_m)
    return pd.DataFrame({"solar_azimuth": azimuth, "solar_elevation": elevation}, index=index)


def calculate_clear_sky_irradiance(
    timestamp: pd.Timestamp | pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    elevation_m: float = 0.0,
    solar_constant: float = 1361.0,
) -> float | np.ndarray:
    """Calculate simplified clear-sky GHI estimate in W/m2."""
    _, elevation = calculate_solar_position(timestamp, latitude, longitude, elevation_m)
    scalar_input = isinstance(elevation, float)
    elevation_values = np.array([elevation]) if scalar_input else elevation
    ghi = np.zeros_like(elevation_values, dtype=float)

    mask = elevation_values > 0
    zenith = 90 - elevation_values[mask]
    air_mass = 1 / (np.cos(np.radians(zenith)) + 0.50572 * (96.07995 - zenith) ** (-1.6364))
    tau = 0.7 ** (air_mass**0.678)
    ghi[mask] = solar_constant * np.sin(np.radians(elevation_values[mask])) * tau

    if scalar_input:
        return float(ghi[0])
    return ghi


def add_lagged_features(frame: pd.DataFrame, columns: list[str], lag_steps: int) -> pd.DataFrame:
    """Append lagged copies of selected columns."""
    pieces = [frame]
    for column in columns:
        for lag in range(1, lag_steps + 1):
            pieces.append(frame[column].shift(lag).rename(f"{column}_lag_{lag}"))
    return pd.concat(pieces, axis=1)
