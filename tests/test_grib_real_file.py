from pathlib import Path

import pandas as pd
import pytest

from aurora.data.grib import read_grib_weather_archive


def test_one_real_slovenian_grib_variable_when_eccodes_is_available():
    """Smoke-test one archive file; CI without the native library skips clearly."""
    root = Path(r"F:\MAG\nwp_data")
    path = root / "NWP_data_t2m" / "t2m_2022-01-01to2022-01-31.grib"
    if not path.exists():
        pytest.skip("Slovenian GRIB archive is not mounted at F:\\MAG\\nwp_data")
    try:
        import cfgrib  # noqa: F401
        import eccodes  # noqa: F401
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"cfgrib/ecCodes unavailable: {exc}")
    sites = pd.DataFrame({"site_id": ["smoke"], "latitude": [46.05], "longitude": [14.5]})
    result, manifest = read_grib_weather_archive(
        # Start after the archive boundary so the smoke range has an eligible
        # issue; targets exactly at the first available issue are intentionally
        # missing under the strict issue < target rule.
        root, sites, pd.date_range("2022-01-01 06:15", periods=4, freq="15min", tz="UTC")
    )
    assert len(result) == 4
    assert result["weather_temperature_2m"].notna().any()
    assert result["weather_forecast_reference_utc"].notna().all()
    assert (result["weather_lead_hours"] >= 0).all()
    assert (result["timestamp_utc"].dt.minute % 15 == 0).all()
    assert result["weather_cloud_cover"].isna().all()
    assert "weather_low_cloud_cover" in result
    assert manifest["source_hashes"]
