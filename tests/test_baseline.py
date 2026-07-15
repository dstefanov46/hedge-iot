import pandas as pd

from aurora.forecasting.baseline import PersistenceForecaster


def test_persistence_forecaster_repeats_latest_value() -> None:
    frame = pd.DataFrame({"actual_mwh": [1.0, 2.5, 3.0]})
    forecaster = PersistenceForecaster()

    forecaster.fit(frame, "actual_mwh")
    forecast = forecaster.predict(frame, horizon_steps=2)

    assert forecast["forecast"].tolist() == [3.0, 3.0]
