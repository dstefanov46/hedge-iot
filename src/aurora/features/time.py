import numpy as np
import pandas as pd

MINUTES_PER_DAY = 24 * 60
DAYS_PER_YEAR = 365.25
TWO_PI = 2 * np.pi


def harmonic_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate sine/cosine time features for daily and yearly cycles."""
    idx = pd.DatetimeIndex(index)
    day_minutes = idx.hour * 60 + idx.minute
    day_fraction = day_minutes / MINUTES_PER_DAY
    year_fraction = (idx.dayofyear.astype(float) + day_fraction) / DAYS_PER_YEAR

    values = np.column_stack(
        [
            np.sin(TWO_PI * year_fraction),
            np.sin(TWO_PI * 2 * year_fraction),
            np.cos(TWO_PI * year_fraction),
            np.cos(TWO_PI * 2 * year_fraction),
            np.sin(TWO_PI * day_fraction),
            np.sin(TWO_PI * 2 * day_fraction),
            np.cos(TWO_PI * day_fraction),
            np.cos(TWO_PI * 2 * day_fraction),
        ]
    )
    columns = [
        "year_sin_1",
        "year_sin_2",
        "year_cos_1",
        "year_cos_2",
        "day_sin_1",
        "day_sin_2",
        "day_cos_1",
        "day_cos_2",
    ]
    return pd.DataFrame(values, index=idx, columns=columns)


def infer_step_minutes(index: pd.DatetimeIndex) -> int:
    """Infer dominant time step in minutes from a datetime index."""
    idx = pd.DatetimeIndex(index)
    if len(idx) < 2:
        return 60

    if idx.freq is not None:
        try:
            return max(1, int(pd.Timedelta(idx.freq).total_seconds() // 60))
        except (TypeError, ValueError):
            pass

    try:
        inferred = pd.infer_freq(idx)
        if inferred is not None:
            return max(1, int(pd.Timedelta(inferred).total_seconds() // 60))
    except (TypeError, ValueError):
        pass

    diffs = np.diff(idx.view(np.int64))
    step_ns = int(pd.Series(diffs).mode().iloc[0])
    return max(1, int(pd.Timedelta(step_ns, unit="ns").total_seconds() // 60))


def infer_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Infer dominant time step as a pandas Timedelta."""
    return pd.Timedelta(minutes=infer_step_minutes(index))
