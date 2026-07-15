import pandas as pd


def enforce_frequency(
    frame: pd.DataFrame, timestamp_column: str, frequency: str = "15min"
) -> pd.DataFrame:
    """Return a time-indexed frame with a regular timestamp frequency."""
    if timestamp_column not in frame.columns:
        raise KeyError(f"Missing timestamp column: {timestamp_column}")

    indexed = frame.copy()
    indexed[timestamp_column] = pd.to_datetime(indexed[timestamp_column], utc=True)
    indexed = indexed.set_index(timestamp_column).sort_index()
    return indexed.asfreq(frequency)
