from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CVFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def train_index(self, full_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return full_index[(full_index >= self.train_start) & (full_index <= self.train_end)]

    def validation_index(self, full_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return full_index[(full_index >= self.val_start) & (full_index <= self.val_end)]

    def test_index(self, full_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return full_index[(full_index >= self.test_start) & (full_index <= self.test_end)]


def generate_rolling_cv_folds(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    train_months: int = 3,
    val_months: int = 1,
    test_months: int = 1,
) -> list[CVFold]:
    """Generate rolling month-based train/validation/test folds."""
    if min(train_months, val_months, test_months) <= 0:
        raise ValueError("train_months, val_months, and test_months must be positive.")

    start = pd.to_datetime(start_date).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = pd.to_datetime(end_date)
    if end == end.normalize():
        end = end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    folds: list[CVFold] = []
    fold_id = 0
    current_start = start
    while True:
        train_start = current_start
        train_end_exclusive = train_start + pd.DateOffset(months=train_months)
        val_start = train_end_exclusive
        val_end_exclusive = val_start + pd.DateOffset(months=val_months)
        test_start = val_end_exclusive
        test_end_exclusive = test_start + pd.DateOffset(months=test_months)

        train_end = train_end_exclusive - pd.Timedelta(microseconds=1)
        val_end = val_end_exclusive - pd.Timedelta(microseconds=1)
        test_end = test_end_exclusive - pd.Timedelta(microseconds=1)
        if test_end > end:
            break

        folds.append(
            CVFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_id += 1
        current_start = current_start + pd.DateOffset(months=1)

    return folds


def generate_expanding_finnish_folds(
    year: int = 2025, timezone: str = "UTC"
) -> list[CVFold]:
    """Jan-Mar/Apr/May through Jan-Oct/Nov/Dec, with an expanding train window."""
    folds: list[CVFold] = []
    def boundary(month: int) -> pd.Timestamp:
        local = pd.Timestamp(year=year, month=1, day=1, tz=timezone) + pd.DateOffset(
            months=month - 1
        )
        return local.tz_convert("UTC")

    for fold_id, validation_month in enumerate(range(4, 12)):
        train_start = boundary(1)
        val_start = boundary(validation_month)
        test_start = boundary(validation_month + 1)
        test_end_exclusive = boundary(validation_month + 2)
        folds.append(
            CVFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=val_start - pd.Timedelta(nanoseconds=1),
                val_start=val_start,
                val_end=test_start - pd.Timedelta(nanoseconds=1),
                test_start=test_start,
                test_end=test_end_exclusive - pd.Timedelta(nanoseconds=1),
            )
        )
    return folds
