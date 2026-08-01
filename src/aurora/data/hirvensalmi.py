import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.datetime import from_excel

from aurora.data.schema import ProductionObservation


@dataclass(frozen=True)
class SheetSummary:
    name: str
    rows: int
    columns: int


@dataclass(frozen=True)
class WorkbookSummary:
    path: Path
    sheets: tuple[SheetSummary, ...]


@dataclass(frozen=True)
class WorkbookParseResult:
    path: Path
    records: tuple[ProductionObservation, ...]
    exclusions: tuple[dict[str, object], ...] = ()


def load_hirvensalmi_workbook(path: Path) -> WorkbookSummary:
    """Load basic metadata from a Hirvensalmi production workbook."""
    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        sheets = tuple(
            SheetSummary(name=sheet.title, rows=sheet.max_row, columns=sheet.max_column)
            for sheet in workbook.worksheets
        )
    finally:
        workbook.close()

    return WorkbookSummary(path=path, sheets=sheets)


def parse_hirvensalmi_workbook(
    path: Path,
    site_id: str = "hirvensalmi",
    source_timezone: str = "Europe/Helsinki",
) -> WorkbookParseResult:
    """Parse Hirvensalmi quarterly sheets into canonical production observations.

    Q1 uses date serial + interval text. Q2-Q4 use timestamp cells that may be
    external workbook references; if cached values are absent, the parser falls
    back to a 15-minute quarter sequence and marks quality accordingly.
    """
    workbook = load_workbook(path, read_only=True, data_only=True)
    timezone = ZoneInfo(source_timezone)
    try:
        records: list[ProductionObservation] = []
        exclusions: list[dict[str, object]] = []
        for sheet in workbook.worksheets:
            if not re.fullmatch(r"Q[1-4]_\d{2}", sheet.title):
                continue
            sheet_records, sheet_exclusions = _parse_sheet(sheet, path.name, site_id, timezone)
            records.extend(sheet_records)
            exclusions.extend(sheet_exclusions)
    finally:
        workbook.close()

    return WorkbookParseResult(
        path=path,
        records=tuple(records),
        exclusions=tuple(exclusions),
    )


def hirvensalmi_records_to_frame(records: tuple[ProductionObservation, ...]) -> pd.DataFrame:
    """Convert parsed production observations into a model-ready DataFrame."""
    rows = [record.model_dump() for record in records]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["site_id", "timestamp_utc"]).reset_index(drop=True)


def _parse_sheet(
    sheet, source: str, site_id: str, timezone: ZoneInfo
) -> tuple[list[ProductionObservation], list[dict[str, object]]]:
    quarter, _ = _parse_quarter_sheet_name(sheet.title)
    start_row = 3 if quarter == 1 else 4
    records: list[ProductionObservation] = []
    exclusions: list[dict[str, object]] = []
    local_occurrences: defaultdict[datetime, int] = defaultdict(int)

    for source_row, row in enumerate(
        sheet.iter_rows(min_row=start_row, values_only=True), start=start_row
    ):
        if quarter == 1:
            local_timestamp = _local_timestamp_from_q1(row[0], row[1])
            planned = _as_float(row[2])
            actual = _as_float(row[3])
        else:
            local_timestamp = _as_local_datetime(row[0])
            planned = _as_float(row[5])
            actual = _as_float(row[6])

        if planned is None and actual is None:
            continue
        if local_timestamp is None:
            exclusions.append(
                {
                    "source_sheet": sheet.title,
                    "source_row": source_row,
                    "reason": "malformed_timestamp",
                    "raw_timestamp": repr(row[0]),
                }
            )
            continue

        occurrence = local_occurrences[local_timestamp]
        local_occurrences[local_timestamp] += 1
        timestamp = _local_to_utc(local_timestamp, timezone, occurrence)

        records.append(
            ProductionObservation(
                site_id=site_id,
                timestamp_utc=timestamp,
                interval_minutes=15,
                planned_mwh=planned,
                actual_mwh=actual,
                source=source,
                source_sheet=sheet.title,
                source_row=source_row,
                quality_flag="ok",
            )
        )

    return records, exclusions


def _parse_quarter_sheet_name(name: str) -> tuple[int, int]:
    match = re.fullmatch(r"Q([1-4])_(\d{2})", name)
    if not match:
        raise ValueError(f"Unsupported Hirvensalmi sheet name: {name}")
    return int(match.group(1)), 2000 + int(match.group(2))


def _timestamp_from_q1(date_value, interval_value, timezone: ZoneInfo) -> datetime | None:
    local_timestamp = _local_timestamp_from_q1(date_value, interval_value)
    if local_timestamp is None:
        return None
    return _local_to_utc(local_timestamp, timezone, occurrence=0)


def _local_timestamp_from_q1(date_value, interval_value) -> datetime | None:
    date_part = _excel_local_date(date_value)
    interval_start = _parse_interval_start(interval_value)
    if date_part is None or interval_start is None:
        return None
    return datetime.combine(date_part, interval_start)


def _excel_local_date(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        parsed = from_excel(value)
        return parsed.date() if isinstance(parsed, (date, datetime)) else None
    return None


def _parse_interval_start(value) -> time | None:
    if not isinstance(value, str):
        return None
    match = re.match(r"^(\d{2})\.(\d{2})\.\d{2}-", value)
    if not match:
        return None
    return time(hour=int(match.group(1)), minute=int(match.group(2)))


def _as_local_datetime(value) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, (int, float)):
        parsed = from_excel(value)
        return parsed.replace(tzinfo=None) if isinstance(parsed, datetime) else None
    return None


def _as_datetime_utc(value, timezone: ZoneInfo) -> datetime | None:
    """Backward-compatible scalar conversion used by older callers and tests."""
    local = _as_local_datetime(value)
    return None if local is None else _local_to_utc(local, timezone, occurrence=0)


def _local_to_utc(local: datetime, timezone: ZoneInfo, occurrence: int) -> datetime:
    """Convert a workbook wall-clock time, retaining both autumn DST occurrences."""
    first = local.replace(tzinfo=timezone, fold=0)
    second = local.replace(tzinfo=timezone, fold=1)
    ambiguous = first.utcoffset() != second.utcoffset()
    fold = 1 if ambiguous and occurrence % 2 == 1 else 0
    return local.replace(tzinfo=timezone, fold=fold).astimezone(UTC)


def _as_float(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
