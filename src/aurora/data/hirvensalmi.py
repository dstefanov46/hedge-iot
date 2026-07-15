import re
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
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
        for sheet in workbook.worksheets:
            if not re.fullmatch(r"Q[1-4]_\d{2}", sheet.title):
                continue
            records.extend(_parse_sheet(sheet, path.name, site_id, timezone))
    finally:
        workbook.close()

    return WorkbookParseResult(path=path, records=tuple(records))


def hirvensalmi_records_to_frame(records: tuple[ProductionObservation, ...]) -> pd.DataFrame:
    """Convert parsed production observations into a model-ready DataFrame."""
    rows = [record.model_dump() for record in records]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["site_id", "timestamp_utc"]).reset_index(drop=True)


def _parse_sheet(
    sheet, source: str, site_id: str, timezone: ZoneInfo
) -> list[ProductionObservation]:
    quarter, year = _parse_quarter_sheet_name(sheet.title)
    start_row = 3 if quarter == 1 else 4
    fallback_start = datetime(year, (quarter - 1) * 3 + 1, 1, tzinfo=timezone)
    records: list[ProductionObservation] = []

    for offset, row in enumerate(sheet.iter_rows(min_row=start_row, values_only=True)):
        if quarter == 1:
            timestamp = _timestamp_from_q1(row[0], row[1], timezone)
            planned = _as_float(row[2])
            actual = _as_float(row[3])
        else:
            timestamp = _as_datetime_utc(row[0], timezone)
            planned = _as_float(row[5])
            actual = _as_float(row[6])

        quality_flag = "ok"
        if timestamp is None:
            timestamp = (fallback_start + timedelta(minutes=15 * offset)).astimezone(UTC)
            quality_flag = "linked_workbook_unresolved" if quarter != 1 else "estimated"

        if planned is None and actual is None:
            continue

        records.append(
            ProductionObservation(
                site_id=site_id,
                timestamp_utc=timestamp,
                interval_minutes=15,
                planned_mwh=planned,
                actual_mwh=actual,
                source=source,
                source_sheet=sheet.title,
                quality_flag=quality_flag,
            )
        )

    return records


def _parse_quarter_sheet_name(name: str) -> tuple[int, int]:
    match = re.fullmatch(r"Q([1-4])_(\d{2})", name)
    if not match:
        raise ValueError(f"Unsupported Hirvensalmi sheet name: {name}")
    return int(match.group(1)), 2000 + int(match.group(2))


def _timestamp_from_q1(date_value, interval_value, timezone: ZoneInfo) -> datetime | None:
    date_part = _excel_date_to_datetime(date_value, timezone)
    if date_part is None:
        return None

    interval_start = _parse_interval_start(interval_value) or time(0, 0)
    local_timestamp = datetime.combine(date_part.date(), interval_start, tzinfo=timezone)
    return local_timestamp.astimezone(UTC)


def _excel_date_to_datetime(value, timezone: ZoneInfo) -> datetime | None:
    if isinstance(value, datetime):
        local_value = value.replace(tzinfo=value.tzinfo or timezone)
        return local_value.astimezone(UTC)
    if isinstance(value, (int, float)):
        parsed = from_excel(value).replace(tzinfo=timezone)
        return parsed.astimezone(UTC)
    return None


def _parse_interval_start(value) -> time | None:
    if not isinstance(value, str):
        return None
    match = re.match(r"^(\d{2})\.(\d{2})\.\d{2}-", value)
    if not match:
        return None
    return time(hour=int(match.group(1)), minute=int(match.group(2)))


def _as_datetime_utc(value, timezone: ZoneInfo) -> datetime | None:
    if isinstance(value, datetime):
        local_value = value.replace(tzinfo=value.tzinfo or timezone)
        return local_value.astimezone(UTC)
    if isinstance(value, (int, float)):
        return from_excel(value).replace(tzinfo=timezone).astimezone(UTC)
    return None


def _as_float(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
