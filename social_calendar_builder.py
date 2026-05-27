"""Build default CN request social calendar (2025-2026) from public holiday schedules.

Sources: 国务院办公厅 2025/2026 节假日安排；大促区间见 idea/time_request_social_calendar.md §9.
Sparse rows only (non-default days); lookup misses -> holiday_type=0, promo_id=0.
"""
from __future__ import annotations

import csv
import os
from datetime import date, timedelta
from typing import Dict, Iterator, List


def _daterange(start: str, end: str) -> Iterator[str]:
    d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    while d <= end_d:
        yield d.isoformat()
        d += timedelta(days=1)


def _ensure(table: Dict[str, Dict], day: str) -> Dict:
    if day not in table:
        table[day] = {"date": day, "holiday_type": 0, "promo_id": 0}
    return table[day]


def _set_promo(table: Dict[str, Dict], start: str, end: str, promo_id: int) -> None:
    for day in _daterange(start, end):
        _ensure(table, day)["promo_id"] = promo_id


def _set_holiday(table: Dict[str, Dict], start: str, end: str, holiday_type: int) -> None:
    for day in _daterange(start, end):
        _ensure(table, day)["holiday_type"] = holiday_type


def _set_holiday_days(table: Dict[str, Dict], days: List[str], holiday_type: int) -> None:
    for day in days:
        _ensure(table, day)["holiday_type"] = holiday_type


def build_default_social_calendar_table() -> List[Dict]:
    """Return sorted sparse rows for 2025-01-01 .. 2026-12-31 special days."""
    t: Dict[str, Dict] = {}

    # --- Commercial promos (lower id wins on overlap: apply narrower promos first, 618 last) ---
    for year in (2025, 2026):
        _set_promo(t, f"{year}-01-10", f"{year}-02-12", 3)  # 年货节
        _ensure(t, f"{year}-02-14")["promo_id"] = 4  # 情人节
        _set_promo(t, f"{year}-10-20", f"{year}-11-15", 2)  # 双11 季
        _set_promo(t, f"{year}-05-20", f"{year}-06-30", 1)  # 618 季（含 520）

    # --- 2025 国务院安排 ---
    _set_holiday_days(t, ["2025-01-01"], 1)
    _set_holiday(t, "2025-01-28", "2025-02-04", 1)
    _set_holiday_days(t, ["2025-01-26", "2025-02-08"], 2)
    _set_holiday(t, "2025-04-04", "2025-04-06", 1)
    _set_holiday(t, "2025-05-01", "2025-05-05", 1)
    _set_holiday_days(t, ["2025-04-27"], 2)
    _set_holiday(t, "2025-05-31", "2025-06-02", 1)
    _set_holiday(t, "2025-10-01", "2025-10-08", 1)
    _set_holiday_days(t, ["2025-09-28", "2025-10-11"], 2)

    # --- 2026 国务院安排 ---
    _set_holiday(t, "2026-01-01", "2026-01-03", 1)
    _set_holiday_days(t, ["2026-01-04"], 2)
    _set_holiday(t, "2026-02-15", "2026-02-23", 1)
    _set_holiday_days(t, ["2026-02-14", "2026-02-28"], 2)
    _set_holiday(t, "2026-04-04", "2026-04-06", 1)
    _set_holiday(t, "2026-05-01", "2026-05-05", 1)
    _set_holiday_days(t, ["2026-05-09"], 2)
    _set_holiday(t, "2026-06-19", "2026-06-21", 1)
    _set_holiday(t, "2026-09-25", "2026-09-27", 1)
    _set_holiday(t, "2026-10-01", "2026-10-07", 1)
    _set_holiday_days(t, ["2026-09-20", "2026-10-10"], 2)

    rows = list(t.values())
    rows.sort(key=lambda r: r["date"])
    return rows


def write_default_calendar_csv(path: str) -> str:
    """Write builder output to CSV; return absolute path."""
    rows = build_default_social_calendar_table()
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "holiday_type", "promo_id"])
        writer.writeheader()
        writer.writerows(rows)
    return os.path.abspath(path)
