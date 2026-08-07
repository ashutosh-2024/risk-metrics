"""Internal date-parsing helpers shared by the other modules.

Not part of the public API - end users should not need to import this
module directly.
"""

from __future__ import annotations

import calendar
import datetime as dt
from typing import Union

DateLike = Union[str, dt.date, dt.datetime]


def to_date(value: DateLike) -> dt.date:
    """Convert a string, ``date`` or ``datetime`` into a plain ``date``.

    Accepted string formats: ``"YYYY-MM-DD"``, ``"YYYYMMDD"`` and
    ``"YYYYMM"`` (the latter is interpreted as the *last* calendar day of
    that month, since monthly reports are usually anchored to month-end).
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 6 and text.isdigit():
            year, month = int(text[:4]), int(text[4:6])
            return month_end(year, month)
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return dt.datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        raise ValueError(
            f"Could not parse date string {value!r}. "
            "Use 'YYYY-MM-DD', 'YYYYMMDD' or 'YYYYMM'."
        )
    raise TypeError(
        f"Expected a string, datetime.date or datetime.datetime, got {type(value)!r}"
    )


def month_end(year: int, month: int) -> dt.date:
    """Return the last calendar day of ``year``-``month``."""
    last_day = calendar.monthrange(year, month)[1]
    return dt.date(year, month, last_day)
