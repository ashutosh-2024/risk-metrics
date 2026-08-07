"""Standard reporting windows: MTD, trailing-N-months, and inception-to-date.

Every fund report slices history into the same handful of windows (this
month so far, trailing 12 months, trailing 36 months, since the fund
started). :func:`get_date_ranges` builds those windows once so every other
metric function in this package can be handed a consistent ``(start, end)``
pair instead of re-deriving the same calendar arithmetic everywhere.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Optional, Tuple

from dateutil.relativedelta import relativedelta

from ._dates import DateLike, to_date

__all__ = ["get_date_ranges", "month_end"]

from ._dates import month_end  # re-exported for convenience


def get_date_ranges(
    as_of: DateLike,
    inception_date: Optional[DateLike] = None,
    trailing_months: Optional[Dict[str, int]] = None,
) -> Dict[str, Tuple[dt.date, dt.date]]:
    """Build the standard set of reporting windows ending on ``as_of``.

    Parameters
    ----------
    as_of : str | datetime.date | datetime.datetime
        The "as of" / cut-off date for the report. Accepts
        ``"YYYY-MM-DD"``, ``"YYYYMMDD"``, or ``"YYYYMM"`` (the latter is
        treated as the last day of that month).
    inception_date : str | datetime.date | datetime.datetime, optional
        The date the fund/strategy/portfolio started. If provided, an
        ``"ITD"`` (Inception-To-Date) window is included.
    trailing_months : dict[str, int], optional
        Extra trailing windows to generate, as ``{label: number_of_months}``.
        Defaults to ``{"12M": 12, "36M": 36}``. Pass ``{}`` to skip trailing
        windows entirely and get only ``"MTD"`` (and ``"ITD"`` if
        ``inception_date`` is given).

    Returns
    -------
    dict[str, tuple[datetime.date, datetime.date]]
        A mapping of window label -> ``(start_date, end_date)``, both
        inclusive. Always contains ``"MTD"``.

    Example
    -------
    >>> get_date_ranges(as_of="2026-01-31", inception_date="2023-03-22")
    {'MTD': (datetime.date(2026, 1, 1), datetime.date(2026, 1, 31)),
     '12M': (datetime.date(2025, 2, 1), datetime.date(2026, 1, 31)),
     '36M': (datetime.date(2023, 2, 1), datetime.date(2026, 1, 31)),
     'ITD': (datetime.date(2023, 3, 22), datetime.date(2026, 1, 31))}
    """
    end_date = to_date(as_of)

    if trailing_months is None:
        trailing_months = {"12M": 12, "36M": 36}

    ranges: Dict[str, Tuple[dt.date, dt.date]] = {
        "MTD": (end_date.replace(day=1), end_date),
    }

    for label, months in trailing_months.items():
        if months <= 0:
            raise ValueError(f"trailing_months[{label!r}] must be positive, got {months}")
        start_date = end_date - relativedelta(months=months) + dt.timedelta(days=1)
        ranges[label] = (start_date, end_date)

    if inception_date is not None:
        ranges["ITD"] = (to_date(inception_date), end_date)

    return ranges
