"""Turning a series of periodic returns into the numbers people actually
read in a report: a total return, an annualized return, a month-by-month
table, and a quarter-by-quarter table.

Every function here accepts a :class:`pandas.Series` indexed by date (a
``DatetimeIndex``, or anything :func:`pandas.to_datetime` can parse) and
containing **simple periodic returns as decimals** (e.g. a day that was up
1.5% is ``0.015``, not ``1.5``) unless the docstring says otherwise.
"""

from __future__ import annotations

import calendar
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "to_log_returns",
    "to_simple_returns",
    "total_return",
    "annualize_return",
    "cumulative_growth",
    "monthly_returns_table",
    "quarterly_returns_table",
]


def _as_dated_series(returns: pd.Series, name: str = "returns") -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(returns)!r}")
    if returns.empty:
        raise ValueError(f"{name} is empty")
    out = returns.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out.astype(float)


def to_log_returns(simple_returns: pd.Series) -> pd.Series:
    """Convert simple returns (``0.01`` = +1%) to log returns.

    Log returns are additive across time, which is why the rest of this
    module compounds through logs internally (``sum`` then
    ``exp(...) - 1``) rather than chaining ``(1 + r)`` multiplications -
    the two are mathematically equivalent, logs are just numerically
    friendlier over long histories.
    """
    series = _as_dated_series(simple_returns, "simple_returns")
    return np.log1p(series)


def to_simple_returns(log_returns: pd.Series) -> pd.Series:
    """Convert log returns back to simple returns (``0.01`` = +1%)."""
    series = _as_dated_series(log_returns, "log_returns")
    return np.expm1(series)


def total_return(returns: pd.Series, log: bool = False) -> float:
    """Compound a series of periodic returns into one total return.

    Parameters
    ----------
    returns : pandas.Series
        Periodic returns as decimals.
    log : bool, default False
        Set to ``True`` if ``returns`` are already log returns rather than
        simple returns.

    Returns
    -------
    float
        The total return over the whole series, as a decimal
        (``0.10`` = +10%).
    """
    series = _as_dated_series(returns)
    if log:
        return float(np.expm1(series.sum()))
    return float(np.expm1(np.log1p(series).sum()))


def annualize_return(
    total_return_value: float,
    num_periods: int,
    periods_per_year: int = 252,
    min_periods_for_annualization: Optional[int] = None,
) -> float:
    """Convert a total return over ``num_periods`` into an annualized rate.

    Parameters
    ----------
    total_return_value : float
        The compounded return over the whole window (decimal, e.g. ``0.10``).
    num_periods : int
        How many return observations (e.g. trading days) made up that window.
    periods_per_year : int, default 252
        How many observations there are in a full year for this data's
        frequency (252 for daily-trading-day data is the common industry
        default; use 12 for monthly data, 365/366 for calendar-day data,
        etc).
    min_periods_for_annualization : int, optional
        If the window is shorter than this many periods, the return is
        returned unannualized instead of being blown up by a fractional
        exponent. Many desks avoid annualizing anything under a year of
        history because the extrapolation is misleading over short
        windows; pass e.g. ``periods_per_year`` to enforce a "need at
        least a year of data" rule. Leave as ``None`` to always annualize.

    Returns
    -------
    float
        The annualized return as a decimal.
    """
    if num_periods <= 0:
        raise ValueError("num_periods must be positive")
    if (
        min_periods_for_annualization is not None
        and num_periods < min_periods_for_annualization
    ):
        return total_return_value
    return float((1.0 + total_return_value) ** (periods_per_year / num_periods) - 1.0)


def cumulative_growth(returns: pd.Series, log: bool = False) -> pd.Series:
    """Running cumulative return curve - "growth of $1 (as a %)" over time.

    Parameters
    ----------
    returns : pandas.Series
        Periodic returns as decimals, indexed by date.
    log : bool, default False
        Set to ``True`` if ``returns`` are log returns.

    Returns
    -------
    pandas.Series
        Same index as the input. Value at each date is the cumulative
        return from the start of the series through that date, as a
        decimal (``0.10`` = +10% grown so far). Multiply by 100 for a
        percentage.
    """
    series = _as_dated_series(returns)
    log_returns = series if log else np.log1p(series)
    return np.expm1(log_returns.cumsum())


def monthly_returns_table(returns: pd.Series, log: bool = False) -> pd.DataFrame:
    """Turn daily/periodic returns into a year-by-month return table.

    This is the classic "performance table" every fund tearsheet has:
    rows are years, columns are JAN..DEC plus a YTD column, values are
    percentages.

    Parameters
    ----------
    returns : pandas.Series
        Periodic returns as decimals, indexed by date.
    log : bool, default False
        Set to ``True`` if ``returns`` are log returns.

    Returns
    -------
    pandas.DataFrame
        Index = year (int). Columns = ``JAN``...``DEC``, ``YTD``. Values
        are percentages (already multiplied by 100), rounded to 2 dp.
        Months with no data are left as ``NaN``.
    """
    series = _as_dated_series(returns)
    log_returns = series if log else np.log1p(series)

    frame = log_returns.to_frame("log_return")
    frame["year"] = frame.index.year
    frame["month"] = frame.index.month

    monthly_log = frame.groupby(["year", "month"])["log_return"].sum()
    monthly_return = np.expm1(monthly_log)

    table = monthly_return.reset_index().pivot(
        index="year", columns="month", values="log_return"
    )
    # NOTE: pivot's "values" column keeps the original series name
    # ("log_return"), but the values themselves are already simple returns
    # at this point (we exponentiated above).
    month_labels = {m: calendar.month_abbr[m].upper() for m in range(1, 13)}
    table = table.rename(columns=month_labels)
    table = table.reindex(columns=[calendar.month_abbr[m].upper() for m in range(1, 13)])

    ytd_log = frame.groupby("year")["log_return"].sum()
    table["YTD"] = np.expm1(ytd_log)

    return (table * 100).round(2)


def quarterly_returns_table(returns: pd.Series, log: bool = False) -> pd.Series:
    """Compound periodic returns into a return-per-calendar-quarter series.

    Parameters
    ----------
    returns : pandas.Series
        Periodic returns as decimals, indexed by date.
    log : bool, default False
        Set to ``True`` if ``returns`` are log returns.

    Returns
    -------
    pandas.Series
        Indexed by labels like ``"2025-Q1"``, values are percentages
        (already multiplied by 100).
    """
    series = _as_dated_series(returns)
    log_returns = series if log else np.log1p(series)

    frame = log_returns.to_frame("log_return")
    frame["label"] = frame.index.year.astype(str) + "-Q" + frame.index.quarter.astype(str)

    quarterly_log = frame.groupby("label")["log_return"].sum()
    # Re-sort chronologically (groupby on a string label sorts lexically,
    # which happens to match chronological order for "YYYY-Qn" labels).
    quarterly_log = quarterly_log.sort_index()
    return (np.expm1(quarterly_log) * 100)
