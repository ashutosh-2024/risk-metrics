"""Long/short exposure, position concentration, and exposure-weighted returns.

"Exposure" throughout this module means a signed dollar (or any single
currency) amount: positive for a long position, negative for a short one.
Nothing here is specific to equities, options, or any particular asset
class - it works the same for a book of stocks, a book of futures, or a
book of FX positions, as long as you can express each position's size as
a signed number in a consistent currency.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = [
    "long_short_gross_net",
    "position_counts",
    "daily_long_short_imbalance",
    "weighted_portfolio_return",
]


def long_short_gross_net(
    exposure: pd.Series, nav: Optional[float] = None
) -> Dict[str, float]:
    """Long, short, gross and net exposure from a snapshot of positions.

    Parameters
    ----------
    exposure : pandas.Series
        One row per position, values are signed exposure amounts
        (positive = long, negative = short) in a consistent currency
        (e.g. USD). This is a single snapshot, not a time series - if you
        have exposures over many dates, group/filter down to one date (or
        an average) before calling this.
    nav : float, optional
        Net asset value (or gross AUM) to express results as percentages
        of. If omitted, only dollar amounts are returned.

    Returns
    -------
    dict[str, float]
        ``"long"``, ``"short"``, ``"gross"`` (long+short), ``"net"``
        (long-short), all in the same currency as the input. If ``nav``
        is given, also includes ``"long_pct"``, ``"short_pct"``,
        ``"gross_pct"``, ``"net_pct"`` (each divided by ``nav`` and
        multiplied by 100).
    """
    if not isinstance(exposure, pd.Series):
        raise TypeError(f"exposure must be a pandas Series, got {type(exposure)!r}")

    values = exposure.astype(float)
    long_amount = float(values[values >= 0].sum())
    short_amount = float(values[values < 0].abs().sum())
    result = {
        "long": long_amount,
        "short": short_amount,
        "gross": long_amount + short_amount,
        "net": long_amount - short_amount,
    }
    if nav is not None:
        if nav == 0:
            raise ValueError("nav must not be zero")
        for key in ("long", "short", "gross", "net"):
            result[f"{key}_pct"] = result[key] / nav * 100
    return result


def position_counts(exposure: pd.Series) -> Dict[str, int]:
    """Count how many positions in a snapshot are long vs. short.

    Parameters
    ----------
    exposure : pandas.Series
        One row per position, signed exposure amounts (see
        :func:`long_short_gross_net`).

    Returns
    -------
    dict[str, int]
        ``{"long": count, "short": count}``.
    """
    if not isinstance(exposure, pd.Series):
        raise TypeError(f"exposure must be a pandas Series, got {type(exposure)!r}")
    values = exposure.astype(float)
    return {
        "long": int((values >= 0).sum()),
        "short": int((values < 0).sum()),
    }


def daily_long_short_imbalance(
    exposure: pd.DataFrame,
    nav: float,
    date_col: str = "date",
    exposure_col: str = "exposure",
    group_col: Optional[str] = None,
):
    """Average day-by-day (long - short) imbalance, as a % of NAV.

    A portfolio that is perfectly long/short balanced within a sector (or
    overall) has an imbalance of 0%; a portfolio that is net long has a
    positive imbalance. Averaging this across every day in a window (a
    month, a year...) tells you the *typical* skew, which is more
    representative than looking at a single day's snapshot.

    Parameters
    ----------
    exposure : pandas.DataFrame
        One row per position per date, with a signed exposure column
        (``exposure_col``) and a date column (``date_col``). Pre-filter
        this to the date window and universe (e.g. one strategy) you care
        about before calling this function.
    nav : float
        Net asset value to divide by. Pass a single representative value
        (e.g. start-of-period NAV) - this function does not look up a
        different NAV per day.
    date_col, exposure_col : str
        Column names in ``exposure``.
    group_col : str, optional
        If given (e.g. a sector or region column), the imbalance is
        computed and averaged separately per group, and the result is a
        :class:`pandas.Series` indexed by group, sorted by descending
        imbalance. If omitted, a single float is returned for the whole
        book.

    Returns
    -------
    float or pandas.Series
        Average daily imbalance as a percentage of ``nav``.
    """
    required = {date_col, exposure_col} | ({group_col} if group_col else set())
    missing = required - set(exposure.columns)
    if missing:
        raise ValueError(f"exposure is missing required columns: {sorted(missing)}")
    if nav == 0:
        raise ValueError("nav must not be zero")

    df = exposure.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[exposure_col] = df[exposure_col].astype(float)

    group_cols = [date_col] + ([group_col] if group_col else [])
    daily = (
        df.groupby(group_cols)[exposure_col]
        .agg(
            long_amount=lambda values: values[values >= 0].sum(),
            short_amount=lambda values: values[values < 0].abs().sum(),
        )
        .reset_index()
    )
    daily["imbalance_pct"] = (daily["long_amount"] - daily["short_amount"]) / nav * 100

    if group_col:
        return daily.groupby(group_col)["imbalance_pct"].mean().sort_values(ascending=False)
    return float(daily["imbalance_pct"].mean())


def weighted_portfolio_return(
    weights: pd.DataFrame, component_returns: pd.DataFrame
) -> pd.Series:
    """Combine per-component returns into one portfolio return series.

    On each date, the portfolio return is the weighted average of that
    date's component returns, where weights are re-normalized to sum to 1
    across whichever components actually have data that day (so a
    component that didn't trade on a given day doesn't silently zero out
    the portfolio return - it's just excluded from that day's average).

    Parameters
    ----------
    weights : pandas.DataFrame
        Indexed by date, one column per component. Values are that
        component's weight in the portfolio on that date (e.g. its share
        of gross exposure - does not need to sum to exactly 1).
    component_returns : pandas.DataFrame
        Indexed by date, same column names as ``weights``. Periodic
        returns as decimals.

    Returns
    -------
    pandas.Series
        Indexed by date, the resulting portfolio-level periodic return.
    """
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a pandas DataFrame")
    if not isinstance(component_returns, pd.DataFrame):
        raise TypeError("component_returns must be a pandas DataFrame")

    shared_cols = [c for c in component_returns.columns if c in weights.columns]
    if not shared_cols:
        raise ValueError("weights and component_returns share no column names")

    returns_df = component_returns[shared_cols].copy()
    weights_df = weights.reindex(index=returns_df.index, columns=shared_cols)
    for df in (returns_df, weights_df):
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    row_weight_sum = weights_df.sum(axis=1, skipna=True).replace(0, np.nan)
    normalized_weights = weights_df.div(row_weight_sum, axis=0)
    return (returns_df * normalized_weights).sum(axis=1, min_count=1)
