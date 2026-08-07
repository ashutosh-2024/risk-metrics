"""Liquidity metrics: how long would it take to unwind a position or a book.

The core idea is "days to liquidate": if you can only trade a fixed
percentage of a security's typical daily volume without moving the market
(the "participation rate"), how many trading days would it take to exit a
position of a given size. This module has no notion of asset class,
exchange, or currency - feed it notional position sizes and average daily
traded notional in whatever consistent unit you like (shares, USD, etc).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "rolling_average_daily_volume",
    "days_to_liquidate",
    "liquidity_bucket",
    "liquidity_distribution",
    "DEFAULT_LIQUIDITY_EDGES",
]

# Each entry is (upper_bound_in_days_inclusive, bucket_label). Anything
# larger than the last upper bound falls into the overflow label passed to
# liquidity_bucket / liquidity_distribution (default ">365d").
DEFAULT_LIQUIDITY_EDGES: List[Tuple[float, str]] = [
    (1, "<=1d"),
    (7, "2d-7d"),
    (30, "8d-30d"),
    (90, "31d-90d"),
    (180, "91d-180d"),
    (365, "181d-365d"),
]

Number = Union[float, int, np.ndarray, pd.Series]


def rolling_average_daily_volume(
    volume: pd.Series, window: int = 21, stat: str = "median"
) -> pd.Series:
    """Rolling average-daily-volume (ADV) from a raw daily volume series.

    Parameters
    ----------
    volume : pandas.Series
        Daily traded volume (shares, contracts, or notional - whatever
        unit you want ADV in), indexed by date.
    window : int, default 21
        Lookback window in periods (21 trading days ~= 1 month is the
        common convention).
    stat : {"median", "mean"}, default "median"
        Median is more robust to one-off volume spikes; mean is the
        simpler textbook definition.

    Returns
    -------
    pandas.Series
        Same index as ``volume``. The first ``window - 1`` values are
        computed from however many observations are available so far
        (i.e. no leading ``NaN``s).
    """
    if not isinstance(volume, pd.Series):
        raise TypeError(f"volume must be a pandas Series, got {type(volume)!r}")
    rolling = volume.astype(float).rolling(window=window, min_periods=1)
    if stat == "median":
        return rolling.median()
    if stat == "mean":
        return rolling.mean()
    raise ValueError(f"stat must be 'median' or 'mean', got {stat!r}")


def days_to_liquidate(
    position_notional: Number, adv_notional: Number, participation_rate: float = 0.20
) -> Number:
    """How many trading days to exit a position at a given participation rate.

    Parameters
    ----------
    position_notional : float | array-like
        The size of the position, in the same currency/unit as
        ``adv_notional`` (dollars, shares, whatever - just be consistent).
        Sign doesn't matter, only magnitude is used.
    adv_notional : float | array-like
        The security's average daily traded amount, same unit as
        ``position_notional``.
    participation_rate : float, default 0.20
        The maximum fraction of a day's ADV you're willing to trade
        without excessive market impact (20% is a common desk default;
        5% is a more conservative one for less liquid names).

    Returns
    -------
    float | array-like
        ``abs(position_notional) / (participation_rate * adv_notional)``.
        Same type/shape as the inputs (works with plain floats, numpy
        arrays, or pandas Series).
    """
    if not 0 < participation_rate <= 1:
        raise ValueError("participation_rate must be in (0, 1]")
    return np.abs(position_notional) / (participation_rate * adv_notional)


def liquidity_bucket(
    days: float,
    edges: Sequence[Tuple[float, str]] = DEFAULT_LIQUIDITY_EDGES,
    overflow_label: str = ">365d",
) -> str:
    """Label a "days to liquidate" number into a human bucket (e.g. "2d-7d").

    Parameters
    ----------
    days : float
        Result of :func:`days_to_liquidate` for a single position.
        ``NaN``, zero, or negative values are treated as immediately
        liquid and fall into the first bucket.
    edges : sequence of (upper_bound, label), default :data:`DEFAULT_LIQUIDITY_EDGES`
        Must be sorted ascending by ``upper_bound``.
    overflow_label : str, default ">365d"
        Label used when ``days`` exceeds every bound in ``edges``.

    Returns
    -------
    str
    """
    if pd.isna(days) or days <= edges[0][0]:
        return edges[0][1]
    for upper_bound, label in edges:
        if days <= upper_bound:
            return label
    return overflow_label


def liquidity_distribution(
    positions: pd.DataFrame,
    notional_col: str,
    adv_col: str,
    participation_rates: Iterable[float] = (0.20, 0.05),
    edges: Sequence[Tuple[float, str]] = DEFAULT_LIQUIDITY_EDGES,
    overflow_label: str = ">365d",
) -> pd.DataFrame:
    """What % of the book could be liquidated within each time bucket.

    For each participation rate, this buckets every position by its
    "days to liquidate" and reports the **cumulative** percentage of book
    notional that falls within each bucket or faster - e.g. a value of
    ``72.0`` in the ``"8d-30d"`` column means 72% of the book (by
    notional) could be unwound in 30 trading days or less at that
    participation rate.

    Parameters
    ----------
    positions : pandas.DataFrame
        One row per position, with a notional-size column and an ADV
        column (see :func:`days_to_liquidate`).
    notional_col, adv_col : str
        Column names in ``positions``.
    participation_rates : iterable of float, default (0.20, 0.05)
        Which participation-rate scenarios to compute (see
        :func:`days_to_liquidate`).
    edges, overflow_label
        Passed through to :func:`liquidity_bucket`.

    Returns
    -------
    pandas.DataFrame
        Index = participation rate label (e.g. ``"20%"``, ``"5%"``).
        Columns = bucket labels, in increasing-liquidity-horizon order.
        Values = cumulative % of total book notional (0-100), rounded to
        2 dp, capped at 100.
    """
    for col in (notional_col, adv_col):
        if col not in positions.columns:
            raise ValueError(f"positions is missing column {col!r}")

    bucket_labels = [label for _, label in edges] + [overflow_label]
    total_notional = positions[notional_col].abs().sum()
    if total_notional == 0:
        raise ValueError("total notional across positions is zero")

    rows = {}
    for rate in participation_rates:
        days = days_to_liquidate(positions[notional_col], positions[adv_col], rate)
        bucket = days.apply(lambda d: liquidity_bucket(d, edges, overflow_label))
        pct_by_bucket = (
            positions[notional_col]
            .abs()
            .groupby(bucket)
            .sum()
            .div(total_notional)
            .mul(100)
            .reindex(bucket_labels, fill_value=0.0)
        )
        cumulative = pct_by_bucket.cumsum().clip(upper=100).round(2)
        rows[f"{rate * 100:g}%"] = cumulative

    return pd.DataFrame(rows).T[bucket_labels]
