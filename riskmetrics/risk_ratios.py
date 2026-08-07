"""Risk and risk-adjusted-return statistics computed from a plain series of
periodic returns: volatility, Sharpe/Sortino/Calmar/Omega ratios, VaR,
drawdowns, skew/kurtosis, autocorrelation, and win-rate.

Every function takes a :class:`pandas.Series` of **simple periodic returns
as decimals** (a day up 1.5% is ``0.015``) indexed by date, and a
``periods_per_year`` you choose to match your data's frequency (252 for
daily trading-day data is the common default; use 12 for monthly data,
52 for weekly, etc). Nothing here queries a database or knows about
"strategies" or "portfolios" - feed it any returns series, including a
single stock, a whole fund, or a backtest.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .returns import annualize_return, total_return

__all__ = [
    "volatility",
    "downside_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "omega_ratio",
    "skewness",
    "excess_kurtosis",
    "autocorrelation",
    "parametric_var",
    "drawdown_series",
    "max_drawdown",
    "drawdown_recovery_time",
    "win_rate",
    "best_period",
    "worst_period",
    "performance_summary",
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


def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of returns, as a decimal.

    Multiply by 100 to display as a percentage (e.g. ``0.16`` -> "16%
    annualized volatility").
    """
    series = _as_dated_series(returns)
    return float(series.std(ddof=1) * np.sqrt(periods_per_year))


def downside_volatility(
    returns: pd.Series, periods_per_year: int = 252, threshold: float = 0.0
) -> float:
    """Annualized standard deviation of returns *below* ``threshold`` only.

    This is the "how bad do the bad days get" half of volatility, used by
    the Sortino ratio. Returns ``nan`` if there are no observations below
    the threshold.
    """
    series = _as_dated_series(returns)
    downside = series[series < threshold]
    if downside.empty:
        return float("nan")
    return float(downside.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio: excess return per unit of total volatility.

    Parameters
    ----------
    returns : pandas.Series
        Periodic returns as decimals.
    risk_free_rate : float, default 0.0
        The *annualized* risk-free rate as a decimal (e.g. ``0.05`` for
        5%). It is converted to a per-period rate internally by dividing
        by ``periods_per_year``.
    periods_per_year : int, default 252

    Returns
    -------
    float
        Higher is better. ``nan`` if volatility is zero.
    """
    series = _as_dated_series(returns)
    per_period_rf = risk_free_rate / periods_per_year
    excess = series - per_period_rf
    std_dev = series.std(ddof=1)
    if std_dev == 0 or np.isnan(std_dev):
        return float("nan")
    return float((excess.mean() / std_dev) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio: excess return per unit of downside risk only.

    Same idea as :func:`sharpe_ratio`, but the denominator only penalizes
    volatility from losing periods (see :func:`downside_volatility`)
    rather than all volatility - a fund that only ever surprises to the
    upside is not "risky" in the way Sortino measures it.
    """
    series = _as_dated_series(returns)
    per_period_rf = risk_free_rate / periods_per_year
    excess_mean = series.mean() - per_period_rf
    downside = series[series < 0]
    if downside.empty:
        return float("nan")
    downside_std = downside.std(ddof=1)
    if downside_std == 0 or np.isnan(downside_std):
        return float("nan")
    return float((excess_mean / downside_std) * np.sqrt(periods_per_year))


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized return divided by the worst (max) drawdown over the period.

    Rewards steady compounding and punishes deep drawdowns more than
    Sharpe/Sortino do, since it only looks at the single worst peak-to-
    trough loss rather than the whole volatility profile.

    Returns ``nan`` if there was no drawdown at all (a straight line up).
    """
    series = _as_dated_series(returns)
    total = total_return(series)
    annualized = annualize_return(total, len(series), periods_per_year=periods_per_year)
    mdd = max_drawdown(series)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(annualized / abs(mdd))


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Ratio of the sum of gains above ``threshold`` to the sum of losses below it.

    A value > 1 means the periodic wins outweighed the periodic losses
    (relative to ``threshold``). Unlike Sharpe/Sortino this uses raw sums
    rather than standard deviations, so it is not annualized - it is a
    ratio over whatever window ``returns`` covers.
    """
    series = _as_dated_series(returns)
    gains = (series[series > threshold] - threshold).sum()
    losses = (threshold - series[series < threshold]).sum()
    if losses == 0:
        return float("nan")
    return float(gains / losses)


def skewness(returns: pd.Series) -> float:
    """Skewness of the return distribution (0 = symmetric, negative = fat left tail)."""
    series = _as_dated_series(returns)
    return float(stats.skew(series, bias=False))


def excess_kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis of the return distribution (0 = normal-distribution-like tails;
    positive = fatter tails / more extreme outliers than a normal distribution)."""
    series = _as_dated_series(returns)
    return float(stats.kurtosis(series, fisher=True, bias=False))


def autocorrelation(returns: pd.Series, lag: int = 1, drop_zero: bool = True) -> float:
    """Lag-``lag`` autocorrelation of returns.

    Parameters
    ----------
    drop_zero : bool, default True
        Drop periods with exactly zero return before computing the
        correlation. Useful for instruments that don't trade every day
        (a run of literal zeros from no trading would otherwise dilute
        the estimate).
    """
    series = _as_dated_series(returns)
    if drop_zero:
        series = series[series != 0]
    if len(series) <= lag:
        return float("nan")
    return float(series.autocorr(lag=lag))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (variance-covariance) Value-at-Risk for a single period.

    Assumes returns are normally distributed and estimates VaR as
    ``z * std_dev``, where ``z`` is the one-sided normal quantile for
    ``confidence`` (e.g. 1.645 for 95%).

    Returns
    -------
    float
        A positive decimal representing the estimated loss magnitude at
        the given confidence level for a single period (e.g. ``0.023`` ->
        "estimated 95% one-day VaR of 2.3%"). This is a simplification
        (real return distributions have fatter tails than normal) - treat
        it as a quick, standard estimate, not a precise tail-risk model.
    """
    series = _as_dated_series(returns)
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0, e.g. 0.95")
    z = stats.norm.ppf(confidence)
    return float(z * series.std(ddof=1))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Peak-to-current drawdown at every point in time.

    Returns
    -------
    pandas.Series
        Same index as the input. Value at each date is
        ``(cumulative_return_so_far / running_peak) - 1``, i.e. 0 at new
        highs and negative while underwater (e.g. ``-0.12`` = 12% below
        the prior peak).
    """
    series = _as_dated_series(returns)
    cumulative = (1.0 + series).cumprod()
    running_peak = cumulative.cummax()
    return (cumulative / running_peak) - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """The single worst peak-to-trough loss over the series, as a negative decimal."""
    dd = drawdown_series(returns)
    return float(dd.min())


def drawdown_recovery_time(returns: pd.Series) -> Optional[int]:
    """Number of periods from the worst drawdown's trough back to break-even.

    Returns
    -------
    int or None
        Number of periods (e.g. trading days) it took to recover from the
        worst drawdown back to a new high-water mark. ``None`` if the
        fund never fully recovered within the series (still underwater at
        the end), or if there was no drawdown at all.
    """
    dd = drawdown_series(returns)
    if dd.min() == 0:
        return None
    trough_date = dd.idxmin()
    after_trough = dd.loc[trough_date:]
    recovered = after_trough[after_trough >= 0]
    if recovered.empty:
        return None
    recovery_date = recovered.index[0]
    return int(len(dd.loc[trough_date:recovery_date]) - 1)


def win_rate(returns: pd.Series) -> float:
    """Percentage of *traded* periods (nonzero return) that were positive, 0-100."""
    series = _as_dated_series(returns)
    traded = series[series != 0]
    if traded.empty:
        return float("nan")
    return float((traded > 0).sum() / len(traded) * 100)


def best_period(returns: pd.Series) -> float:
    """The single best periodic return in the series, as a percentage."""
    series = _as_dated_series(returns)
    return float(series.max() * 100)


def worst_period(returns: pd.Series) -> float:
    """The single worst periodic return in the series, as a percentage."""
    series = _as_dated_series(returns)
    return float(series.min() * 100)


def performance_summary(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    min_periods_for_annualization: Optional[int] = None,
) -> Dict[str, float]:
    """Run every metric in this module once and return them as one dict.

    This is the convenient "just give me the whole report row" entry
    point - equivalent to calling each of the other functions in this
    module yourself and collecting the results.

    Parameters
    ----------
    returns : pandas.Series
        Periodic returns as decimals, indexed by date.
    risk_free_rate : float, default 0.0
        Annualized risk-free rate as a decimal, used by Sharpe/Sortino.
    periods_per_year : int, default 252
    min_periods_for_annualization : int, optional
        See :func:`riskmetrics.returns.annualize_return`.

    Returns
    -------
    dict[str, float]
        Keys: ``total_return_pct``, ``annualized_return_pct``,
        ``annualized_volatility_pct``, ``downside_volatility_pct``,
        ``sharpe_ratio``, ``sortino_ratio``, ``calmar_ratio``,
        ``omega_ratio``, ``skewness``, ``excess_kurtosis``,
        ``autocorrelation``, ``parametric_var_95_pct``,
        ``max_drawdown_pct``, ``drawdown_recovery_periods``,
        ``win_rate_pct``, ``best_period_pct``, ``worst_period_pct``.
        All ``*_pct`` values are already multiplied by 100.
    """
    series = _as_dated_series(returns)
    total = total_return(series)
    annualized = annualize_return(
        total,
        len(series),
        periods_per_year=periods_per_year,
        min_periods_for_annualization=min_periods_for_annualization,
    )

    return {
        "total_return_pct": total * 100,
        "annualized_return_pct": annualized * 100,
        "annualized_volatility_pct": volatility(series, periods_per_year) * 100,
        "downside_volatility_pct": downside_volatility(series, periods_per_year) * 100,
        "sharpe_ratio": sharpe_ratio(series, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(series, risk_free_rate, periods_per_year),
        "calmar_ratio": calmar_ratio(series, periods_per_year),
        "omega_ratio": omega_ratio(series),
        "skewness": skewness(series),
        "excess_kurtosis": excess_kurtosis(series),
        "autocorrelation": autocorrelation(series),
        "parametric_var_95_pct": parametric_var(series, 0.95) * 100,
        "max_drawdown_pct": max_drawdown(series) * 100,
        "drawdown_recovery_periods": drawdown_recovery_time(series),
        "win_rate_pct": win_rate(series),
        "best_period_pct": best_period(series),
        "worst_period_pct": worst_period(series),
    }
