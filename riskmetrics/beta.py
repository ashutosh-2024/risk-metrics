"""Beta, correlation, and risk-decomposition helpers.

"Beta" here always means the standard regression beta of one return series
against another: ``Cov(asset, benchmark) / Var(benchmark)`` - how much the
asset tends to move for a 1-unit move in the benchmark. Everything in this
module works on plain :class:`pandas.Series`/:class:`pandas.DataFrame`
objects of returns; there is no notion of "strategy" or "portfolio" baked
in, so it works equally well for a single stock vs. an index, a fund vs.
a peer benchmark, or a position vs. its own portfolio (for risk
decomposition).
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .exposure import weighted_portfolio_return

__all__ = [
    "beta",
    "beta_table",
    "correlation_matrix",
    "marginal_contribution_to_risk",
    "concentration_top_n",
    "factor_exposure_summary",
]


def _align_drop_na(a: pd.Series, b: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    return joined["a"].to_numpy(dtype=float), joined["b"].to_numpy(dtype=float)


def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Regression beta of ``asset_returns`` against ``benchmark_returns``.

    Parameters
    ----------
    asset_returns, benchmark_returns : pandas.Series
        Periodic returns as decimals, indexed by date. The two series are
        aligned on their shared dates automatically; rows where either
        side is missing are dropped.

    Returns
    -------
    float
        ``Cov(asset, benchmark) / Var(benchmark)``. ``nan`` if fewer than
        2 overlapping observations, or if the benchmark has zero variance.
    """
    a, b = _align_drop_na(asset_returns, benchmark_returns)
    if a.size < 2:
        return float("nan")
    variance = np.var(b, ddof=1)
    if variance == 0:
        return float("nan")
    covariance = np.cov(a, b, ddof=1)[0, 1]
    return float(covariance / variance)


def beta_table(
    asset_returns: pd.Series,
    benchmarks: pd.DataFrame,
    windows: Optional[Dict[str, Tuple[dt.date, dt.date]]] = None,
) -> pd.DataFrame:
    """Beta of one asset against several benchmarks, over several time windows.

    This is the standard "beta to Equities / Bonds / Credit / ..." table
    every risk report has, generalized to any set of benchmark series and
    any set of windows.

    Parameters
    ----------
    asset_returns : pandas.Series
        The thing you want betas *for* (e.g. a fund's or strategy's daily
        returns), indexed by date.
    benchmarks : pandas.DataFrame
        Indexed by date, one column per benchmark (e.g. an equity index,
        a bond index, a credit index...). Column names become row labels
        in the output.
    windows : dict[str, (date, date)], optional
        E.g. the output of :func:`riskmetrics.periods.get_date_ranges`.
        If omitted, a single ``"Full History"`` window covering all
        overlapping dates is used.

    Returns
    -------
    pandas.DataFrame
        Index = ``"Beta to <benchmark column name>"``. Columns = window
        labels. Values are betas, rounded to 2 dp.
    """
    if not isinstance(benchmarks, pd.DataFrame):
        raise TypeError(f"benchmarks must be a pandas DataFrame, got {type(benchmarks)!r}")

    asset = asset_returns.copy()
    if not isinstance(asset.index, pd.DatetimeIndex):
        asset.index = pd.to_datetime(asset.index)

    bench = benchmarks.copy()
    if not isinstance(bench.index, pd.DatetimeIndex):
        bench.index = pd.to_datetime(bench.index)

    if windows is None:
        windows = {"Full History": (asset.index.min().date(), asset.index.max().date())}

    rows = {f"Beta to {col}": {} for col in bench.columns}
    for label, (start, end) in windows.items():
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        asset_window = asset[(asset.index >= start_ts) & (asset.index <= end_ts)]
        bench_window = bench[(bench.index >= start_ts) & (bench.index <= end_ts)]
        for col in bench.columns:
            rows[f"Beta to {col}"][label] = beta(asset_window, bench_window[col])

    table = pd.DataFrame(rows).T
    table = table[list(windows.keys())]
    return table.round(2)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation matrix of a set of return series.

    Parameters
    ----------
    returns : pandas.DataFrame
        Indexed by date, one column per series (e.g. one column per
        strategy or per book). Rows with any missing values for a given
        pair are excluded pairwise (this is plain :meth:`DataFrame.corr`).

    Returns
    -------
    pandas.DataFrame
        Square correlation matrix, rounded to 2 dp.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a pandas DataFrame, got {type(returns)!r}")
    return returns.corr().round(2)


def marginal_contribution_to_risk(
    component_returns: pd.DataFrame,
    weights: pd.DataFrame,
    portfolio_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Decompose portfolio volatility into each component's contribution.

    For each component (e.g. each position/ticker), this computes:

    - ``beta``: how much that component's returns move with the
      portfolio's returns (``Cov(portfolio, component) / Var(portfolio)``)
    - ``weight``: that component's average weight in the portfolio
    - ``mc`` (marginal contribution): ``weight * beta * portfolio_volatility``
    - ``mc_pct``: ``mc`` as a share of the total marginal contribution
      across all components (sums to ~100%)

    This tells you, for example, which 10 positions out of 500 are
    actually driving your portfolio's risk - a position can be large in
    dollar terms but low-beta (small ``mc_pct``), or small in dollar terms
    but highly correlated with everything else (outsized ``mc_pct``).

    Parameters
    ----------
    component_returns : pandas.DataFrame
        Indexed by date, one column per component. Daily/periodic returns
        as decimals. Missing values (e.g. a stock that didn't trade that
        day) are fine and handled per-column.
    weights : pandas.DataFrame
        Same shape/columns as ``component_returns`` (or at least the same
        column names): each component's weight in the portfolio on each
        date (e.g. its share of total gross exposure that day). Only the
        per-column average is used.
    portfolio_returns : pandas.Series, optional
        The portfolio's own daily/periodic returns, indexed by date. If
        omitted, it is computed as the per-date weighted sum of
        ``component_returns`` using ``weights`` (weights are normalized
        to sum to 1 across available components on each date).

    Returns
    -------
    pandas.DataFrame
        Indexed by component name, columns ``["beta", "weight", "mc",
        "mc_pct"]``, sorted by descending ``abs(mc_pct)`` (biggest risk
        drivers first).
    """
    if not isinstance(component_returns, pd.DataFrame):
        raise TypeError("component_returns must be a pandas DataFrame")
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a pandas DataFrame")

    returns_df = component_returns.copy()
    weights_df = weights.copy()
    for df in (returns_df, weights_df):
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    shared_cols = [c for c in returns_df.columns if c in weights_df.columns]
    if not shared_cols:
        raise ValueError("component_returns and weights share no column names")
    returns_df = returns_df[shared_cols]
    weights_df = weights_df.reindex(index=returns_df.index, columns=shared_cols)

    if portfolio_returns is None:
        portfolio = weighted_portfolio_return(weights_df, returns_df)
    else:
        portfolio = portfolio_returns.copy()
        if not isinstance(portfolio.index, pd.DatetimeIndex):
            portfolio.index = pd.to_datetime(portfolio.index)
        portfolio = portfolio.reindex(returns_df.index)

    sigma_p = float(portfolio.std(ddof=1))

    # Vectorised per-column beta: mask the portfolio series with each
    # column's NaN pattern so beta_i only uses dates where component i
    # actually has a return, matching what you'd get calling beta() in a
    # loop but without the Python-level loop.
    r_p = portfolio.to_numpy(dtype=float)
    r_c = returns_df.to_numpy(dtype=float)
    r_p_masked = np.where(np.isnan(r_c), np.nan, r_p[:, np.newaxis])
    r_p_mean = np.nanmean(r_p_masked, axis=0)
    r_c_mean = np.nanmean(r_c, axis=0)
    cov_num = np.nansum((r_p_masked - r_p_mean) * (r_c - r_c_mean), axis=0)
    var_den = np.nansum((r_p_masked - r_p_mean) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        betas = cov_num / np.where(var_den == 0, np.nan, var_den)

    mean_weight = weights_df.mean(axis=0, skipna=True)

    result = pd.DataFrame(
        {
            "beta": betas,
            "weight": mean_weight.reindex(shared_cols).to_numpy(),
        },
        index=shared_cols,
    )
    result["mc"] = result["weight"] * result["beta"].fillna(0) * sigma_p
    total_mc = result["mc"].sum()
    result["mc_pct"] = result["mc"] / total_mc if total_mc != 0 else float("nan")
    result = result.reindex(result["mc_pct"].abs().sort_values(ascending=False).index)
    return result


def concentration_top_n(
    mc_pct: pd.Series, n_values: Iterable[int] = (10, 50)
) -> Dict[int, float]:
    """Share of total risk contribution held by the top-N components.

    Parameters
    ----------
    mc_pct : pandas.Series
        The ``mc_pct`` column from :func:`marginal_contribution_to_risk`
        (or any series of per-component risk-contribution shares).
    n_values : iterable of int, default (10, 50)
        Which "top N" cutoffs to report.

    Returns
    -------
    dict[int, float]
        ``{n: percentage}`` - e.g. ``{10: 42.5}`` means the top 10
        components account for 42.5% of total risk contribution.
    """
    ordered = mc_pct.reindex(mc_pct.abs().sort_values(ascending=False).index)
    return {n: float(ordered.iloc[:n].sum() * 100) for n in n_values}


def factor_exposure_summary(
    exposure: pd.DataFrame,
    windows: Dict[str, Tuple[dt.date, dt.date]],
    date_col: str = "date",
    factor_col: str = "factor",
    value_col: str = "exposure",
    top_n: Optional[int] = 5,
    rank_by: Optional[str] = None,
) -> pd.DataFrame:
    """Sum a factor-exposure time series into reporting windows, keep the biggest.

    Typical use: you have a long table of ``(date, factor, dollar
    exposure)`` rows (e.g. from a Barra-style risk model) and want "how
    much dollar exposure did we have to each factor, this month / trailing
    12 months / since inception", restricted to the factors that matter
    most.

    Parameters
    ----------
    exposure : pandas.DataFrame
        Long-format table with at least the three columns named by
        ``date_col``, ``factor_col``, ``value_col``.
    windows : dict[str, (date, date)]
        E.g. the output of :func:`riskmetrics.periods.get_date_ranges`.
    top_n : int, optional
        Keep only the top ``top_n`` factors by exposure magnitude. Pass
        ``None`` to keep all factors.
    rank_by : str, optional
        Which window's column to rank factors by when picking the top-N
        (must be a key in ``windows``). Defaults to the sum of absolute
        exposure across *all* windows.

    Returns
    -------
    pandas.DataFrame
        Index = factor name. Columns = window labels. Values = summed
        exposure within that window.
    """
    required = {date_col, factor_col, value_col}
    missing = required - set(exposure.columns)
    if missing:
        raise ValueError(f"exposure is missing required columns: {sorted(missing)}")

    df = exposure[[date_col, factor_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])

    summary = {}
    for label, (start, end) in windows.items():
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        mask = (df[date_col] >= start_ts) & (df[date_col] <= end_ts)
        summary[label] = df.loc[mask].groupby(factor_col)[value_col].sum()

    table = pd.DataFrame(summary)

    if top_n is not None:
        if rank_by is not None:
            if rank_by not in table.columns:
                raise ValueError(f"rank_by={rank_by!r} is not one of the window labels {list(windows)}")
            ranking = table[rank_by].abs()
        else:
            ranking = table.abs().sum(axis=1)
        table = table.loc[ranking.nlargest(top_n).index]

    return table
