"""Ready-made matplotlib charts for the standard tearsheet visuals: a
growth curve, a bar chart of periodic returns, an underwater drawdown
chart, and a rolling-volatility line.

Every ``plot_*`` function returns a :class:`matplotlib.figure.Figure` -
nothing is shown or saved automatically. Call :func:`save_figure` (or
``fig.savefig(...)`` yourself) when you're ready to write it to disk, or
hand the figure to a notebook / PDF builder / Dash app.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .risk_ratios import drawdown_series

__all__ = [
    "tick_decimal_places",
    "style_minimal_axes",
    "format_percent_axis",
    "plot_growth_curve",
    "plot_period_returns_bar",
    "plot_drawdown",
    "plot_rolling_volatility",
    "save_figure",
]


def _as_dated_series(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError(f"expected a pandas Series, got {type(returns)!r}")
    out = returns.astype(float).copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    return out.sort_index()


def tick_decimal_places(ticks) -> int:
    """How many decimal places axis-tick labels need to look distinct.

    Looks at the gap between consecutive ticks (not their absolute
    values) and picks 0, 1, or 2 decimal places so labels like "1%",
    "2%", "3%" don't get needlessly padded to "1.00%", while a fine-
    grained axis like "0.1%, 0.2%, ..." doesn't collapse into
    indistinguishable rounded labels.

    Parameters
    ----------
    ticks : array-like of float
        Typically the output of ``ax.get_yticks()`` / ``ax.get_xticks()``.

    Returns
    -------
    int
        0, 1, or 2.
    """
    values = np.asarray(ticks, dtype=float)
    if values.size < 2:
        return 1
    step = np.min(np.abs(np.diff(values))) * (1 + 1e-9)
    if step >= 1:
        return 0
    if step >= 0.1:
        return 1
    return 2


def style_minimal_axes(ax: Axes) -> None:
    """Apply a clean, minimal look: no top/right border, dashed horizontal grid."""
    ax.spines[["right", "top"]].set_visible(False)
    ax.margins(x=0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)


def format_percent_axis(ax: Axes, axis: str = "y", decimals: Optional[int] = None) -> None:
    """Format an axis's tick labels as percentages (multiplies the raw value by nothing -
    the data itself must already be in percentage units, e.g. 12.5 for "12.5%").

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    axis : {"x", "y"}, default "y"
    decimals : int, optional
        Fixed number of decimal places. If omitted, it's chosen
        automatically per :func:`tick_decimal_places` based on the
        current tick spacing.
    """
    target = ax.yaxis if axis == "y" else ax.xaxis
    ticks = ax.get_yticks() if axis == "y" else ax.get_xticks()
    dp = decimals if decimals is not None else tick_decimal_places(ticks)
    target.set_major_formatter(lambda value, _pos: f"{value:.{dp}f}%")


def plot_growth_curve(
    cumulative_growth: pd.Series,
    ax: Optional[Axes] = None,
    color: str = "#0b3d66",
    linewidth: float = 2.5,
    title: Optional[str] = None,
) -> Figure:
    """Plot a cumulative-growth ("growth of $1, as a %") line chart.

    Parameters
    ----------
    cumulative_growth : pandas.Series
        Indexed by date, values as decimals (e.g. ``0.10`` = +10% grown
        so far). This is exactly the output of
        :func:`riskmetrics.returns.cumulative_growth`.
    ax : matplotlib.axes.Axes, optional
        Draw onto an existing axes instead of creating a new figure.
    color, linewidth : styling passthroughs.
    title : str, optional
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    series = _as_dated_series(cumulative_growth) * 100
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 6), dpi=150)

    ax.plot(series.index, series.values, lw=linewidth, color=color)
    style_minimal_axes(ax)
    format_percent_axis(ax)
    ax.tick_params(axis="x", rotation=45)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_period_returns_bar(
    returns: pd.Series,
    ax: Optional[Axes] = None,
    color: str = "#2e3f4f",
    title: Optional[str] = None,
) -> Figure:
    """Bar chart of periodic (e.g. daily) returns, in percent.

    Parameters
    ----------
    returns : pandas.Series
        Periodic returns as decimals (``0.01`` = +1%), indexed by date.
    """
    series = _as_dated_series(returns) * 100
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 6), dpi=150)

    ax.bar(series.index, series.values, width=0.8, color=color, alpha=0.7)
    style_minimal_axes(ax)
    format_percent_axis(ax)
    ax.tick_params(axis="x", rotation=45)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_drawdown(
    returns: pd.Series,
    ax: Optional[Axes] = None,
    color: str = "#2c3e50",
    title: Optional[str] = None,
) -> Figure:
    """"Underwater" chart: how far below the running peak the portfolio is over time.

    Parameters
    ----------
    returns : pandas.Series
        Periodic returns as decimals, indexed by date.
    """
    dd = drawdown_series(returns) * 100
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 6), dpi=150)

    ax.fill_between(dd.index, dd.values, 0, color=color)
    ax.spines[["right", "top"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    format_percent_axis(ax)
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlim(dd.index.min(), dd.index.max())
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_rolling_volatility(
    returns: pd.Series,
    window: int = 252,
    periods_per_year: int = 252,
    ax: Optional[Axes] = None,
    color: str = "#0b3d66",
    title: Optional[str] = None,
) -> Figure:
    """Rolling annualized volatility line chart.

    Parameters
    ----------
    returns : pandas.Series
        Periodic returns as decimals, indexed by date.
    window : int, default 252
        Rolling lookback window, in periods.
    periods_per_year : int, default 252
        Used to annualize the rolling standard deviation (see
        :func:`riskmetrics.risk_ratios.volatility`).
    """
    series = _as_dated_series(returns)
    rolling_vol = series.rolling(window=window, min_periods=1).std() * np.sqrt(periods_per_year) * 100
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 6), dpi=150)

    ax.plot(rolling_vol.index, rolling_vol.values, lw=2.5, color=color)
    style_minimal_axes(ax)
    format_percent_axis(ax)
    ax.tick_params(axis="x", rotation=45)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path, dpi: int = 150) -> None:
    """Save a figure to disk, tightly cropped with no extra padding.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str | pathlib.Path
        Destination file path (parent directories are created if needed).
    dpi : int, default 150
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05, facecolor="white")
