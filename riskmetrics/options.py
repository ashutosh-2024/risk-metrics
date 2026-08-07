"""Black-Scholes option pricing/Greeks, and price-shock ("what if the
underlying moved X%") scenario analysis for an options book.

This is a standalone, textbook Black-Scholes implementation (no external
market-data or vendor-library dependency) so the package works for anyone,
not just users with access to a specific pricing library. It's a
simplification of real option pricing (constant volatility, European
exercise, no dividends) - good enough for a quick risk-scenario table, not
a substitute for a production options pricer.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "black_scholes_price",
    "black_scholes_greeks",
    "price_shock_pnl",
    "price_shock_summary",
]

ArrayLike = Union[float, Sequence[float], np.ndarray, pd.Series]


def _to_float_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _is_call_mask(option_type: ArrayLike) -> np.ndarray:
    types = np.atleast_1d(np.asarray(option_type, dtype=object))
    lowered = np.array([str(t).strip().lower() for t in types])
    is_call = lowered == "call"
    is_put = lowered == "put"
    if not np.all(is_call | is_put):
        bad = sorted(set(lowered[~(is_call | is_put)].tolist()))
        raise ValueError(f"option_type must be 'call' or 'put' (case-insensitive), got: {bad}")
    return is_call


def _bs_terms(spot, strike, time_to_expiry, risk_free_rate, volatility):
    spot = _to_float_array(spot)
    strike = _to_float_array(strike)
    time_to_expiry = _to_float_array(time_to_expiry)
    volatility = _to_float_array(volatility)

    if np.any(time_to_expiry <= 0):
        raise ValueError("time_to_expiry must be positive for every row")
    if np.any(volatility <= 0):
        raise ValueError("volatility must be positive for every row")
    if np.any(spot <= 0) or np.any(strike <= 0):
        raise ValueError("spot and strike must be positive for every row")

    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (
        volatility * sqrt_t
    )
    d2 = d1 - volatility * sqrt_t
    discount = np.exp(-risk_free_rate * time_to_expiry)
    return spot, strike, time_to_expiry, volatility, sqrt_t, d1, d2, discount


def black_scholes_price(
    spot: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    risk_free_rate: float,
    volatility: ArrayLike,
    option_type: ArrayLike,
) -> np.ndarray:
    """Black-Scholes theoretical price of a European call or put.

    All array-like arguments (``spot``, ``strike``, ``time_to_expiry``,
    ``volatility``, ``option_type``) must be the same length (or scalars,
    which broadcast against the others) - one entry per option.

    Parameters
    ----------
    spot : float | array-like
        Current price of the underlying.
    strike : float | array-like
        Option strike price.
    time_to_expiry : float | array-like
        Time to expiry, in **years** (e.g. 30 calendar days = ``30/365``).
    risk_free_rate : float
        Annualized risk-free rate as a decimal (e.g. ``0.05`` for 5%).
        A single scalar applied to every row.
    volatility : float | array-like
        Annualized implied volatility as a decimal (e.g. ``0.25`` for 25%).
    option_type : str | array-like of str
        ``"call"`` or ``"put"`` per row (case-insensitive).

    Returns
    -------
    numpy.ndarray
        Theoretical option price(s), same shape as the broadcast inputs.
    """
    spot, strike, time_to_expiry, volatility, sqrt_t, d1, d2, discount = _bs_terms(
        spot, strike, time_to_expiry, risk_free_rate, volatility
    )
    is_call = _is_call_mask(option_type)

    call_price = spot * norm.cdf(d1) - strike * discount * norm.cdf(d2)
    put_price = strike * discount * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return np.where(is_call, call_price, put_price)


def black_scholes_greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    risk_free_rate: float,
    volatility: ArrayLike,
    option_type: ArrayLike,
) -> Dict[str, np.ndarray]:
    """Black-Scholes delta, gamma, vega and theta for a European call or put.

    Same input conventions as :func:`black_scholes_price`.

    Returns
    -------
    dict[str, numpy.ndarray]
        - ``"delta"``: change in option price per 1.0 (100%) change in
          spot price, e.g. 0.5 for an at-the-money call.
        - ``"gamma"``: change in delta per 1.0 change in spot price.
        - ``"vega"``: change in option price per 1.0 (100 percentage
          points) change in volatility - divide by 100 for "per 1 vol
          point".
        - ``"theta"``: change in option price per 1.0 **year** of time
          decay (i.e. per calendar day, divide by 365) - typically
          negative (options lose value as expiry approaches).
    """
    spot, strike, time_to_expiry, volatility, sqrt_t, d1, d2, discount = _bs_terms(
        spot, strike, time_to_expiry, risk_free_rate, volatility
    )
    is_call = _is_call_mask(option_type)
    pdf_d1 = norm.pdf(d1)

    delta = np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)
    gamma = pdf_d1 / (spot * volatility * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t

    theta_call = -(spot * pdf_d1 * volatility) / (2 * sqrt_t) - risk_free_rate * strike * discount * norm.cdf(d2)
    theta_put = -(spot * pdf_d1 * volatility) / (2 * sqrt_t) + risk_free_rate * strike * discount * norm.cdf(-d2)
    theta = np.where(is_call, theta_call, theta_put)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def price_shock_pnl(
    positions: pd.DataFrame,
    shocks: Iterable[float] = (-0.05, -0.02, -0.01, 0.01, 0.02, 0.05),
    risk_free_rate: float = 0.0,
    spot_col: str = "spot",
    strike_col: str = "strike",
    time_to_expiry_col: str = "time_to_expiry",
    volatility_col: str = "volatility",
    option_type_col: str = "option_type",
    quantity_col: str = "quantity",
    multiplier_col: str = "contract_multiplier",
    fx_rate_col: Optional[str] = None,
) -> pd.DataFrame:
    """Reprice every option position under a set of underlying-price shocks.

    For each row (position) and each shock percentage, this repricing the
    option at ``spot * (1 + shock)`` (holding volatility and time to
    expiry fixed) and reports the resulting dollar P&L versus the
    unshocked price - i.e. "if the underlying moved +2% right now, this
    position would gain/lose $X".

    Parameters
    ----------
    positions : pandas.DataFrame
        One row per option position, with columns named by the
        ``*_col`` parameters below. Any extra columns (e.g. a date,
        symbol, or strategy name) are preserved untouched in the output,
        so you can group/filter the result afterwards.
    shocks : iterable of float, default (-5%, -2%, -1%, +1%, +2%, +5%)
        Underlying price shocks to apply, as decimals (``0.02`` = +2%).
    risk_free_rate : float, default 0.0
        Annualized risk-free rate as a decimal, applied to every row.
    quantity_col : str
        Signed position size (positive = long, negative = short), in
        number of contracts/options.
    multiplier_col : str
        Contract multiplier (e.g. 100 shares per equity option contract).
    fx_rate_col : str, optional
        If your positions are priced in a currency other than your
        reporting currency, name the column holding the FX rate to
        convert to reporting currency (multiplied in). Omit if already
        in reporting currency.

    Returns
    -------
    pandas.DataFrame
        A copy of ``positions`` with one extra column per shock, named
        like ``"pnl_shock_+2%"`` / ``"pnl_shock_-5%"``, holding the
        dollar P&L impact of that shock for each position.
    """
    required = [spot_col, strike_col, time_to_expiry_col, volatility_col, option_type_col, quantity_col, multiplier_col]
    missing = [c for c in required if c not in positions.columns]
    if missing:
        raise ValueError(f"positions is missing required columns: {missing}")

    df = positions.copy()
    spot = df[spot_col].to_numpy(dtype=float)
    strike = df[strike_col].to_numpy(dtype=float)
    ttm = df[time_to_expiry_col].to_numpy(dtype=float)
    vol = df[volatility_col].to_numpy(dtype=float)
    option_type = df[option_type_col].to_numpy()
    quantity = df[quantity_col].to_numpy(dtype=float)
    multiplier = df[multiplier_col].to_numpy(dtype=float)
    fx_rate = df[fx_rate_col].to_numpy(dtype=float) if fx_rate_col else 1.0

    base_price = black_scholes_price(spot, strike, ttm, risk_free_rate, vol, option_type)

    for shock in shocks:
        shocked_price = black_scholes_price(
            spot * (1.0 + shock), strike, ttm, risk_free_rate, vol, option_type
        )
        pnl = (shocked_price - base_price) * quantity * multiplier * fx_rate
        label = f"pnl_shock_{shock:+.0%}"
        df[label] = pnl

    return df


def price_shock_summary(
    shocked_positions: pd.DataFrame,
    shock_columns: Sequence[str],
    nav: float,
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate per-position shock P&L (from :func:`price_shock_pnl`) into a %-of-NAV summary.

    Parameters
    ----------
    shocked_positions : pandas.DataFrame
        Output of :func:`price_shock_pnl` (or any DataFrame with the
        shock P&L columns you want summarized).
    shock_columns : sequence of str
        Which columns to summarize (e.g. the ``"pnl_shock_*"`` columns
        added by :func:`price_shock_pnl`).
    nav : float
        Net asset value to divide by, expressing the result as a
        percentage of the book.
    date_col : str, optional
        If given, P&L is first summed per date, then the *average* daily
        %-of-NAV impact across all dates is reported (matches how a
        monthly/annual report would describe a typical day's exposure to
        the shock). If omitted, all rows are summed once and divided by
        ``nav``.

    Returns
    -------
    pandas.DataFrame
        One row, columns = ``shock_columns``, values = % of NAV.
    """
    if nav == 0:
        raise ValueError("nav must not be zero")
    missing = [c for c in shock_columns if c not in shocked_positions.columns]
    if missing:
        raise ValueError(f"shocked_positions is missing columns: {missing}")

    if date_col:
        daily = shocked_positions.groupby(date_col)[list(shock_columns)].sum()
        pct_of_nav = (daily / nav * 100).mean()
    else:
        pct_of_nav = shocked_positions[list(shock_columns)].sum() / nav * 100

    return pct_of_nav.to_frame(name="pct_of_nav").T
