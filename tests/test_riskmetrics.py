"""A handful of sanity-check tests using known, hand-checkable inputs.

Not an exhaustive test suite - just enough to catch obvious regressions
(sign errors, off-by-one windowing, broken imports) in each module.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import riskmetrics as rm


DATES = pd.bdate_range("2024-01-01", periods=300)


def test_get_date_ranges_mtd_and_itd():
    windows = rm.get_date_ranges(as_of="2024-03-15", inception_date="2023-01-01")
    assert windows["MTD"] == (dt.date(2024, 3, 1), dt.date(2024, 3, 15))
    assert windows["ITD"] == (dt.date(2023, 1, 1), dt.date(2024, 3, 15))
    assert windows["12M"][1] == dt.date(2024, 3, 15)


def test_total_return_compounds_correctly():
    returns = pd.Series([0.10, 0.10], index=DATES[:2])
    # 1.10 * 1.10 - 1 = 0.21
    assert rm.total_return(returns) == pytest.approx(0.21, abs=1e-9)


def test_annualize_return_respects_min_periods():
    # Short window -> returned unannualized when below the threshold.
    result = rm.annualize_return(0.05, num_periods=10, periods_per_year=252, min_periods_for_annualization=252)
    assert result == 0.05


def test_volatility_and_sharpe_are_finite_and_signed():
    rng = np.random.default_rng(0)
    up_returns = pd.Series(rng.normal(0.001, 0.01, 252), index=DATES[:252])
    vol = rm.volatility(up_returns)
    assert vol > 0
    sharpe = rm.sharpe_ratio(up_returns, risk_free_rate=0.0)
    assert np.isfinite(sharpe)


def test_max_drawdown_is_zero_for_monotonic_gains():
    always_up = pd.Series([0.01] * 50, index=DATES[:50])
    assert rm.max_drawdown(always_up) == 0.0


def test_beta_of_series_with_itself_is_one():
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(0, 0.01, 100), index=DATES[:100])
    assert rm.beta(series, series) == pytest.approx(1.0, abs=1e-9)


def test_long_short_gross_net():
    exposure = pd.Series([100.0, -50.0, 25.0, -25.0])
    result = rm.long_short_gross_net(exposure, nav=1000)
    assert result["long"] == 125.0
    assert result["short"] == 75.0
    assert result["gross"] == 200.0
    assert result["net"] == 50.0
    assert result["gross_pct"] == pytest.approx(20.0)


def test_days_to_liquidate_and_bucket():
    days = rm.days_to_liquidate(position_notional=1_000_000, adv_notional=1_000_000, participation_rate=0.20)
    assert days == pytest.approx(5.0)
    assert rm.liquidity_bucket(days) == "2d-7d"


def test_black_scholes_call_put_parity():
    call = rm.black_scholes_price(100, 100, 1.0, 0.05, 0.2, "call")
    put = rm.black_scholes_price(100, 100, 1.0, 0.05, 0.2, "put")
    # Put-call parity: C - P = S - K*exp(-rT)
    lhs = call - put
    rhs = 100 - 100 * np.exp(-0.05 * 1.0)
    assert lhs == pytest.approx(rhs, abs=1e-6)


def test_marginal_contribution_to_risk_sums_to_one():
    rng = np.random.default_rng(2)
    returns = pd.DataFrame(rng.normal(0, 0.01, (200, 5)), index=DATES[:200], columns=list("ABCDE"))
    weights = pd.DataFrame(rng.random((200, 5)), index=DATES[:200], columns=list("ABCDE"))
    weights = weights.div(weights.sum(axis=1), axis=0)
    mc = rm.marginal_contribution_to_risk(returns, weights)
    assert mc["mc_pct"].sum() == pytest.approx(1.0, abs=1e-6)
