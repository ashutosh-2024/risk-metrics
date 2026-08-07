"""Runnable demo of every module in the riskmetrics package, using made-up
(synthetic) data - no database, no external file, no real fund required.

Run it with:

    cd riskmetrics
    pip install -e .
    python examples/quickstart.py

It prints each metric with a short label, and writes a few example charts
to ``examples/output/``. Read this file top-to-bottom as a cookbook: each
section shows the shape of data a function expects and what it hands back.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import riskmetrics as rm

OUTPUT_DIR = Path(__file__).parent / "output"


def make_synthetic_data(seed: int = 7):
    """Build a fake but internally-consistent dataset: 3 years of daily
    returns for one portfolio, two benchmark indices, 30 individual
    positions with weights/returns/exposure, and a small options book."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", "2025-12-31")

    portfolio_returns = pd.Series(rng.normal(0.0004, 0.009, len(dates)), index=dates)

    benchmarks = pd.DataFrame(
        {
            "Equities": rng.normal(0.0003, 0.008, len(dates)),
            "Bonds": rng.normal(0.0001, 0.003, len(dates)),
        },
        index=dates,
    )

    tickers = [f"STOCK_{i:02d}" for i in range(30)]
    component_returns = pd.DataFrame(
        rng.normal(0.0003, 0.02, (len(dates), len(tickers))), index=dates, columns=tickers
    )
    raw_weights = rng.random((len(dates), len(tickers)))
    weights = pd.DataFrame(raw_weights, index=dates, columns=tickers)
    weights = weights.div(weights.sum(axis=1), axis=0)  # normalize to sum to 1 each day

    latest_exposure = pd.Series(
        rng.normal(0, 500_000, len(tickers)), index=tickers, name="exposure_usd"
    )

    positions = pd.DataFrame(
        {
            "symbol": tickers,
            "notional": latest_exposure.abs().values,
            "adv_notional": rng.uniform(200_000, 5_000_000, len(tickers)),
        }
    )

    options_book = pd.DataFrame(
        {
            "symbol": ["OPT_CALL_1", "OPT_PUT_1", "OPT_CALL_2"],
            "spot": [100.0, 100.0, 250.0],
            "strike": [105.0, 95.0, 240.0],
            "time_to_expiry": [30 / 365, 30 / 365, 60 / 365],
            "volatility": [0.25, 0.30, 0.22],
            "option_type": ["call", "put", "call"],
            "quantity": [50, -30, 20],
            "contract_multiplier": [100, 100, 100],
        }
    )

    return portfolio_returns, benchmarks, component_returns, weights, latest_exposure, positions, options_book


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (
        portfolio_returns,
        benchmarks,
        component_returns,
        weights,
        latest_exposure,
        positions,
        options_book,
    ) = make_synthetic_data()

    # ---------------------------------------------------------------
    section("periods: standard reporting windows")
    # ---------------------------------------------------------------
    as_of = portfolio_returns.index.max().date()
    windows = rm.get_date_ranges(as_of=as_of, inception_date="2023-01-02")
    for label, (start, end) in windows.items():
        print(f"  {label:>4}: {start} -> {end}")

    # ---------------------------------------------------------------
    section("returns: total / annualized / monthly table")
    # ---------------------------------------------------------------
    total = rm.total_return(portfolio_returns)
    annualized = rm.annualize_return(total, len(portfolio_returns))
    print(f"  total return over full history: {total * 100:.2f}%")
    print(f"  annualized return:               {annualized * 100:.2f}%")

    monthly_table = rm.monthly_returns_table(portfolio_returns)
    print("\n  Monthly returns table (%, first 3 years):")
    print(monthly_table.to_string())

    quarterly = rm.quarterly_returns_table(portfolio_returns)
    print("\n  Quarterly returns (%):")
    print(quarterly.round(2).to_string())

    growth = rm.cumulative_growth(portfolio_returns)

    # ---------------------------------------------------------------
    section("risk_ratios: one-call performance summary")
    # ---------------------------------------------------------------
    summary = rm.performance_summary(portfolio_returns, risk_free_rate=0.03)
    for key, value in summary.items():
        print(f"  {key:<28}: {value}")

    # ---------------------------------------------------------------
    section("beta: vs benchmarks, correlation, risk concentration")
    # ---------------------------------------------------------------
    beta_to_equities = rm.beta(portfolio_returns, benchmarks["Equities"])
    print(f"  beta to Equities (full history): {beta_to_equities:.2f}")

    beta_tbl = rm.beta_table(portfolio_returns, benchmarks, windows=windows)
    print("\n  Beta table by window:")
    print(beta_tbl.to_string())

    corr = rm.correlation_matrix(component_returns.iloc[:, :5])
    print("\n  Correlation matrix (first 5 tickers):")
    print(corr.to_string())

    mc = rm.marginal_contribution_to_risk(component_returns, weights)
    print("\n  Top 5 risk contributors (marginal_contribution_to_risk):")
    print(mc.head(5).round(4).to_string())

    concentration = rm.concentration_top_n(mc["mc_pct"], n_values=(5, 10))
    print(f"\n  Top-5 positions account for {concentration[5]:.1f}% of total risk contribution")
    print(f"  Top-10 positions account for {concentration[10]:.1f}% of total risk contribution")

    # ---------------------------------------------------------------
    section("exposure: long/short/gross/net, imbalance, weighted return")
    # ---------------------------------------------------------------
    nav = 50_000_000
    ls = rm.long_short_gross_net(latest_exposure, nav=nav)
    print(f"  Long: ${ls['long']:,.0f} ({ls['long_pct']:.1f}% NAV)")
    print(f"  Short: ${ls['short']:,.0f} ({ls['short_pct']:.1f}% NAV)")
    print(f"  Gross: {ls['gross_pct']:.1f}% NAV   Net: {ls['net_pct']:.1f}% NAV")

    counts = rm.position_counts(latest_exposure)
    print(f"  Position count: {counts['long']} long / {counts['short']} short")

    combined_return = rm.weighted_portfolio_return(weights, component_returns)
    print(f"  Reconstructed portfolio return, first day: {combined_return.iloc[0] * 100:.3f}%")

    # ---------------------------------------------------------------
    section("liquidity: ADV and days-to-liquidate")
    # ---------------------------------------------------------------
    liq_table = rm.liquidity_distribution(positions, notional_col="notional", adv_col="adv_notional")
    print("  Cumulative % of book liquidatable within each horizon:")
    print(liq_table.to_string())

    # ---------------------------------------------------------------
    section("options: Black-Scholes pricing and a price-shock scenario table")
    # ---------------------------------------------------------------
    base_prices = rm.black_scholes_price(
        options_book["spot"], options_book["strike"], options_book["time_to_expiry"],
        risk_free_rate=0.04, volatility=options_book["volatility"], option_type=options_book["option_type"],
    )
    print(f"  Base theoretical prices: {np.round(base_prices, 2).tolist()}")

    shocked = rm.price_shock_pnl(options_book, risk_free_rate=0.04)
    shock_cols = [c for c in shocked.columns if c.startswith("pnl_shock_")]
    print("\n  Per-position P&L under each shock ($):")
    print(shocked[["symbol", *shock_cols]].round(0).to_string(index=False))

    shock_summary = rm.price_shock_summary(shocked, shock_cols, nav=nav)
    print("\n  Book-wide impact, % of NAV:")
    print(shock_summary.round(3).to_string(index=False))

    # ---------------------------------------------------------------
    section("charts: saving example figures")
    # ---------------------------------------------------------------
    rm.save_figure(rm.plot_growth_curve(growth, title="Growth of $1"), OUTPUT_DIR / "growth_curve.png")
    rm.save_figure(rm.plot_drawdown(portfolio_returns, title="Drawdown"), OUTPUT_DIR / "drawdown.png")
    rm.save_figure(
        rm.plot_rolling_volatility(portfolio_returns, window=63, title="Rolling 3M Volatility"),
        OUTPUT_DIR / "rolling_volatility.png",
    )
    rm.save_figure(
        rm.plot_period_returns_bar(portfolio_returns.tail(60), title="Last 60 Days"),
        OUTPUT_DIR / "daily_returns.png",
    )
    print(f"  Saved 4 example charts to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
