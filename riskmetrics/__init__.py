"""riskmetrics: a generic, dependency-light toolkit for portfolio risk and
performance reporting.

Every function in this package is a plain function of pandas/numpy data
you already have (a series of returns, a table of positions, ...) - there
is no database, no fixed portfolio/strategy schema, and no assumption
about asset class. It was extracted from a hedge-fund reporting pipeline,
generalized so the underlying math is reusable on its own.

See the project README for a full walkthrough. Quick map of the modules:

- :mod:`riskmetrics.periods` - standard MTD/12M/36M/ITD reporting windows
- :mod:`riskmetrics.returns` - total/annualized return, monthly & quarterly tables
- :mod:`riskmetrics.risk_ratios` - volatility, Sharpe/Sortino/Calmar/Omega, VaR, drawdown
- :mod:`riskmetrics.beta` - beta, correlation, marginal contribution to risk, factor exposure
- :mod:`riskmetrics.exposure` - long/short/gross/net exposure, imbalance, weighted returns
- :mod:`riskmetrics.liquidity` - average daily volume, days-to-liquidate, liquidity buckets
- :mod:`riskmetrics.options` - Black-Scholes pricing/Greeks and price-shock scenarios
- :mod:`riskmetrics.charts` - matplotlib charts for growth, drawdown, rolling vol, returns

The most commonly used functions are re-exported at the top level, e.g.
``riskmetrics.performance_summary(...)`` works without importing the
submodule directly.
"""

from .periods import get_date_ranges
from .returns import (
    annualize_return,
    cumulative_growth,
    monthly_returns_table,
    quarterly_returns_table,
    to_log_returns,
    to_simple_returns,
    total_return,
)
from .risk_ratios import (
    autocorrelation,
    calmar_ratio,
    downside_volatility,
    drawdown_recovery_time,
    drawdown_series,
    excess_kurtosis,
    max_drawdown,
    omega_ratio,
    parametric_var,
    performance_summary,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    volatility,
    win_rate,
)
from .beta import (
    beta,
    beta_table,
    concentration_top_n,
    correlation_matrix,
    factor_exposure_summary,
    marginal_contribution_to_risk,
)
from .exposure import (
    daily_long_short_imbalance,
    long_short_gross_net,
    position_counts,
    weighted_portfolio_return,
)
from .liquidity import (
    days_to_liquidate,
    liquidity_bucket,
    liquidity_distribution,
    rolling_average_daily_volume,
)
from .options import (
    black_scholes_greeks,
    black_scholes_price,
    price_shock_pnl,
    price_shock_summary,
)
from .charts import (
    plot_drawdown,
    plot_growth_curve,
    plot_period_returns_bar,
    plot_rolling_volatility,
    save_figure,
)

__version__ = "0.1.0"

__all__ = [
    "get_date_ranges",
    "annualize_return",
    "cumulative_growth",
    "monthly_returns_table",
    "quarterly_returns_table",
    "to_log_returns",
    "to_simple_returns",
    "total_return",
    "autocorrelation",
    "calmar_ratio",
    "downside_volatility",
    "drawdown_recovery_time",
    "drawdown_series",
    "excess_kurtosis",
    "max_drawdown",
    "omega_ratio",
    "parametric_var",
    "performance_summary",
    "sharpe_ratio",
    "skewness",
    "sortino_ratio",
    "volatility",
    "win_rate",
    "beta",
    "beta_table",
    "concentration_top_n",
    "correlation_matrix",
    "factor_exposure_summary",
    "marginal_contribution_to_risk",
    "daily_long_short_imbalance",
    "long_short_gross_net",
    "position_counts",
    "weighted_portfolio_return",
    "days_to_liquidate",
    "liquidity_bucket",
    "liquidity_distribution",
    "rolling_average_daily_volume",
    "black_scholes_greeks",
    "black_scholes_price",
    "price_shock_pnl",
    "price_shock_summary",
    "plot_drawdown",
    "plot_growth_curve",
    "plot_period_returns_bar",
    "plot_rolling_volatility",
    "save_figure",
]
