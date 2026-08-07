# riskmetrics

A toolkit of ready-made calculations for measuring investment performance
and risk — the kind of numbers you'd see in a fund's monthly report
(returns, volatility, Sharpe ratio, drawdown, beta, exposure, liquidity,
options risk) — packaged as plain Python functions anyone can call on
their own data.

You do **not** need to know how these formulas work internally to use
this package. You just need to know:

1. What shape of data each function wants (usually: "a column of daily
   returns" or "a table of positions").
2. What it hands back.

This README is written for that purpose. Every function below has a
plain-English description, a picture of what you feed it, and a picture
of what you get back.

> This package was extracted from a hedge fund's internal reporting
> pipeline and rewritten to be generic — nothing here talks to a
> database, knows about "portfolios" or "strategies", or assumes a
> particular asset class. It works the same whether your "portfolio" is
> a single stock, a mutual fund, a crypto wallet, or a spreadsheet of
> made-up numbers for a class project.

---

## Table of contents

1. [Before you start: the two data shapes you'll use everywhere](#before-you-start-the-two-data-shapes-youll-use-everywhere)
2. [Installing it](#installing-it)
3. [Five-minute cookbook](#five-minute-cookbook)
4. [Glossary — plain-English definitions of the finance terms](#glossary--plain-english-definitions-of-the-finance-terms)
5. [Function reference, module by module](#function-reference-module-by-module)
   - [`periods` — standard reporting windows](#periods--standard-reporting-windows)
   - [`returns` — turning daily returns into headline numbers](#returns--turning-daily-returns-into-headline-numbers)
   - [`risk_ratios` — how risky was it, and was the risk worth it](#risk_ratios--how-risky-was-it-and-was-the-risk-worth-it)
   - [`beta` — how correlated/concentrated is the risk](#beta--how-correlatedconcentrated-is-the-risk)
   - [`exposure` — how much is long, how much is short](#exposure--how-much-is-long-how-much-is-short)
   - [`liquidity` — how fast could you sell it](#liquidity--how-fast-could-you-sell-it)
   - [`options` — options pricing and "what if the market moved" scenarios](#options--options-pricing-and-what-if-the-market-moved-scenarios)
   - [`charts` — turning any of the above into a picture](#charts--turning-any-of-the-above-into-a-picture)
6. [Common mistakes and gotchas](#common-mistakes-and-gotchas)

---

## Before you start: the two data shapes you'll use everywhere

Almost every function in this package wants one of two things:

### Shape 1: "a series of returns"

A list of dates, each with a return for that day (or week, or month —
whatever period you're tracking). In Excel terms, two columns: **Date**
and **Return**. In Python, this is a `pandas.Series` with dates as the
index.

| Date       | Return  |
|------------|---------|
| 2025-01-02 | 0.008   | ← up 0.8% that day
| 2025-01-03 | -0.012  | ← down 1.2% that day
| 2025-01-06 | 0.003   |

**Important:** returns are decimals, not percentages. "Up 1%" is
written as `0.01`, not `1`. If your spreadsheet has percentages like
`1.5`, divide by 100 first.

```python
import pandas as pd

returns = pd.Series(
    [0.008, -0.012, 0.003],
    index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
)
```

### Shape 2: "a table of positions"

A list of positions/holdings, one row each, with whatever columns a
given function needs (e.g. a size, an average daily trading volume, an
option strike price). This is a `pandas.DataFrame` — think of it exactly
like a spreadsheet with headers.

| symbol | notional | adv_notional |
|--------|----------|---------------|
| AAPL   | 500000   | 2000000       |
| TSLA   | -300000  | 1500000       |

```python
import pandas as pd

positions = pd.DataFrame({
    "symbol": ["AAPL", "TSLA"],
    "notional": [500_000, -300_000],
    "adv_notional": [2_000_000, 1_500_000],
})
```

Every function's reference entry below tells you exactly which columns
it needs and what it calls them (you can always rename your own columns
to match, or pass the actual column name as an argument).

---

## Installing it

```bash
cd riskmetrics
pip install -e .
```

That installs the package (and its dependencies — pandas, numpy, scipy,
matplotlib) so you can `import riskmetrics` from anywhere. If you don't
want to install it, you can also just run scripts from inside this
`riskmetrics/` directory with `PYTHONPATH=.` set.

To confirm it worked and see every function in action on made-up data:

```bash
python examples/quickstart.py
```

This prints every metric with a label, and saves four example charts
into `examples/output/`.

---

## Five-minute cookbook

You have a spreadsheet of daily returns for something (a fund, a stock,
a strategy) and want the headline numbers. Load it into a
`pandas.Series` indexed by date (see Shape 1 above), then:

```python
import riskmetrics as rm

# One call gets you the whole "report row": total return, annualized
# return, volatility, Sharpe, Sortino, Calmar, max drawdown, and more.
summary = rm.performance_summary(daily_returns, risk_free_rate=0.04)
print(summary)

# A classic year-by-month return table (rows = years, columns = months + YTD).
table = rm.monthly_returns_table(daily_returns)
print(table)

# A "growth of $1" chart, saved to a PNG.
growth = rm.cumulative_growth(daily_returns)
fig = rm.plot_growth_curve(growth, title="My Portfolio")
rm.save_figure(fig, "growth.png")
```

That's it for the most common use case. Everything else in this README
is for more specific questions ("how concentrated is my risk", "how fast
could I sell this book", "what happens to my options if the market drops
5%").

---

## Glossary — plain-English definitions of the finance terms

| Term | What it means |
|------|----------------|
| **Return** | How much something gained or lost over a period, as a fraction (0.01 = up 1%). |
| **Volatility** | How much returns bounce around. Higher = more unpredictable, day to day. |
| **Annualized** | Scaled up to "what this would look like over a full year" so you can compare a 3-month number to a 3-year number fairly. |
| **Sharpe ratio** | Return earned per unit of total risk taken. Higher is better; above 1.0 is generally considered good, above 2.0 very good (rules of thumb vary by strategy). |
| **Sortino ratio** | Like Sharpe, but only counts the downside — it doesn't penalize you for volatility that comes from gains. |
| **Calmar ratio** | Return earned per unit of the worst peak-to-trough loss. Punishes deep drawdowns specifically. |
| **Omega ratio** | Ratio of total gains to total losses. Above 1 means gains outweighed losses. |
| **Drawdown** | How far below its previous high point something has fallen. "-15% drawdown" means it's 15% below its peak value. |
| **Max drawdown** | The single worst drawdown over the whole period — the deepest hole it ever fell into. |
| **Recovery time** | How long it took to climb back out of the worst hole and reach a new high. |
| **Value at Risk (VaR)** | A rough estimate of "how much could I lose on a bad-but-not-catastrophic day," at a given confidence level (e.g. 95%). |
| **Skewness** | Whether extreme moves tend to be worse on the downside (negative skew) or upside (positive skew). |
| **Kurtosis** | Whether extreme moves happen more often than a "normal" bell curve would predict (fat tails). |
| **Beta** | How much something tends to move when a benchmark (like a market index) moves. Beta of 1.0 means it moves in lockstep; 0 means no relationship; negative means it moves opposite. |
| **Correlation** | How closely two things move together, from -1 (perfectly opposite) to +1 (perfectly together). |
| **Marginal contribution to risk** | Out of all the risk in a portfolio, how much of it comes from this one position. Useful for finding "hidden" concentration. |
| **Long / short / gross / net exposure** | Long = bets that something will go up. Short = bets it will go down. Gross = long + short (total money at risk). Net = long - short (which direction you're actually leaning). |
| **NAV (Net Asset Value)** | The total value of the fund/portfolio — the denominator you divide exposures by to get a percentage. |
| **ADV (Average Daily Volume)** | How much of something typically trades in a day — a measure of how liquid (easy to buy/sell) it is. |
| **Days to liquidate** | How many trading days it would take to sell a position without trading more than some % of its normal daily volume. |
| **Greeks (delta, gamma, vega, theta)** | Standard measures of how an option's price reacts to the underlying moving (delta, gamma), volatility changing (vega), or time passing (theta). |
| **Price shock** | "What if the underlying price suddenly moved X%?" — a stress-test scenario. |

---

## Function reference, module by module

Every function is available directly as `riskmetrics.function_name(...)`
(you don't need to remember which submodule it lives in, though the
headers below tell you for reference).

### `periods` — standard reporting windows

Almost every risk report slices history into the same handful of
windows: "this month so far", "trailing 12 months", "trailing 36
months", "since we started". This one function builds all of those at
once so you don't have to do calendar math by hand.

#### `get_date_ranges(as_of, inception_date=None, trailing_months=None)`

- **What it tells you:** the start/end dates for MTD, 12M, 36M, and (if
  you give it a start date) ITD ("Inception To Date" — since the very
  beginning).
- **You give it:**
  - `as_of` — the report's cut-off date. Accepts `"2025-01-31"`,
    `"20250131"`, or just `"202501"` (which is treated as the last day
    of January).
  - `inception_date` *(optional)* — when the fund/strategy started, so
    an `"ITD"` window can be included.
  - `trailing_months` *(optional)* — customize which trailing windows
    you want, e.g. `{"6M": 6, "24M": 24}` instead of the default 12M/36M.
- **You get back:** a dictionary like:
  ```python
  {
      "MTD": (date(2025, 1, 1), date(2025, 1, 31)),
      "12M": (date(2024, 2, 1), date(2025, 1, 31)),
      "36M": (date(2022, 2, 1), date(2025, 1, 31)),
      "ITD": (date(2023, 3, 22), date(2025, 1, 31)),
  }
  ```
- **Example:**
  ```python
  windows = rm.get_date_ranges(as_of="2025-01-31", inception_date="2023-03-22")
  start, end = windows["12M"]
  last_12_months = daily_returns[(daily_returns.index.date >= start) & (daily_returns.index.date <= end)]
  ```

---

### `returns` — turning daily returns into headline numbers

| Function | What it tells you | You give it | You get back |
|----------|--------------------|--------------|----------------|
| `total_return(returns)` | The single compounded return over the whole series (e.g. "+12.4% over the year"). | A returns series. | One number, as a decimal (`0.124`). |
| `annualize_return(total_return_value, num_periods, periods_per_year=252)` | What a return "would look like" scaled to a full year, so you can compare a 3-month return to a 3-year one fairly. | A total return, how many periods it covered, and how many periods make up a year (252 trading days is the common default; use 12 for monthly data). | One number, as a decimal. |
| `cumulative_growth(returns)` | The running "growth of $1" curve over time — feeds directly into `plot_growth_curve`. | A returns series. | A series, same dates, each value is the cumulative return up to that point. |
| `monthly_returns_table(returns)` | The classic tearsheet table: rows = years, columns = Jan-Dec + YTD, values = % return that month. | A daily (or any sub-monthly) returns series. | A table (DataFrame), values already in %. |
| `quarterly_returns_table(returns)` | Same idea, but by calendar quarter (e.g. "2025-Q1"). | A returns series. | A series indexed by quarter label, values in %. |
| `to_log_returns(returns)` / `to_simple_returns(returns)` | Converts between "simple" returns (0.01 = +1%) and "log" returns (used internally for compounding math; you rarely need to call this yourself). | A returns series. | A returns series of the other kind. |

**Note on `min_periods_for_annualization`:** many desks avoid
annualizing a return computed from only a few weeks of data, because
scaling a short window up to "a full year" can be misleading. Pass
`min_periods_for_annualization=252` (or whatever your minimum is) to
`annualize_return` to get the un-annualized number back instead when the
window is too short. `performance_summary` (below) exposes the same
option.

**Example — building a monthly table:**
```python
table = rm.monthly_returns_table(daily_returns)
#       JAN   FEB   MAR  ...  YTD
# 2024  1.2  -0.4   2.1  ...  8.9
# 2025  0.8   1.1  -0.6  ...  1.3
```

---

### `risk_ratios` — how risky was it, and was the risk worth it

This is the heart of the package. Every function here takes **one
returns series** and answers a specific "how good/bad was this"
question. If you only use one function from this whole package, use
`performance_summary` — it runs everything below in one call.

#### `performance_summary(returns, risk_free_rate=0.0, periods_per_year=252)`

- **What it tells you:** every metric in this section, at once, as one
  dictionary — the "give me the whole report row" function.
- **You give it:** a returns series. Optionally, an annualized
  risk-free rate (e.g. `0.04` for 4%, used by Sharpe/Sortino) and how
  many periods make up a year for your data.
- **You get back:** a dictionary with keys `total_return_pct`,
  `annualized_return_pct`, `annualized_volatility_pct`,
  `downside_volatility_pct`, `sharpe_ratio`, `sortino_ratio`,
  `calmar_ratio`, `omega_ratio`, `skewness`, `excess_kurtosis`,
  `autocorrelation`, `parametric_var_95_pct`, `max_drawdown_pct`,
  `drawdown_recovery_periods`, `win_rate_pct`, `best_period_pct`,
  `worst_period_pct`.

The individual functions behind each of those keys, if you want them one
at a time or with different settings:

| Function | What it tells you | Output |
|----------|--------------------|--------|
| `volatility(returns, periods_per_year=252)` | How much returns bounce around, scaled to a year. | Decimal (multiply by 100 for %). |
| `downside_volatility(returns, ..., threshold=0.0)` | Same, but only counting days below `threshold` — the "bad day" volatility. | Decimal. |
| `sharpe_ratio(returns, risk_free_rate=0.0, ...)` | Return per unit of total risk. Higher = better. | Plain number (not a %). |
| `sortino_ratio(returns, risk_free_rate=0.0, ...)` | Return per unit of *downside* risk only. | Plain number. |
| `calmar_ratio(returns, ...)` | Return per unit of the worst drawdown. | Plain number. |
| `omega_ratio(returns, threshold=0.0)` | Total gains ÷ total losses (relative to `threshold`). | Plain number; >1 is good. |
| `skewness(returns)` | Is the distribution lopsided towards big losses (negative) or big gains (positive)? | Plain number; 0 = symmetric. |
| `excess_kurtosis(returns)` | Are extreme days more common than a normal bell curve predicts? | Plain number; 0 = normal, positive = fat tails. |
| `autocorrelation(returns, lag=1)` | Does today's return tell you anything about tomorrow's? | Plain number, -1 to 1. |
| `parametric_var(returns, confidence=0.95)` | A rough "how bad could a single bad-but-not-catastrophic day be" estimate. | Decimal, a positive loss magnitude. |
| `drawdown_series(returns)` | The full underwater curve — how far below the peak, at every point in time. | A series (feeds `plot_drawdown`). |
| `max_drawdown(returns)` | The single deepest hole, over the whole series. | Negative decimal (e.g. `-0.18` = -18%). |
| `drawdown_recovery_time(returns)` | How many periods it took to climb out of the worst hole. | Integer, or `None` if it never fully recovered. |
| `win_rate(returns)` | What % of traded days were positive. | Number 0-100. |
| `best_period(returns)` / `worst_period(returns)` | The single best/worst day. | % number. |

**Example:**
```python
summary = rm.performance_summary(daily_returns, risk_free_rate=0.04)
print(f"Sharpe: {summary['sharpe_ratio']:.2f}")
print(f"Worst drawdown: {summary['max_drawdown_pct']:.1f}%")
```

---

### `beta` — how correlated/concentrated is the risk

#### `beta(asset_returns, benchmark_returns)`

- **What it tells you:** how much `asset_returns` tends to move for a
  1-unit move in `benchmark_returns`. Beta of 1.5 means "when the
  benchmark moves 1%, this tends to move 1.5%".
- **You give it:** two returns series (e.g. your fund and a market
  index). They're automatically lined up by date.
- **You get back:** one number.

#### `beta_table(asset_returns, benchmarks, windows=None)`

- **What it tells you:** beta to *several* benchmarks, over *several*
  time windows, as one table — the classic "Beta to Equities / Bonds /
  Credit" report block.
- **You give it:** your returns series, a table (`DataFrame`) with one
  column per benchmark, and optionally the output of
  `get_date_ranges(...)` to control which windows are shown.
- **You get back:** a table, rows = `"Beta to <benchmark>"`, columns =
  window labels.

#### `correlation_matrix(returns)`

- **What it tells you:** how closely every pair of things in your table
  moves together.
- **You give it:** a table with one returns column per thing you're
  comparing (e.g. one column per strategy, or one per stock).
- **You get back:** a square table of correlations (-1 to +1).

#### `marginal_contribution_to_risk(component_returns, weights, portfolio_returns=None)`

- **What it tells you:** out of all your portfolio's risk, how much
  comes from each individual position. This is how you find the "one
  small position that's secretly your biggest risk" — a position can be
  tiny in dollar terms but highly correlated with everything else, which
  makes it a bigger risk driver than its size suggests.
- **You give it:**
  - `component_returns` — a table, one column per position/ticker, one
    row per date, values = that position's daily return.
  - `weights` — a table with the same column names, one row per date,
    values = how much of the portfolio that position represented that
    day (doesn't need to add up to exactly 1).
  - `portfolio_returns` *(optional)* — the portfolio's own overall daily
    return. If you don't have it, this function will compute it for you
    from `component_returns` and `weights`.
- **You get back:** a table, one row per position, columns `beta`,
  `weight`, `mc`, `mc_pct` — sorted so the biggest risk contributors are
  at the top.

#### `concentration_top_n(mc_pct, n_values=(10, 50))`

- **What it tells you:** "what % of total risk comes from just my top 10
  (or top 50) positions" — a quick concentration check.
- **You give it:** the `mc_pct` column from
  `marginal_contribution_to_risk`.
- **You get back:** a dictionary like `{10: 42.5, 50: 78.1}` — top 10
  positions are 42.5% of total risk, top 50 are 78.1%.

#### `factor_exposure_summary(exposure, windows, top_n=5, ...)`

- **What it tells you:** if you have a table of exposure to different
  risk "factors" (e.g. from a factor/Barra-style model) over time, this
  sums it into your reporting windows and keeps only the biggest ones.
- **You give it:** a table with a date column, a factor-name column, and
  an exposure-amount column (column names are configurable), plus
  reporting windows from `get_date_ranges`.
- **You get back:** a table, rows = factor name, columns = window
  labels, values = summed exposure in that window — restricted to the
  `top_n` biggest factors.

---

### `exposure` — how much is long, how much is short

#### `long_short_gross_net(exposure, nav=None)`

- **What it tells you:** the standard "book summary": how much is long,
  how much is short, total money at risk (gross), and net directional
  lean (net).
- **You give it:** a series of position sizes (positive = long, negative
  = short — e.g. dollar amounts), and optionally the fund's NAV to get
  percentages instead of just dollars.
- **You get back:** a dictionary: `{"long": ..., "short": ...,
  "gross": ..., "net": ...}` in dollars, plus `"long_pct"`,
  `"short_pct"`, `"gross_pct"`, `"net_pct"` if you gave a `nav`.

#### `position_counts(exposure)`

- **What it tells you:** how many positions are long vs. short (a count,
  not a dollar amount).
- **You give it:** the same kind of series as above.
- **You get back:** `{"long": 42, "short": 17}`.

#### `daily_long_short_imbalance(exposure, nav, date_col="date", exposure_col="exposure", group_col=None)`

- **What it tells you:** on a *typical* day, how skewed long vs. short
  is the book (or a slice of it, like one sector) — averaged over many
  days rather than one snapshot, which is more representative.
- **You give it:** a table with one row per position per date (a date
  column and an exposure column), a NAV to divide by, and optionally a
  grouping column (e.g. sector) to break the result out by group.
- **You get back:** one number (average % of NAV) if no grouping, or a
  series (one number per group, biggest imbalance first) if you gave a
  `group_col`.

#### `weighted_portfolio_return(weights, component_returns)`

- **What it tells you:** reconstructs an overall portfolio return series
  from individual position weights and returns — useful when you have
  position-level data but not a ready-made portfolio-level return.
- **You give it:** two tables with matching column names (one column per
  position): one of weights, one of returns, both indexed by date.
- **You get back:** a single returns series, one value per date.

---

### `liquidity` — how fast could you sell it

#### `rolling_average_daily_volume(volume, window=21, stat="median")`

- **What it tells you:** a security's "typical" daily trading volume,
  smoothed over a rolling window (21 trading days ≈ 1 month is standard)
  so one unusually busy or quiet day doesn't distort the estimate.
- **You give it:** a series of daily trading volume (or dollar volume),
  indexed by date.
- **You get back:** a series of the same length, the rolling
  median (or mean) volume.

#### `days_to_liquidate(position_notional, adv_notional, participation_rate=0.20)`

- **What it tells you:** how many trading days it would take to fully
  exit a position if you only trade up to `participation_rate` (e.g.
  20%) of the security's normal daily volume, so you don't move the
  price against yourself.
- **You give it:** the position's size and the security's average daily
  volume, in the same units (both in dollars, or both in shares).
- **You get back:** one number — days.

#### `liquidity_bucket(days)`

- **What it tells you:** turns a "days to liquidate" number into a
  human label like `"2d-7d"` or `"8d-30d"`.
- **You give it:** a number of days (e.g. from `days_to_liquidate`).
- **You get back:** a string label.

#### `liquidity_distribution(positions, notional_col, adv_col, participation_rates=(0.20, 0.05))`

- **What it tells you:** the headline liquidity summary for a whole
  book: "72% of the portfolio could be sold within 30 days at a 20%
  participation rate."
- **You give it:** a table with one row per position, a column for
  position size and a column for that position's ADV.
- **You get back:** a table, rows = participation rate (e.g. `"20%"`,
  `"5%"`), columns = time buckets, values = the **cumulative** % of the
  book liquidatable within that bucket or faster.

---

### `options` — options pricing and "what if the market moved" scenarios

These functions use the standard Black-Scholes formula — a simplified,
textbook model of options pricing. It's a reasonable estimate for a
scenario table, not a substitute for a professional trading system.

#### `black_scholes_price(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type)`

- **What it tells you:** the theoretical fair value of a call or put
  option.
- **You give it:** the underlying's current price, the option's strike
  price, time left until expiry (**in years** — 30 days = `30/365`), an
  annualized risk-free interest rate, the option's implied volatility,
  and `"call"` or `"put"`. Every argument can be a single number or a
  list/column of numbers (one option per row).
- **You get back:** the price (or a list of prices, matching your
  input).

#### `black_scholes_greeks(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type)`

- **What it tells you:** delta, gamma, vega, and theta — the standard
  sensitivities of the option's price to the underlying moving (delta,
  gamma), volatility changing (vega), or time passing (theta).
- **You give it:** the same inputs as `black_scholes_price`.
- **You get back:** a dictionary `{"delta": ..., "gamma": ..., "vega":
  ..., "theta": ...}`, each a number (or array matching your input).

#### `price_shock_pnl(positions, shocks=(-0.05, -0.02, -0.01, 0.01, 0.02, 0.05), risk_free_rate=0.0, ...)`

- **What it tells you:** for every option position, "if the underlying
  suddenly moved +2% (or -5%, etc.), how much money would this position
  gain or lose right now."
- **You give it:** a table with one row per option position — spot
  price, strike, time to expiry, volatility, option type, position size
  (`quantity`, positive=long/negative=short), and contract multiplier
  (column names are configurable if yours differ). You can override the
  default shock list.
- **You get back:** a copy of your table with one extra column per
  shock (e.g. `"pnl_shock_+2%"`), showing the dollar impact per position.

#### `price_shock_summary(shocked_positions, shock_columns, nav, date_col=None)`

- **What it tells you:** rolls the position-level shock table above up
  into "what % of NAV would the whole book gain/lose under each
  scenario" — the number that actually goes in a report.
- **You give it:** the output of `price_shock_pnl`, which shock columns
  to summarize, and the fund's NAV. If your positions have a date column
  (multiple days of history), pass it to get the *average daily* impact
  instead of a single one-time total.
- **You get back:** a one-row table, one column per shock, in % of NAV.

---

### `charts` — turning any of the above into a picture

Every `plot_*` function returns a matplotlib `Figure` — nothing is shown
or saved automatically. Call `save_figure` to write it to a PNG, or
`fig.savefig(...)` yourself, or hand it to a notebook/PDF/Dash app.

| Function | What it draws | You give it |
|----------|-----------------|--------------|
| `plot_growth_curve(cumulative_growth, title=None)` | A "growth of $1" line chart. | The output of `returns.cumulative_growth(...)`. |
| `plot_period_returns_bar(returns, title=None)` | A bar chart of daily (or periodic) returns. | A returns series. |
| `plot_drawdown(returns, title=None)` | The "underwater" chart — how far below the peak, over time. | A returns series. |
| `plot_rolling_volatility(returns, window=252, title=None)` | A line chart of rolling annualized volatility. | A returns series. |
| `save_figure(fig, path, dpi=150)` | Writes any of the above to a PNG file, tightly cropped. | A `Figure` and a destination file path. |

**Example:**
```python
growth = rm.cumulative_growth(daily_returns)
fig = rm.plot_growth_curve(growth, title="Fund Growth")
rm.save_figure(fig, "reports/fund_growth.png")
```

---

## Common mistakes and gotchas

- **Decimals, not percentages.** A return of "up 2%" must be given as
  `0.02`, not `2`. If your data is in percentage form, divide by 100
  first (`returns_pct / 100`).
- **`periods_per_year` must match your data's frequency.** Daily
  trading-day data → 252 (the default). Monthly data → 12. Weekly → 52.
  Using the wrong value will silently give you a wrong-but-plausible-
  looking annualized number.
- **Dates should be real dates.** Every returns series needs an index
  that's a real calendar date (a `pandas.DatetimeIndex`, or a column of
  date strings you convert with `pd.to_datetime(...)` first) — plain
  integers like `1, 2, 3, ...` won't work with the windowing functions.
- **Column names are configurable, not fixed.** Most table-based
  functions (`liquidity_distribution`, `price_shock_pnl`,
  `daily_long_short_imbalance`, `factor_exposure_summary`) let you pass
  your own column names (e.g. `notional_col="size_usd"`) instead of
  renaming your data to match a hardcoded default.
- **`nav` is a single number, not a time series.** Functions that take a
  `nav` argument want one representative value (e.g. start-of-month
  NAV), not a day-by-day series. If your NAV changes a lot within the
  period you're measuring, consider using an average.
- **Missing data is usually fine.** Most functions handle `NaN`/missing
  values sensibly (e.g. a stock that didn't trade on a given day is
  simply excluded from that day's calculation) — but always sanity-check
  your output against a number you can compute by hand for one simple
  case before trusting it on real data.
