# stock_research

Statistics for selling **out-of-the-money (OTM) call options** at market prices on
ETFs and companies with a market cap / assets-under-management of **≥ $1 billion**.

It pulls live option chains from Yahoo Finance (`yfinance`), computes the metrics that
matter for income-oriented covered-call / OTM-call writing, and gives you two views:

- a **universe screener** that ranks tradable OTM calls across a watchlist into a CSV, and
- a **per-ticker deep dive** that dumps the full stats grid across strikes/expiries (with optional charts).

## Metrics computed

| Metric | Meaning |
| --- | --- |
| **Annualized premium yield** | `premium / stock_price`, scaled to a year by days-to-expiry. The static income return if the call expires worthless. **This is the default ranking** — the top of the output is the highest-yielding contracts. |
| **Score** | `annual_yield × prob_otm` — the annualized yield weighted by the chance you actually keep it. Use `--sort score` to rank by this risk-adjusted view instead of raw yield. |
| **If-called yield** | Return if the stock is called away at the strike: `(premium + (strike − price)) / price`, annualized. |
| **Probability OTM** | Lognormal `P(S_T < K)` from implied vol — the chance you keep the shares and the full premium. |
| **Delta** | Black–Scholes call delta (≈ probability of assignment). |
| **Downside cushion** | `premium / stock_price` — how far the stock can fall before the premium stops covering it. |
| **Breakeven** | `stock_price − premium`, your effective cost basis. |
| **IV / HV** | Implied vol vs. trailing realized (historical) vol — is the option rich or cheap? |
| **Liquidity** | Bid/ask spread, open interest, and volume filters so only tradable strikes survive. |

> ⚠️ Yahoo data is delayed and occasionally incomplete; mid-prices approximate where you'd
> actually fill. Treat outputs as research, not trade signals. Not investment advice.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -e .
```

## Usage

```bash
# Screen the default universe (config/universe.yaml) for OTM calls 20–50 days out,
# 2–15% out of the money, ranked by annualized yield. Writes output/screen_*.csv.
stock-research screen

# Tighten the screen
stock-research screen --min-dte 25 --max-dte 45 --min-otm 0.03 --max-otm 0.10 --top 50

# Rank by the risk-adjusted score (yield × probability of expiring OTM) instead of raw yield
stock-research screen --sort score

# Deep dive on one ticker (prints a grid; --charts saves PNGs to output/)
stock-research deepdive MSFT --charts
```

Or run the modules directly without installing:

```bash
python -m stock_research.cli screen
python -m stock_research.cli deepdive QQQ
```

## Configuration

- **`config/universe.yaml`** — the watchlist (stocks + ETFs). The screener re-verifies each
  ticker's market cap / AUM ≥ $1B at runtime and drops anything below.
- **`config/settings.yaml`** — defaults: risk-free rate, DTE window, OTM band, and liquidity thresholds.

## Layout

```
src/stock_research/
  blackscholes.py   # d1/d2, call delta, prob-OTM, BS price
  data.py           # yfinance access: price, market cap/AUM, option chains, realized vol
  metrics.py        # per-contract stat computation
  screener.py       # universe scan -> ranked CSV
  deepdive.py       # single-ticker grid + charts
  cli.py            # `screen` / `deepdive` subcommands
config/             # universe.yaml, settings.yaml
tests/              # unit tests for the math
```
