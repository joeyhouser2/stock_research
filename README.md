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
| **Underlying value** (`--value`) | P/E, forward P/E, PEG, price/book, EV/EBITDA, profit margin, ROE, position vs. 52-week high/low, and analyst-target upside — because writing calls can leave you holding the shares. |

### Weekly vs. monthly expirations

Every contract is tagged `exp_type` = `monthly` (the standard 3rd-Friday expiry) or
`weekly` (anything else). Filter with `--weekly` or `--monthly`. `--weekly` also
auto-narrows the DTE window to 1–14 days (override with `--min-dte`/`--max-dte`).

### Value & risk filters

| Flag | Effect |
| --- | --- |
| `--max-pe N` | Keep only underlyings with trailing P/E ≤ N (implies `--value`). |
| `--max-forward-pe N` | Same, on forward P/E. |
| `--max-peg N` | Same, on PEG. |
| `--min-prob-otm F` | Keep only contracts with ≥ F probability of expiring OTM (e.g. `0.70`). |

A P/E cap drops names whose P/E Yahoo doesn't report (most ETFs, occasionally
loss-making companies) — if you ask for "good P/E" we won't pass through names whose
P/E we can't see.

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

# Weekly options only (auto-narrows to 1–14 DTE), with underlying value metrics
stock-research screen --weekly --value

# Standard monthly (3rd-Friday) expirations only
stock-research screen --monthly

# Best weekly OTM calls on reasonably-valued names (trailing P/E ≤ 20),
# ranked by risk-adjusted score. --max-pe implies --value.
stock-research screen --weekly --max-pe 20 --sort score

# Cap assignment risk: only strikes with ≥70% chance of expiring worthless
stock-research screen --weekly --min-prob-otm 0.70

# Screen the ENTIRE weeklies universe (every US symbol with weekly options,
# ~680 names) instead of the watchlist, gated to market cap / AUM ≥ $1B.
stock-research fetch-weeklys                 # download the Cboe list (once)
stock-research screen --weeklys --weekly --max-pe 25 --sort score

# Faster first pass: only the 150 largest qualifying names, gentle on rate limits
stock-research screen --weeklys --max-tickers 150 --throttle 0.3

# Deep dive on one ticker (prints a grid; --charts saves PNGs to output/)
stock-research deepdive MSFT --charts
```

Or run the modules directly without installing:

```bash
python -m stock_research.cli screen
python -m stock_research.cli deepdive QQQ
```

## Universe: watchlist vs. the whole weeklies market

By default the screener scans **`config/universe.yaml`** — a hand-picked watchlist
(~75 ETFs + large-caps). Pass **`--weeklys`** to instead screen *every* US symbol that
has weekly options, sourced from Cboe's official
[Available Weeklys](https://www.cboe.com/available_weeklys/) list (~680 names):

1. `fetch-weeklys` downloads + parses the Cboe CSV into `config/weeklys.csv`
   (a checked-in copy is the offline fallback).
2. A cached **market-cap gate** keeps only names ≥ $1B. Sizes are cached to
   `cache/marketcaps.json` for 7 days (`--cache-ttl-days`), so the slow, rate-limited
   pass runs once and reruns are fast — only the survivors' option chains are re-fetched.
3. `--max-tickers N` caps the scan at the N largest qualifying names; `--throttle SECS`
   paces the size lookups to avoid Yahoo rate limits.

> The first full `--weeklys` run does ~680 size lookups (a few minutes, occasionally
> throttled by Yahoo). After that the cache makes it quick. Use `--max-tickers` /
> `--throttle` on the first run if you hit rate limiting.

## Configuration

- **`config/universe.yaml`** — the default watchlist (stocks + ETFs). The screener
  re-verifies each ticker's market cap / AUM ≥ $1B at runtime and drops anything below.
- **`config/weeklys.csv`** — the cached Cboe weeklies universe (used by `--weeklys`).
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
