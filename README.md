python -m venv .venv
.venv\Scripts\Activate
pip install -e .
stock-research screen --weekly --max-pe 25 --sort score
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
| **Score+** (`score_adj`) | `score × IV/HV` (capped) — the risk-adjusted yield *also* weighted by how rich the option's vol is. Best single "bang-for-buck" rank; pair with `--min-iv-hv`. |
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
| `--min-iv-hv F` | Rich-vol gate: keep only contracts with IV/HV ≥ F (e.g. `1.2`) — i.e. you're paid more than the stock actually moves. Pair with `--sort score_adj`. |

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

# Deep dive with Monte-Carlo probabilities per strike: sim_otm / sim_touch next to
# the Black-Scholes prob_otm (fat-tailed, valuation-drift-aware)
stock-research deepdive AAPL --simulate

# Rank the underlyings themselves by simulated risk/return (drift, vol, Sharpe,
# touch odds, VaR/CVaR/drawdown) -> output/risk_*.csv
stock-research riskscan --horizon 45 --sort sharpe
stock-research riskscan --weeklys --max-tickers 150 --throttle 0.3   # whole universe

# Best "bang for buck" OTM writes: gate on rich vol, rank by edge-weighted score
stock-research screen --weekly --min-iv-hv 1.2 --sort score_adj

# Rank names as long-equity value buys (cheapness + quality + analyst upside)
stock-research valuescan --sort value_score

# Value picker: smaller names (cap band) passing multiple cheap-and-quality gates
stock-research valuescan --min-market-cap 5e9 --max-market-cap 50e9 \
    --max-pe 20 --max-peg 1.5 --min-roe 0.12 --min-upside 0.10

# Honest odds of BUYING a call (vega-aware P&L distribution)
stock-research longcall AAPL --expiry 2026-07-17 --strike 300

# Log a daily option-chain snapshot to build implied-vol history (schedule this)
stock-research log-chains --tickers AAPL MSFT NVDA

# Download SEC EDGAR point-in-time fundamentals (the value-model data foundation)
SEC_USER_AGENT="Your Name you@email" stock-research fetch-fundamentals AAPL MSFT --save

# Build the as-of feature panel (EDGAR + prices + FRED macro) for a value model
SEC_USER_AGENT="Your Name you@email" stock-research build-panel AAPL MSFT XOM --start 2020-01-01

# Add the simulated assignment/touch columns to the option screen itself
stock-research screen --weekly --simulate --sort score

# Validate the model out-of-sample: are the forecast probabilities calibrated?
stock-research backtest SPY AAPL MSFT XOM --horizon 30 --model garch

# Train the autoregressive TCN generative model (GPU) and simulate with it
stock-research generate NVDA --horizon 30 --target-pct 0.10 --target-pct -0.10

# Train on early history, then score the frozen TCN vs a refit GARCH out-of-sample
stock-research generate AAPL --horizon 30 --compare-baseline

# Monte-Carlo risk & probability: will MSFT touch $250 (or fall to $200) in 45 days?
stock-research simulate MSFT --target 250 --target 200 --horizon 45

# Same, expressed as moves from spot, with the fat-tailed GARCH model (the default)
stock-research simulate NVDA --target-pct 0.10 --target-pct -0.10
```

Or run the modules directly without installing:

```bash
python -m stock_research.cli screen
python -m stock_research.cli deepdive QQQ
```

## Dashboard (web UI)

A Streamlit dashboard wraps every tool across eight tabs — **OTM calls** (rich-vol
gate + edge-weighted score), **Long calls** (vega-aware buy-side P&L), **Value buys**
(value rank + a likely-price simulation for any name), risk scan, deep dive, simulate,
backtest, and the generative / cross-asset models — with the same engine the CLI uses,
plus inline tables, charts (terminal-price histograms, calibration curves), and CSV
download.

```bash
pip install -e ".[ui]"      # installs streamlit
streamlit run app.py        # opens http://localhost:8501
```

Global model settings (risk-free rate, DTE/OTM window, valuation-drift knobs) live
in the sidebar; each tab has a **Run** button so the network/GPU work only fires on
demand, and results persist as you tweak other controls. The GPU is used
automatically for the generative tabs when a CUDA build of torch is installed.

## Risk & probability simulation

`simulate` answers three questions the screener's closed-form `prob_otm` can't, by
running a Monte-Carlo ensemble of price paths (GPU-accelerated when available):

- **Touch / first-passage** — P(the stock reaches a price *at any point* before the
  horizon). This is what "will it hit $X" usually means, and it's materially larger
  than the terminal probability — the relevant number for American-style assignment
  risk. A Broadie–Glasserman–Kou continuity correction makes the daily-step estimate
  approximate true continuous monitoring.
- **Terminal** — P(S_T is above/below a level *at* expiry) — the European assignment
  probability, matching `blackscholes.prob_otm`.
- **Risk** — VaR, CVaR / expected shortfall, and the max-drawdown distribution.

Each probability is printed next to its **GBM closed-form** value as a cross-check;
the gap is the tail risk the lognormal misses.

`deepdive --simulate` brings the same engine to the per-strike grid: it adds
`sim_otm` (simulated P(expires OTM)) and `sim_touch` (P(the strike is touched
before expiry — early-assignment risk)) right next to the Black–Scholes `prob_otm`,
so you can see where the lognormal under-states assignment odds. One ensemble runs
per expiry; the GARCH fit and valuation drift are computed once. `screen --simulate`
adds the same two columns to every ranked contract in the universe screen.

### Universe risk scan (`riskscan`)

Where `screen` ranks option *contracts*, `riskscan` ranks the *underlyings* by their
simulated forward risk/return over a horizon — one row per name, written to
`output/risk_*.csv`:

| Column | Meaning |
| --- | --- |
| `drift` / `sigma` | Annualized valuation drift and realized vol. |
| `sharpe` | `(drift − rf) / sigma` — excess return per unit vol; the default ranking. |
| `prob_up` / `prob_down` | P(touch +X% / −X%) over the horizon (`--target-pct`, default 5%). |
| `prob_term_up` | P(finishing above +X%). |
| `var` / `cvar` | 95% Value-at-Risk and expected shortfall on the holding-period return. |
| `mdd` | Mean max drawdown over the path. |

Risk columns (`var`, `cvar`, `mdd`, `prob_down`) rank ascending (lower = better);
the rest descending. It reuses the same universe options as `screen` (`--tickers`,
`--weeklys`, `--max-tickers`, `--throttle`) and the same drift flags. Each name
needs only a price snapshot and a return history — no option chains — so it's
lighter than a full screen.

### Path models (`--model`)

| Model | What it captures |
| --- | --- |
| `gbm` | Lognormal baseline (matches the closed form). |
| `t` | Fat tails via Student-t innovations. |
| `garch` | **Default.** Volatility clustering + fat tails; GARCH(1,1)-t fit by MLE on trailing returns. Falls back to `t` if the fit fails. |
| `bootstrap` | Non-parametric: resamples historical returns in blocks (`--block`), preserving real autocorrelation / clustering. |

Vol and the GARCH/bootstrap inputs are estimated from `--lookback` trading days of
history. `--paths` sets the ensemble size (default 100k); `--cpu` forces NumPy even
when a GPU is present.

### Drift: valuation-conditioned expected return (`--drift-model`)

Rather than a flat drift, the simulator estimates a forward-looking expected return
from the underlying's fundamentals, so a richly-valued name drifts differently from
a cheap one. The default is the **Grinold–Kroner** decomposition:

> `E[R] = dividend yield + earnings growth + P/E re-rating`

where the re-rating term is mean-reversion of the current P/E toward an anchor
(default: the PEG=1 "fair" multiple, ~100×growth) over `--reversion-years`. Each
component is printed, e.g.:

```
drift: mu=+23.3%/yr [fundamental]  =  income +0.0% + growth +33.3% + reversion -10.0% (P/E 55.0 vs anchor 33.3)
```

| Flag | Effect |
| --- | --- |
| `--drift N` | Fixed annual drift; overrides the model entirely. |
| `--drift-model` | `fixed` (flat rf + equity premium), `fundamental` (default), `analyst` (mean-target-implied), `blend`. |
| `--erp F` | Equity risk premium for the baseline (default 0.045). |
| `--pe-anchor N` | Override the reversion target P/E. |
| `--reversion-years N` | Horizon over which P/E reverts (default 5). |
| `--reversion-shrink F` | Weight 0–1 on the reversion term (default 1). |

> ⚠️ At short horizons (days–weeks) the drift barely moves the distribution — vol
> scales with √t while drift scales with t — so this is a marginal tilt that grows
> in importance with the horizon. Single-name fundamental drift is noisy; treat it
> as a lean, not a forecast. To keep a bad data point from dominating, the earnings-
> growth term is capped at ±25%/yr and the P/E-reversion term at ±15%/yr (a "reversion
> clamped" note prints when it binds). ETFs and loss-making names lack the inputs and
> fall back to the market baseline (rf + equity premium).

### GPU

The simulator uses PyTorch + CUDA when installed and falls back to NumPy on CPU — the
math is identical, only the backend differs. Install the GPU extra with a CUDA build
of torch matching your driver:

```bash
pip install -e ".[gpu]"   # then check torch.cuda.is_available()
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

## Value buys (`valuescan`)

`valuescan` ranks the *underlyings* as long-equity **value buys** — a transparent,
rules-based composite that scores each name cross-sectionally on **cheapness** (P/E,
forward P/E, PEG, EV/EBITDA, P/B; lower = better), **quality** (ROE, profit margin),
and **analyst upside**, blended into a `value_score` in [0, 1] (default weights
50/30/20). A negative P/E is treated as missing, not "cheap".

```bash
stock-research valuescan --sort value_score          # watchlist
stock-research valuescan --weeklys --max-tickers 200  # whole $1B+ universe
```

**Value picker.** Pass a market-cap **band** and hard parameter gates to hunt for
specific setups (e.g. smaller, cheap, quality names): `--min-market-cap` /
`--max-market-cap` plus `--max-pe`, `--max-forward-pe`, `--max-peg`, `--max-pb`,
`--max-ev-ebitda`, `--min-roe`, `--min-margin`, `--min-upside`. Survivors are ranked
by the composite. The watchlist/weeklys universes skew large-cap; add `--sec-universe`
(broad SEC filer list, pair with `--max-tickers`/`--throttle`) to reach smaller names.

> ⚠️ This uses **current** fundamentals, so it's valid for a *live* ranking (today's
> data, today's decision) but is **not backtested or trained** — honestly validating
> a value model needs point-in-time fundamentals (the planned next step). Treat it as
> a disciplined screen, not alpha.

## Long calls (`longcall`)

Selling calls earns the volatility risk premium; **buying** them pays it, so a long
call only makes sense with a directional view or genuinely cheap vol. `longcall`
gives the honest odds: it simulates the joint **(price, volatility)** path with the
GARCH stochastic-vol model and values the call along each path — terminal payoff if
held to expiry, or Black–Scholes with the *simulated* vol for an earlier exit (so
the P&L is **vega-aware**). Output: P(profit), P(2×), P(lose ≥90%), expected return,
and a **cheap/rich** flag from the contract's IV vs realized vol.

```bash
stock-research longcall AAPL --expiry 2026-07-17 --strike 300         # hold to expiry
stock-research longcall NVDA --expiry 2026-08-15 --strike 220 --hold 10  # exit in 10 days
```

> Data note: free sources have no *historical* implied vol, so the IV **level** is
> anchored to the contract's IV observed *now* (today's IV/RV ratio) and evolves with
> the simulated vol process — an explicit calibration, not an IV forecast. To unlock
> a true IV-path model later, `log-chains` appends a daily option-chain snapshot to
> `data/chains/` so per-name IV history accumulates — run it on a schedule.

## Data foundations (scheduled collection)

Two pieces accumulate the point-in-time data that free APIs don't provide, so that
*trained* models (a value model, a true IV-path model) can be built without
look-ahead later:

- **`fetch-fundamentals`** — pulls SEC EDGAR XBRL `companyfacts` (revenue, net income,
  assets, equity, EPS, …), each value tagged with the date it was **filed**, into
  `data/edgar/`. That filing date is what makes it point-in-time. Set
  `SEC_USER_AGENT="Your Name you@email"` (SEC requires a contact); responses are cached.
- **`log-chains`** — appends a daily option-chain snapshot (with implied vol) to
  `data/chains/`, building per-name IV history over time. Schedule it to run daily
  after the close. A ready-made launcher is in [scripts/log_chains.bat](scripts/log_chains.bat);
  on Windows register it with Task Scheduler, e.g.:

  ```powershell
  $a = New-ScheduledTaskAction -Execute "<repo>\scripts\log_chains.bat"
  $t = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Mon,Tue,Wed,Thu,Fri -At 4:15PM
  Register-ScheduledTask -TaskName "stock_research-log-chains" -Action $a -Trigger $t -Force
  ```

- **`build-panel`** — assembles the **as-of feature panel**: one row per (ticker,
  rebalance date) where every feature uses only data available then — fundamentals
  filed by that date (TTM / latest), prices up to it, and macro as-of — plus a
  forward-return label. Features span valuation yields (earnings/book/sales/cash-flow),
  quality (ROE, margin, asset turnover), YoY growth, 12-1 momentum, 3-month realized
  vol, size, and the rate environment. Writes `data/panel/panel_*.csv`.

  ```bash
  SEC_USER_AGENT="Your Name you@email" stock-research build-panel AAPL MSFT XOM --start 2020-01-01
  ```

## Value model (`train-value`)

`train-value` trains a **LightGBM cross-sectional forward-return ranker** on the
panel and validates it **walk-forward with purging** — for each rebalance date it
trains only on rows whose forward-return window ended before that date, so no
overlapping label can leak. Name-specific features are percentile-ranked within
each date (relative cheapness/quality, robust to regime drift); macro is left raw.

```bash
pip install -e ".[value]"     # lightgbm + scikit-learn
SEC_USER_AGENT="Your Name you@email" stock-research train-value \
    AAPL MSFT NVDA JPM XOM KO PG JNJ ... --start 2016-01-01 --horizon-days 126
```

It reports **rank IC** (Spearman of prediction vs realized return per date) with its
t-stat, the top-minus-bottom quantile spread, and a single-factor **baseline** —
because the model has to beat just buying cheap, not just beat zero.

> 📊 **First validated result** (30 large-caps, 2016–2025, 36 dates, 6-mo horizon):
> rank IC **+0.068**, hit-rate 67%, **t-stat +1.55**, quantile spread +1.2%/6mo;
> the earnings-yield baseline was **dead (IC ~0)** over this period (value's lost
> decade), so the model's edge came from combining quality/growth/momentum. Honest
> read: a **promising but not yet statistically significant** signal (t < 2 on a
> small sample), and — importantly — the 30 names are *today's* large-caps, so the
> result carries **survivorship bias**. A trustworthy verdict needs a point-in-time
> universe and more breadth. Treat it as a hypothesis, not an edge.

## Validation (`backtest`)

Before trusting a forecast, check it out-of-sample. `backtest` walks the model
forward through history with **no look-ahead** — at each rebalance date it fits on
the trailing window, simulates the horizon, records the forecast, then waits for
the outcome — and scores the pooled forecast/outcome pairs:

- **Calibration** — reliability table + Brier + ECE for the touch/terminal
  probabilities (when it says 30%, does it happen ~30% of the time?).
- **VaR coverage** — Kupiec proportion-of-failures test: does the realized return
  breach the VaR forecast at the nominal rate?
- **Pinball loss** on the predicted return quantiles — a proper scoring rule, the
  number a better model has to beat.

Windows are non-overlapping by default (`--step` = horizon), so pooled records are
statistically independent. Realized touch is measured on daily closes, so the
simulated touch is read *without* the continuity correction — an apples-to-apples
discrete comparison.

> ⚠️ **What can and can't be validated.** yfinance exposes only *current*
> fundamentals, so the valuation drift can't be reconstructed point-in-time without
> look-ahead. The backtest therefore runs a **flat baseline drift** and validates
> the *price model* (GARCH/t/bootstrap touch, terminal and tail forecasts). In a
> trending sample this makes upside touch look under-predicted — that gap is the
> missing drift, not a broken risk model. Honestly backtesting the valuation edge
> needs a point-in-time fundamentals source (a future addition).

## Generative path model (`generate`)

An **autoregressive TCN** (dilated causal convolutions with a Student-t head)
learns the one-step-ahead conditional return distribution by maximum likelihood,
then samples paths on the GPU. Returns are de-meaned before training, so the net
models the *shock* process and drift stays pluggable (same convention as the other
engines). Sampled paths are wrapped in a `Simulation`, so the learned model plugs
into the same touch / VaR / drawdown readouts and the same `backtest` scoring.

Two honest tests, both training on early history and scoring the **frozen** model
against a **refit GARCH** on the held-out tail (identical windows, same flat drift):

- `generate TICKER --compare-baseline` — single-name.
- `generate --pooled --tickers ...` — one TCN trained on returns **pooled across
  many names** (each standardized by its own scale), then scored per-name. This is
  the data GARCH structurally can't use (GARCH is one fit per name).

> 📊 **What validation showed.** *Single-name*, the first-cut TCN clearly **loses**
> to GARCH-t (AAPL 30d, 35 windows: TCN VaR breached 14% vs the 5% target, Kupiec
> rejects). GARCH-t already nails single-name vol-clustering + fat tails, and a small
> net on ~3.5 years of one name overfits. *Pooled across 14 large-caps* (30d, 490
> windows) the gap **closes to a tie**: the pooled TCN wins on pinball loss (0.0149
> vs 0.0151) and terminal calibration, loses slightly on touch calibration, matches
> on VaR. Conclusion: pooling helps exactly as predicted, but on liquid large-caps
> at short horizons the two are a wash — **GARCH-t is the pragmatic production engine**
> (no GPU, refits per name, validated), and the TCN is the research track with
> headroom (more names, bigger model, regime/cross-asset structure GARCH can't see).
> Shipping the `backtest` harness first is what let us measure this instead of
> guessing.

Requires PyTorch (`pip install -e ".[gpu]"`); runs on CUDA when available, else CPU.

### Cross-asset joint model — where the generative model wins (`generate --joint`)

GARCH is fit one name at a time, so a portfolio built from independent GARCH
samples assumes the names move independently — it **over-credits diversification**
and under-states the risk that the whole basket sells off together. A cross-asset
**dynamic factor model** (a fat-tailed common market factor learned by the TCN +
per-name betas and idiosyncratic Student-t shocks) produces realistic joint tails.

`generate --joint --tickers ...` trains it on early history and scores the frozen
model vs per-name refit GARCH on held-out windows, on a **portfolio** scorecard and
a **single-name** scorecard side by side.

> ✅ **This is where it wins.** On a 14-name basket (30d), independent GARCH breached
> its 5% portfolio VaR **8.6%** of the time (over-confident about diversification),
> while the joint model came in at **2.9%** — conservative and closer to nominal —
> and edged portfolio pinball. The cost: single-name marginals are slightly worse
> than per-name GARCH (the one-factor approximation). So GARCH-t stays the
> single-name engine; the **joint model is the right tool for basket/portfolio tail
> risk** — directly relevant to writing calls across many names (the odds they all
> get assigned in the same selloff). Caveat: at 35 windows the win is directionally
> clear but not yet statistically airtight (Kupiec needs more windows).

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
  screener.py       # universe scan -> ranked CSV (option contracts)
  riskscan.py       # universe scan -> ranked CSV (underlyings, by simulated risk/return)
  valuescan.py      # universe scan -> ranked CSV (underlyings, by value/quality composite)
  longcall.py       # vega-aware P&L distribution of BUYING a call (GARCH stochastic vol)
  chainlog.py       # daily option-chain snapshot logger -> data/chains/ (builds IV history)
  edgar.py          # SEC EDGAR point-in-time fundamentals fetch/parse -> data/edgar/
  fred.py           # FRED macro series (rates) with as-of lookup -> data/fred/
  panel.py          # as-of feature panel (EDGAR + prices + FRED) -> data/panel/
  valuemodel.py     # LightGBM cross-sectional value ranker + purged walk-forward IC
  deepdive.py       # single-ticker grid + charts
  simulate.py       # Monte-Carlo touch/terminal probabilities, VaR/CVaR, drawdown (GPU/CPU)
  expected_return.py# valuation-conditioned drift (Grinold-Kroner / analyst) for the sim
  backtest.py       # walk-forward calibration / VaR-coverage / pinball validation
  generative.py     # autoregressive TCN generative path model (PyTorch, GPU)
  joint.py          # cross-asset dynamic factor model: joint/portfolio tail risk
  cli.py            # screen / valuescan / longcall / log-chains / riskscan / deepdive / ...
app.py              # Streamlit dashboard wrapping every tool (streamlit run app.py)
config/             # universe.yaml, settings.yaml
tests/              # unit tests for the math
```
