"""Command-line entry point: `stock-research screen` and `stock-research deepdive`."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import backtest, deepdive, riskscan, screener, simulate
from .config import load_settings, load_universe, override


def _add_common_screen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--risk-free-rate", type=float, help="Annual risk-free rate (e.g. 0.04).")
    p.add_argument("--min-dte", type=int, help="Minimum days to expiry.")
    p.add_argument("--max-dte", type=int, help="Maximum days to expiry.")
    p.add_argument("--min-otm", type=float, help="Minimum %% OTM as a fraction (0.02 = 2%%).")
    p.add_argument("--max-otm", type=float, help="Maximum %% OTM as a fraction (0.15 = 15%%).")
    p.add_argument("--hv-window", type=int, help="Trailing days for realized vol.")
    p.add_argument("--value", action="store_true",
                   help="Add underlying value metrics (P/E, PEG, P/B, margins, ROE, "
                        "52w position, analyst upside).")
    exp = p.add_mutually_exclusive_group()
    exp.add_argument("--weekly", action="store_true",
                     help="Only weekly (non-3rd-Friday) expirations; defaults DTE to 1-14 "
                          "if you don't set --min-dte/--max-dte.")
    exp.add_argument("--monthly", action="store_true",
                     help="Only standard monthly (3rd-Friday) expirations.")


def _add_drift_args(p: argparse.ArgumentParser) -> None:
    """Valuation-drift knobs shared by `simulate` and `riskscan`."""
    p.add_argument("--drift-model", choices=("fixed", "fundamental", "analyst", "blend"),
                   help="How to estimate drift from fundamentals (default: settings.drift_model "
                        "= fundamental). fixed = flat rf + equity premium.")
    p.add_argument("--erp", type=float, dest="equity_risk_premium",
                   help="Equity risk premium for the baseline drift (default 0.045).")
    p.add_argument("--pe-anchor", type=float,
                   help="P/E the multiple reverts toward (default: PEG=1 / market).")
    p.add_argument("--reversion-years", type=float, dest="pe_reversion_years",
                   help="Years over which P/E reverts to the anchor (default 5).")
    p.add_argument("--reversion-shrink", type=float, dest="pe_reversion_shrink",
                   help="Weight 0..1 on the valuation-reversion term (default 1).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stock-research",
        description="OTM call-option sale statistics for ETFs and large-cap companies.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sc = sub.add_parser("screen", help="Rank OTM call sales across the universe into a CSV.")
    _add_common_screen_args(sc)
    sc.add_argument("--min-market-cap", type=float, help="Minimum market cap / AUM in USD.")
    sc.add_argument("--min-open-interest", type=int, help="Minimum open interest.")
    sc.add_argument("--min-volume", type=int, help="Minimum contract volume.")
    sc.add_argument("--max-spread-pct", type=float, help="Max (ask-bid)/mid spread.")
    sc.add_argument("--top", type=int, help="Number of ranked rows to keep.")
    sc.add_argument("--sort", choices=screener.SORT_KEYS, default="annual_yield",
                    help="Rank by this column, descending (default: annual_yield).")
    sc.add_argument("--max-pe", type=float,
                    help="Only underlyings with trailing P/E at or below this (implies --value).")
    sc.add_argument("--max-forward-pe", type=float,
                    help="Only underlyings with forward P/E at or below this (implies --value).")
    sc.add_argument("--max-peg", type=float,
                    help="Only underlyings with PEG at or below this (implies --value).")
    sc.add_argument("--min-prob-otm", type=float,
                    help="Only contracts with at least this probability of expiring OTM "
                         "(0.70 = 70%%), to cap assignment risk.")
    sc.add_argument("--tickers", nargs="+", help="Override the universe with these tickers.")
    sc.add_argument("--weeklys", action="store_true",
                    help="Screen the full Cboe weeklys universe (every symbol with weekly "
                         "options) instead of config/universe.yaml. Gated by --min-market-cap.")
    sc.add_argument("--refresh-weeklys", action="store_true",
                    help="Re-download the Cboe weeklys list before screening (implies --weeklys).")
    sc.add_argument("--max-tickers", type=int,
                    help="With --weeklys, cap the scan at the N largest qualifying names.")
    sc.add_argument("--cache-ttl-days", type=float, default=7,
                    help="How long cached market caps stay fresh (default: 7 days).")
    sc.add_argument("--throttle", type=float, default=0.0,
                    help="Seconds to pause between size lookups, to ease Yahoo rate limits.")
    sc.add_argument("--quiet", action="store_true", help="Suppress per-ticker progress.")
    sc.add_argument("--full", action="store_true",
                    help="Print every column (raw pandas dump) instead of the curated table.")
    sc.add_argument("--simulate", action="store_true",
                    help="Add Monte-Carlo sim_otm / sim_touch per contract (fat-tailed, "
                         "valuation-drift-aware) next to the Black-Scholes prob_otm.")
    sc.add_argument("--sim-model", choices=("gbm", "t", "garch", "bootstrap"), default="garch",
                    help="Path model for --simulate (default: garch).")
    sc.add_argument("--sim-paths", type=int, default=20_000,
                    help="Monte-Carlo paths per expiry for --simulate (default: 20k).")

    fw = sub.add_parser("fetch-weeklys",
                        help="Download the Cboe weeklys list to config/weeklys.csv.")
    fw.add_argument("--quiet", action="store_true", help="Only print the final count.")

    sm = sub.add_parser("simulate",
                        help="Monte-Carlo risk & touch-probability for one ticker.")
    sm.add_argument("ticker", help="Ticker symbol, e.g. MSFT.")
    sm.add_argument("--risk-free-rate", type=float, help="Annual risk-free rate (e.g. 0.04).")
    sm.add_argument("--target", type=float, action="append", metavar="PRICE",
                    help="Target/barrier price; repeatable (e.g. --target 250 --target 200).")
    sm.add_argument("--target-pct", type=float, action="append", metavar="FRAC",
                    help="Target as a fractional move from spot, repeatable "
                         "(0.10 = +10%%, -0.10 = -10%%).")
    sm.add_argument("--horizon", type=int, help="Horizon in calendar days (default: max-dte).")
    sm.add_argument("--model", choices=("gbm", "t", "garch", "bootstrap"), default="garch",
                    help="Path model (default: garch — falls back to t if the fit fails).")
    sm.add_argument("--drift", type=float,
                    help="Annualized real-world drift, fixed number. Overrides --drift-model.")
    _add_drift_args(sm)
    sm.add_argument("--vol", type=float, help="Annualized vol override (default: realized).")
    sm.add_argument("--lookback", type=int, default=504,
                    help="Trading days of history to fit on (default: 504 ~ 2yr).")
    sm.add_argument("--block", type=int, default=5,
                    help="Block length for the bootstrap model (default: 5).")
    sm.add_argument("--paths", type=int, default=100_000, help="Monte-Carlo paths (default: 100k).")
    sm.add_argument("--var-level", type=float, default=0.95,
                    help="Confidence level for VaR/CVaR (default: 0.95).")
    sm.add_argument("--seed", type=int, default=12345, help="RNG seed.")
    sm.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available.")

    rs = sub.add_parser("riskscan",
                        help="Rank the universe by simulated risk/return into a CSV.")
    rs.add_argument("--risk-free-rate", type=float, help="Annual risk-free rate (e.g. 0.04).")
    rs.add_argument("--min-market-cap", type=float, help="Minimum market cap / AUM in USD.")
    rs.add_argument("--horizon", type=int, help="Horizon in calendar days (default: max-dte).")
    rs.add_argument("--target-pct", type=float, default=0.05,
                    help="Touch-probability band as a fractional move (default: 0.05 = +/-5%%).")
    rs.add_argument("--model", choices=("gbm", "t", "garch", "bootstrap"), default="garch",
                    help="Path model (default: garch).")
    rs.add_argument("--paths", type=int, default=30_000,
                    help="Monte-Carlo paths per name (default: 30k).")
    rs.add_argument("--sort", choices=riskscan.SORT_KEYS, default="sharpe",
                    help="Rank by this column (risk metrics ascending; default: sharpe).")
    _add_drift_args(rs)
    rs.add_argument("--tickers", nargs="+", help="Override the universe with these tickers.")
    rs.add_argument("--weeklys", action="store_true",
                    help="Scan the full Cboe weeklys universe instead of config/universe.yaml.")
    rs.add_argument("--refresh-weeklys", action="store_true",
                    help="Re-download the Cboe weeklys list before scanning (implies --weeklys).")
    rs.add_argument("--max-tickers", type=int,
                    help="With --weeklys, cap the scan at the N largest qualifying names.")
    rs.add_argument("--cache-ttl-days", type=float, default=7,
                    help="How long cached market caps stay fresh (default: 7 days).")
    rs.add_argument("--throttle", type=float, default=0.0,
                    help="Seconds to pause between names, to ease Yahoo rate limits.")
    rs.add_argument("--quiet", action="store_true", help="Suppress per-ticker progress.")

    gn = sub.add_parser("generate",
                        help="Train the autoregressive TCN generative model and simulate with it.")
    gn.add_argument("ticker", nargs="?", help="Ticker symbol, e.g. MSFT (omit for --pooled).")
    gn.add_argument("--pooled", action="store_true",
                    help="Train ONE TCN pooled across many names, then score the frozen "
                         "pooled model vs per-name refit GARCH on held-out windows.")
    gn.add_argument("--joint", action="store_true",
                    help="Train a cross-asset factor model and score PORTFOLIO tail risk "
                         "(joint vs independent GARCH) plus single-name marginals.")
    gn.add_argument("--tickers", nargs="+", help="Names for --pooled (default: universe.yaml).")
    gn.add_argument("--max-tickers", type=int, help="Cap the pooled set at N names.")
    gn.add_argument("--min-market-cap", type=float, help="Minimum market cap / AUM in USD.")
    gn.add_argument("--risk-free-rate", type=float, help="Annual risk-free rate (e.g. 0.04).")
    gn.add_argument("--target", type=float, action="append", metavar="PRICE",
                    help="Target/barrier price; repeatable.")
    gn.add_argument("--target-pct", type=float, action="append", metavar="FRAC",
                    help="Target as a fractional move from spot; repeatable.")
    gn.add_argument("--horizon", type=int, help="Horizon in calendar days (default: max-dte).")
    gn.add_argument("--drift", type=float,
                    help="Annualized real-world drift, fixed. Overrides --drift-model.")
    _add_drift_args(gn)
    gn.add_argument("--history", type=int, default=2520, dest="history_days",
                    help="Calendar days of history to train on (default: 2520 ~ 7yr).")
    gn.add_argument("--steps", type=int, default=1500, help="Training steps (default: 1500).")
    gn.add_argument("--paths", type=int, default=20_000, help="Sample paths (default: 20k).")
    gn.add_argument("--var-level", type=float, default=0.95, help="VaR/CVaR level (default 0.95).")
    gn.add_argument("--compare-baseline", action="store_true",
                    help="Train on early history, then backtest the frozen TCN vs a refit "
                         "GARCH on the held-out tail (calibration / VaR / pinball).")
    gn.add_argument("--seed", type=int, default=0, help="RNG seed.")
    gn.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available.")
    gn.add_argument("--quiet", action="store_true", help="Suppress training progress.")

    bt = sub.add_parser("backtest",
                        help="Walk-forward calibration of the simulation's forecasts.")
    bt.add_argument("tickers", nargs="+", help="Ticker(s); records are pooled.")
    bt.add_argument("--risk-free-rate", type=float, help="Annual risk-free rate (e.g. 0.04).")
    bt.add_argument("--horizon", type=int, help="Forecast horizon, calendar days (default: max-dte).")
    bt.add_argument("--target-pct", type=float, default=0.05,
                    help="Touch band as a fractional move (default: 0.05 = +/-5%%).")
    bt.add_argument("--model", choices=("gbm", "t", "garch", "bootstrap"), default="garch",
                    help="Path model to validate (default: garch).")
    bt.add_argument("--lookback", type=int, default=504,
                    help="Trailing trading days fit at each step (default: 504).")
    bt.add_argument("--step", type=int,
                    help="Trading days between windows (default: horizon = non-overlapping).")
    bt.add_argument("--history", type=int, default=1825, dest="history_days",
                    help="Calendar days of price history to pull (default: 1825 ~ 5yr).")
    bt.add_argument("--paths", type=int, default=20_000, help="Paths per window (default: 20k).")
    bt.add_argument("--drift", type=float,
                    help="Flat annual drift for the backtest (default: rf + equity premium). "
                         "Point-in-time fundamentals aren't available, so valuation drift "
                         "can't be backtested without look-ahead.")
    bt.add_argument("--var-level", type=float, default=0.95,
                    help="Confidence level for the VaR coverage test (default: 0.95).")
    bt.add_argument("--seed", type=int, default=12345, help="RNG seed.")
    bt.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available.")
    bt.add_argument("--quiet", action="store_true", help="Suppress per-ticker progress.")

    dd = sub.add_parser("deepdive", help="Full OTM-call stat grid for one ticker.")
    dd.add_argument("ticker", help="Ticker symbol, e.g. MSFT.")
    _add_common_screen_args(dd)
    dd.add_argument("--liquid-only", action="store_true",
                    help="Apply the liquidity filters (default: show all strikes).")
    dd.add_argument("--charts", action="store_true", help="Save yield/prob PNGs to output/.")
    dd.add_argument("--simulate", action="store_true",
                    help="Add Monte-Carlo sim_otm / sim_touch columns per strike "
                         "(fat-tailed, valuation-drift-aware) next to the Black-Scholes prob_otm.")
    dd.add_argument("--sim-model", choices=("gbm", "t", "garch", "bootstrap"), default="garch",
                    help="Path model for --simulate (default: garch).")
    dd.add_argument("--sim-paths", type=int, default=50_000,
                    help="Monte-Carlo paths per expiry for --simulate (default: 50k).")

    return parser


def _resolved_settings(args) -> "object":
    base = load_settings()

    weekly = getattr(args, "weekly", False)
    monthly = getattr(args, "monthly", False)
    expiry_type = "weekly" if weekly else "monthly" if monthly else None

    # "Selling weeklies" usually means the near term, so when --weekly is given and
    # the user hasn't pinned a DTE window, narrow it to the next couple of weeks.
    min_dte = getattr(args, "min_dte", None)
    max_dte = getattr(args, "max_dte", None)
    if weekly and min_dte is None and max_dte is None:
        min_dte, max_dte = 1, 14

    return override(
        base,
        risk_free_rate=getattr(args, "risk_free_rate", None),
        min_market_cap=getattr(args, "min_market_cap", None),
        min_dte=min_dte,
        max_dte=max_dte,
        expiry_type=expiry_type,
        min_otm=getattr(args, "min_otm", None),
        max_otm=getattr(args, "max_otm", None),
        min_open_interest=getattr(args, "min_open_interest", None),
        min_volume=getattr(args, "min_volume", None),
        max_spread_pct=getattr(args, "max_spread_pct", None),
        hv_window=getattr(args, "hv_window", None),
        top=getattr(args, "top", None),
        max_pe=getattr(args, "max_pe", None),
        max_forward_pe=getattr(args, "max_forward_pe", None),
        max_peg=getattr(args, "max_peg", None),
        min_prob_otm=getattr(args, "min_prob_otm", None),
        drift_model=getattr(args, "drift_model", None),
        equity_risk_premium=getattr(args, "equity_risk_premium", None),
        pe_anchor=getattr(args, "pe_anchor", None),
        pe_reversion_years=getattr(args, "pe_reversion_years", None),
        pe_reversion_shrink=getattr(args, "pe_reversion_shrink", None),
    )


# Curated columns for the terminal table: (df_column, header, formatter).
# The full set still lands in the CSV; --full prints everything.
def _pct(v, dp=1):
    return "-" if v != v else f"{v * 100:.{dp}f}%"


def _money(v):
    return "-" if v != v else f"{v:,.2f}"


def _num(v, dp=2):
    return "-" if v != v else f"{v:.{dp}f}"


def _int(v):
    return "-" if v != v else f"{int(v):,}"


_SCREEN_COLUMNS = [
    ("ticker", "Ticker", str),
    ("expiry", "Expiry", str),
    ("dte", "DTE", _int),
    ("strike", "Strike", _money),
    ("spot", "Spot", _money),
    ("pct_otm", "OTM%", _pct),
    ("mid", "Mid", _money),
    ("annual_yield", "Ann.Yld", _pct),
    ("score", "Score", lambda v: _num(v, 2)),
    ("prob_otm", "P(OTM)", _pct),
    ("sim_otm", "Sim(OTM)", _pct),
    ("sim_touch", "Sim(Touch)", _pct),
    ("delta", "Delta", lambda v: _num(v, 2)),
    ("open_interest", "OI", _int),
    ("spread_pct", "Spread", _pct),
]


def _print_screen_table(df, n: int) -> None:
    from rich.console import Console
    from rich.table import Table

    cols = [c for c in _SCREEN_COLUMNS if c[0] in df.columns]
    table = Table(title=f"Top {n} results", title_style="bold", header_style="bold cyan")
    for i, (_, header, _fmt) in enumerate(cols):
        table.add_column(header, justify="left" if i == 0 else "right", no_wrap=True)
    for _, row in df.head(n).iterrows():
        table.add_row(*(fmt(row[key]) if fmt is not str else str(row[key])
                        for key, _header, fmt in cols))
    Console().print(table)


def _cmd_screen(args) -> int:
    settings = _resolved_settings(args)
    verbose = not args.quiet
    use_weeklys = args.weeklys or args.refresh_weeklys

    if use_weeklys:
        if verbose:
            print(f"Screening the Cboe weeklys universe "
                  f"(DTE {settings.min_dte}-{settings.max_dte}, "
                  f"OTM {settings.min_otm:.0%}-{settings.max_otm:.0%}, "
                  f"{settings.expiry_type} expiries), ranked by {args.sort}...")
        df = screener.run_weeklys(
            settings, sort_by=args.sort, with_value=args.value,
            simulate_risk=args.simulate, sim_model=args.sim_model, sim_paths=args.sim_paths,
            refresh_weeklys=args.refresh_weeklys, cache_ttl_days=args.cache_ttl_days,
            throttle=args.throttle, max_tickers=args.max_tickers, verbose=verbose,
        )
    else:
        tickers = [t.upper() for t in args.tickers] if args.tickers else load_universe()
        if verbose:
            print(f"Screening {len(tickers)} tickers "
                  f"(DTE {settings.min_dte}-{settings.max_dte}, "
                  f"OTM {settings.min_otm:.0%}-{settings.max_otm:.0%}, "
                  f"{settings.expiry_type} expiries), "
                  f"ranked by {args.sort}...")
        df = screener.run(tickers, settings, sort_by=args.sort,
                          with_value=args.value, simulate_risk=args.simulate,
                          sim_model=args.sim_model, sim_paths=args.sim_paths, verbose=verbose)

    if df.empty:
        return 1
    n = min(len(df), 20)
    if args.full:
        pd.set_option("display.max_columns", None, "display.width", 200)
        print("\nTop results:")
        print(df.head(n).to_string(index=False))
    else:
        _print_screen_table(df, n)
    return 0


def _cmd_fetch_weeklys(args) -> int:
    from . import cboe
    try:
        records = cboe.refresh()
    except Exception as exc:
        print(f"Error fetching Cboe weeklys list: {exc}", file=sys.stderr)
        return 1
    etfs = sum(1 for r in records if r["type"] == "ETF")
    equities = sum(1 for r in records if r["type"] == "EQUITY")
    print(f"Saved {len(records)} weeklys symbols ({equities} equities, {etfs} ETFs/ETNs) "
          f"-> {cboe.WEEKLYS_CSV}")
    return 0


def _cmd_simulate(args) -> int:
    settings = _resolved_settings(args)
    try:
        sim, report = simulate.run(
            args.ticker.upper(), settings,
            targets=args.target, target_pcts=args.target_pct,
            horizon_days=args.horizon, model=args.model,
            mu=args.drift, sigma=args.vol, lookback=args.lookback,
            block=args.block, n_paths=args.paths, var_level=args.var_level,
            seed=args.seed, force_cpu=args.cpu,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(simulate.render_text(sim, report))
    return 0


_RISKSCAN_COLUMNS = [
    ("ticker", "Ticker", str),
    ("quote_type", "Type", str),
    ("spot", "Spot", _money),
    ("drift", "Drift", _pct),
    ("sigma", "Vol", _pct),
    ("sharpe", "Sharpe", lambda v: _num(v, 2)),
    ("exp_return", "E[ret]", _pct),
    ("prob_up", "Touch+", _pct),
    ("prob_down", "Touch-", _pct),
    ("var", "VaR", _pct),
    ("cvar", "CVaR", _pct),
    ("mdd", "MaxDD", _pct),
]


def _print_riskscan_table(df, n: int, target_pct: float) -> None:
    from rich.console import Console
    from rich.table import Table

    cols = [c for c in _RISKSCAN_COLUMNS if c[0] in df.columns]
    table = Table(title=f"Top {n} by risk/return (touch band +/-{target_pct:.0%})",
                  title_style="bold", header_style="bold cyan")
    for i, (_, header, _fmt) in enumerate(cols):
        table.add_column(header, justify="left" if i == 0 else "right", no_wrap=True)
    for _, row in df.head(n).iterrows():
        table.add_row(*(fmt(row[key]) if fmt is not str else str(row[key])
                        for key, _header, fmt in cols))
    Console().print(table)


def _cmd_riskscan(args) -> int:
    settings = _resolved_settings(args)
    verbose = not args.quiet
    use_weeklys = args.weeklys or args.refresh_weeklys
    common = dict(horizon_days=args.horizon, target_pct=args.target_pct, model=args.model,
                  n_paths=args.paths, sort_by=args.sort, verbose=verbose)
    if use_weeklys:
        df = riskscan.run_weeklys(
            settings, refresh_weeklys=args.refresh_weeklys, cache_ttl_days=args.cache_ttl_days,
            throttle=args.throttle, max_tickers=args.max_tickers, **common,
        )
    else:
        tickers = [t.upper() for t in args.tickers] if args.tickers else load_universe()
        if verbose:
            horizon = args.horizon or settings.max_dte
            print(f"Risk-scanning {len(tickers)} tickers over {horizon}d "
                  f"({args.model} paths), ranked by {args.sort}...")
        df = riskscan.run(tickers, settings, throttle=args.throttle, **common)

    if df.empty:
        return 1
    _print_riskscan_table(df, min(len(df), 25), args.target_pct)
    return 0


def _render_compare(res: dict, target_pct: float, var_level: float,
                    header: str = "Frozen TCN vs refit GARCH on held-out tail") -> str:
    g, t = res["garch"], res["tcn"]
    rows = [
        ("touch+ ECE", f"{g['touch_up'].ece:.4f}", f"{t['touch_up'].ece:.4f}", "lower"),
        ("touch- ECE", f"{g['touch_down'].ece:.4f}", f"{t['touch_down'].ece:.4f}", "lower"),
        ("term+ ECE", f"{g['term_up'].ece:.4f}", f"{t['term_up'].ece:.4f}", "lower"),
        ("touch+ Brier", f"{g['touch_up'].brier:.4f}", f"{t['touch_up'].brier:.4f}", "lower"),
        (f"VaR obs (exp {1-var_level:.0%})",
         f"{g['var']['observed_rate']:.1%}", f"{t['var']['observed_rate']:.1%}", "~exp"),
        ("VaR Kupiec p", f"{g['var']['kupiec_p']:.3f}", f"{t['var']['kupiec_p']:.3f}", ">0.05"),
        ("pinball overall", f"{g['pinball']['overall']:.4f}", f"{t['pinball']['overall']:.4f}", "lower"),
    ]
    lines = [f"\n{header}   "
             f"({res['windows']} windows, touch band +/-{target_pct:.0%})",
             f"    {'metric':>18} {'GARCH':>10} {'TCN':>10}   {'better':>6}"]
    for name, gv, tv, better in rows:
        lines.append(f"    {name:>18} {gv:>10} {tv:>10}   {better:>6}")
    return "\n".join(lines)


def _render_joint(res: dict, var_level: float) -> str:
    p, s = res["portfolio"], res["single_name"]
    exp = 1 - var_level
    lines = [f"\nPORTFOLIO tail risk (the basket) — joint factor model vs independent GARCH"
             f"   ({res['windows']} windows)",
             f"    {'metric':>20} {'indepGARCH':>11} {'joint':>11}   {'want':>8}",
             f"    {'VaR obs (exp '+format(exp,'.0%')+')':>20} "
             f"{p['independent']['var']['observed_rate']:>11.1%} "
             f"{p['joint']['var']['observed_rate']:>11.1%}   {'~'+format(exp,'.0%'):>8}",
             f"    {'VaR Kupiec p':>20} {p['independent']['var']['kupiec_p']:>11.3f} "
             f"{p['joint']['var']['kupiec_p']:>11.3f}   {'>0.05':>8}",
             f"    {'pinball overall':>20} {p['independent']['pinball']['overall']:>11.4f} "
             f"{p['joint']['pinball']['overall']:>11.4f}   {'lower':>8}",
             f"\nSINGLE-NAME marginals (terminal) — joint model vs GARCH"
             f"   ({res['windows']*res['n_tickers']} records)",
             f"    {'metric':>20} {'GARCH':>11} {'joint':>11}   {'want':>8}",
             f"    {'VaR obs (exp '+format(exp,'.0%')+')':>20} "
             f"{s['garch']['var']['observed_rate']:>11.1%} "
             f"{s['joint']['var']['observed_rate']:>11.1%}   {'~'+format(exp,'.0%'):>8}",
             f"    {'term+ ECE':>20} {s['garch']['term_up'].ece:>11.4f} "
             f"{s['joint']['term_up'].ece:>11.4f}   {'lower':>8}",
             f"    {'pinball overall':>20} {s['garch']['pinball']['overall']:>11.4f} "
             f"{s['joint']['pinball']['overall']:>11.4f}   {'lower':>8}"]
    return "\n".join(lines)


def _cmd_generate(args) -> int:
    from . import data, expected_return, generative
    settings = _resolved_settings(args)
    verbose = not args.quiet
    horizon = args.horizon or settings.max_dte
    band = args.target_pct[0] if args.target_pct else 0.05
    mu_baseline = (args.drift if args.drift is not None
                   else settings.risk_free_rate + settings.equity_risk_premium)

    if args.joint:
        from . import joint
        tickers = [t.upper() for t in args.tickers] if args.tickers else load_universe()
        if args.max_tickers:
            tickers = tickers[:args.max_tickers]
        closes_by = {}
        for tk in tickers:
            closes = data.close_history(tk, args.history_days)
            if len(closes) > 600:
                closes_by[tk] = closes
            if verbose:
                print(f"  {tk}: {len(closes)} closes")
        if len(closes_by) < 2:
            print("Error: need >= 2 tickers with sufficient history.", file=sys.stderr)
            return 1
        res = joint.compare_joint(
            closes_by, horizon_days=horizon, target_pct=band, n_paths=args.paths,
            mu=mu_baseline, var_level=args.var_level, seed=args.seed, force_cpu=args.cpu,
            tcn_hp={"steps": args.steps}, verbose=verbose)
        m = res["model"]
        print(f"\nJoint factor model: {res['n_tickers']} names, equal-weight basket "
              f"(factor TCN NLL {m.final_loss:.4f}, backend {m.device})")
        print(_render_joint(res, args.var_level))
        return 0

    if args.pooled:
        tickers = [t.upper() for t in args.tickers] if args.tickers else load_universe()
        if args.max_tickers:
            tickers = tickers[:args.max_tickers]
        closes_by = {}
        for tk in tickers:
            closes = data.close_history(tk, args.history_days)
            if len(closes) > 600:
                closes_by[tk] = closes
            if verbose:
                print(f"  {tk}: {len(closes)} closes")
        if len(closes_by) < 2:
            print("Error: need >= 2 tickers with sufficient history.", file=sys.stderr)
            return 1
        res = generative.compare_universe(
            closes_by, horizon_days=horizon, target_pct=band, n_paths=args.paths,
            mu=mu_baseline, var_level=args.var_level, seed=args.seed, force_cpu=args.cpu,
            tcn_hp={"steps": args.steps}, verbose=verbose)
        m = res["model"]
        print(f"\nPooled TCN trained on {res['n_tickers']} names "
              f"(final NLL {m.final_loss:.4f}, RF {m.receptive_field}d, backend {m.device})")
        print(_render_compare(res, band, args.var_level,
                              header="Pooled frozen TCN vs per-name refit GARCH (held-out)"))
        return 0

    if not args.ticker:
        print("Error: a ticker is required (or use --pooled).", file=sys.stderr)
        return 1
    ticker = args.ticker.upper()

    snap = data.get_snapshot(ticker)
    if snap is None:
        print(f"Error: No price/size data for {ticker!r}.", file=sys.stderr)
        return 1
    horizon = args.horizon or settings.max_dte
    if args.drift is not None:
        drift = expected_return.DriftEstimate(args.drift, "fixed (--drift)", {}, [])
    else:
        drift = expected_return.estimate(
            snap.info, snap.price, snap.dividend_yield,
            model=settings.drift_model, rf=settings.risk_free_rate,
            erp=settings.equity_risk_premium, pe_anchor=settings.pe_anchor,
            reversion_years=settings.pe_reversion_years, shrink=settings.pe_reversion_shrink)

    if args.compare_baseline:
        closes = data.close_history(ticker, args.history_days)
        res = generative.compare_on_split(
            closes, horizon_days=horizon, target_pct=args.target_pct[0] if args.target_pct else 0.05,
            n_paths=args.paths, mu=drift.annual_drift, var_level=args.var_level,
            seed=args.seed, force_cpu=args.cpu, tcn_hp={"steps": args.steps}, verbose=verbose)
        print(f"\n{ticker}: trained TCN (final NLL {res['model'].final_loss:.4f}, "
              f"RF {res['model'].receptive_field}d, backend {res['model'].device})")
        print(_render_compare(res, args.target_pct[0] if args.target_pct else 0.05, args.var_level))
        return 0

    returns = data.daily_log_returns(ticker, args.history_days)
    model = generative.fit_tcn(returns, steps=args.steps, seed=args.seed,
                               force_cpu=args.cpu, verbose=verbose)
    simu = generative.sample_simulation(
        model, spot=snap.price, context_returns=returns, horizon_days=horizon,
        mu=drift.annual_drift, n_paths=args.paths, seed=args.seed)

    levels = list(args.target or [])
    levels += [round(snap.price * (1 + p), 4) for p in (args.target_pct or [])]
    if not levels:
        levels = [round(snap.price * 1.10, 2), round(snap.price * 0.90, 2)]
    report = {"ticker": ticker, "quote_type": snap.quote_type, "spot": snap.price,
              "drift": drift, "var_level": args.var_level,
              "targets": simulate._target_rows(simu, levels, args.var_level)}
    print(simulate.render_text(simu, report))
    print(f"\n  TCN: final NLL {model.final_loss:.4f}   receptive field {model.receptive_field}d   "
          f"learned vol {model.annual_vol:.1%}/yr   backend {model.device}")
    return 0


def _cmd_backtest(args) -> int:
    settings = _resolved_settings(args)
    horizon = args.horizon or settings.max_dte
    mu = args.drift if args.drift is not None else (
        settings.risk_free_rate + settings.equity_risk_premium)
    _records, scores = backtest.run(
        [t.upper() for t in args.tickers], settings,
        horizon_days=args.horizon, target_pct=args.target_pct, model=args.model,
        lookback=args.lookback, step=args.step, n_paths=args.paths,
        history_days=args.history_days, mu=args.drift, var_level=args.var_level,
        seed=args.seed, force_cpu=args.cpu, verbose=not args.quiet,
    )
    print(backtest.render_text(scores, model=args.model, horizon_days=horizon,
                               target_pct=args.target_pct, mu=mu))
    return 0 if scores["n"] else 1


def _cmd_deepdive(args) -> int:
    settings = _resolved_settings(args)
    try:
        grid, header = deepdive.build(args.ticker.upper(), settings,
                                      apply_liquidity=args.liquid_only, with_value=args.value,
                                      simulate_risk=args.simulate, sim_model=args.sim_model,
                                      sim_paths=args.sim_paths)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    pd.set_option("display.max_columns", None, "display.width", 220)
    print(deepdive.render_text(grid, header))
    if args.charts:
        paths = deepdive.save_charts(grid, header)
        for p in paths:
            print(f"Saved chart -> {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "screen":
        return _cmd_screen(args)
    if args.command == "fetch-weeklys":
        return _cmd_fetch_weeklys(args)
    if args.command == "simulate":
        return _cmd_simulate(args)
    if args.command == "riskscan":
        return _cmd_riskscan(args)
    if args.command == "backtest":
        return _cmd_backtest(args)
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "deepdive":
        return _cmd_deepdive(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
