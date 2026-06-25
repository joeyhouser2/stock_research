"""Command-line entry point: `stock-research screen` and `stock-research deepdive`."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import deepdive, screener
from .config import load_settings, load_universe, override


def _add_common_screen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--risk-free-rate", type=float, help="Annual risk-free rate (e.g. 0.04).")
    p.add_argument("--min-dte", type=int, help="Minimum days to expiry.")
    p.add_argument("--max-dte", type=int, help="Maximum days to expiry.")
    p.add_argument("--min-otm", type=float, help="Minimum %% OTM as a fraction (0.02 = 2%%).")
    p.add_argument("--max-otm", type=float, help="Maximum %% OTM as a fraction (0.15 = 15%%).")
    p.add_argument("--hv-window", type=int, help="Trailing days for realized vol.")


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
    sc.add_argument("--tickers", nargs="+", help="Override the universe with these tickers.")
    sc.add_argument("--quiet", action="store_true", help="Suppress per-ticker progress.")

    dd = sub.add_parser("deepdive", help="Full OTM-call stat grid for one ticker.")
    dd.add_argument("ticker", help="Ticker symbol, e.g. MSFT.")
    _add_common_screen_args(dd)
    dd.add_argument("--liquid-only", action="store_true",
                    help="Apply the liquidity filters (default: show all strikes).")
    dd.add_argument("--charts", action="store_true", help="Save yield/prob PNGs to output/.")

    return parser


def _resolved_settings(args) -> "object":
    base = load_settings()
    return override(
        base,
        risk_free_rate=getattr(args, "risk_free_rate", None),
        min_market_cap=getattr(args, "min_market_cap", None),
        min_dte=getattr(args, "min_dte", None),
        max_dte=getattr(args, "max_dte", None),
        min_otm=getattr(args, "min_otm", None),
        max_otm=getattr(args, "max_otm", None),
        min_open_interest=getattr(args, "min_open_interest", None),
        min_volume=getattr(args, "min_volume", None),
        max_spread_pct=getattr(args, "max_spread_pct", None),
        hv_window=getattr(args, "hv_window", None),
        top=getattr(args, "top", None),
    )


def _cmd_screen(args) -> int:
    settings = _resolved_settings(args)
    tickers = [t.upper() for t in args.tickers] if args.tickers else load_universe()
    if not args.quiet:
        print(f"Screening {len(tickers)} tickers "
              f"(DTE {settings.min_dte}-{settings.max_dte}, "
              f"OTM {settings.min_otm:.0%}-{settings.max_otm:.0%}), "
              f"ranked by {args.sort}...")
    df = screener.run(tickers, settings, sort_by=args.sort, verbose=not args.quiet)
    if df.empty:
        return 1
    pd.set_option("display.max_columns", None, "display.width", 200)
    print("\nTop results:")
    print(df.head(min(len(df), 20)).to_string(index=False))
    return 0


def _cmd_deepdive(args) -> int:
    settings = _resolved_settings(args)
    try:
        grid, header = deepdive.build(args.ticker.upper(), settings, apply_liquidity=args.liquid_only)
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
    if args.command == "deepdive":
        return _cmd_deepdive(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
