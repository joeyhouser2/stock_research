"""Universe scan: rank tradable OTM call sales across the watchlist into a CSV."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from . import data, metrics
from .config import REPO_ROOT, Settings

# Stable column order for the output CSV.
COLUMNS = [
    "ticker", "quote_type", "size_b", "expiry", "dte", "strike", "spot",
    "pct_otm", "mid", "annual_yield", "score", "if_called_yield", "prob_otm", "delta",
    "downside_cushion", "breakeven", "iv", "hv", "iv_hv",
    "open_interest", "volume", "spread_pct", "contract",
]

# Columns the screener can rank by (descending).
SORT_KEYS = ("annual_yield", "score", "if_called_yield", "prob_otm", "downside_cushion")


def analyze_ticker(ticker: str, settings: Settings, *, verbose: bool = False) -> list[dict]:
    """Return stat rows for every OTM call on ``ticker`` passing the filters."""
    snap = data.get_snapshot(ticker)
    if snap is None:
        _log(verbose, f"  {ticker}: no price/size data — skipped")
        return []
    if snap.size_usd < settings.min_market_cap:
        _log(verbose, f"  {ticker}: size ${snap.size_usd/1e9:.2f}B < floor — skipped")
        return []

    hv = data.historical_volatility(ticker, settings.hv_window)
    today = dt.date.today()
    rows: list[dict] = []

    for expiry in data.list_expirations(ticker):
        dte = data.days_to_expiry(expiry, today)
        if dte < settings.min_dte or dte > settings.max_dte:
            continue
        chain = data.get_call_chain(ticker, expiry)
        if chain.empty:
            continue
        for _, contract in chain.iterrows():
            stat = metrics.compute(
                row=contract,
                spot=snap.price,
                dte=dte,
                risk_free_rate=settings.risk_free_rate,
                hv=hv,
                dividend_yield=snap.dividend_yield,
            )
            if stat is None:
                continue
            if not (settings.min_otm <= stat["pct_otm"] <= settings.max_otm):
                continue
            if not metrics.passes_liquidity(
                stat,
                min_oi=settings.min_open_interest,
                min_volume=settings.min_volume,
                max_spread=settings.max_spread_pct,
            ):
                continue
            stat.update(
                ticker=ticker,
                quote_type=snap.quote_type,
                size_b=round(snap.size_usd / 1e9, 2),
                expiry=expiry,
            )
            rows.append(stat)

    _log(verbose, f"  {ticker}: {len(rows)} qualifying OTM calls")
    return rows


def run(
    tickers: list[str],
    settings: Settings,
    *,
    sort_by: str = "annual_yield",
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Scan every ticker, rank by ``sort_by`` (descending), and write a CSV."""
    if sort_by not in SORT_KEYS:
        raise ValueError(f"sort_by must be one of {SORT_KEYS}, got {sort_by!r}")
    all_rows: list[dict] = []
    for ticker in tickers:
        try:
            all_rows.extend(analyze_ticker(ticker, settings, verbose=verbose))
        except Exception as exc:  # one bad ticker shouldn't kill the run
            _log(verbose, f"  {ticker}: error {exc!r} — skipped")

    if not all_rows:
        _log(verbose, "No qualifying contracts found.")
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(all_rows)
    df = df.reindex(columns=COLUMNS)
    df = df.sort_values(sort_by, ascending=False, na_position="last").reset_index(drop=True)
    df = df.head(settings.top)

    out_dir = out_dir or (REPO_ROOT / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"screen_{stamp}.csv"
    df.to_csv(out_path, index=False)
    _log(verbose, f"\nWrote {len(df)} rows -> {out_path}")
    return df


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)
