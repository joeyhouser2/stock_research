"""Rules-based cross-sectional value/quality ranking of the underlyings.

This ranks names by how cheap *and* how high-quality they look right now, as a
"good value buys" screen. It's transparent and rules-based: each fundamental is
turned into a cross-sectional percentile (cheap = good for valuation metrics,
high = good for quality), then blended into a single ``value_score`` in [0, 1].

It uses **current** fundamentals (from the same ``fundamentals`` module the option
tools use), so it's valid for a *live* decision — you're ranking today's names
with today's data, no look-ahead. It is NOT a backtested or trained model: doing
that honestly needs point-in-time fundamentals (a future addition). Treat this as
a disciplined screen, not alpha.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, fundamentals
from .config import REPO_ROOT, Settings

# Valuation metrics where LOWER is better, quality where HIGHER is better.
CHEAP = ["trailing_pe", "forward_pe", "peg", "ev_ebitda", "price_to_book"]
QUALITY = ["roe", "profit_margin"]
UPSIDE = ["analyst_upside"]
DEFAULT_WEIGHTS = (0.5, 0.3, 0.2)        # cheap, quality, upside

COLUMNS = [
    "ticker", "quote_type", "size_b", "price",
    "trailing_pe", "forward_pe", "peg", "ev_ebitda", "price_to_book",
    "roe", "profit_margin", "analyst_upside", "pct_off_52w_high",
    "value_cheap", "value_quality", "value_upside", "value_score",
]
SORT_KEYS = ("value_score", "value_cheap", "value_quality", "value_upside")

# Hard parameter filters for the value picker. Each maps a kwarg -> (column, direction).
# "max" keeps rows at/below the threshold (cheapness caps); "min" at/above (quality floors).
# A missing figure fails the filter — we don't pass through what we can't measure.
VALUE_FILTERS = {
    "max_pe": ("trailing_pe", "max"),
    "max_forward_pe": ("forward_pe", "max"),
    "max_peg": ("peg", "max"),
    "max_pb": ("price_to_book", "max"),
    "max_ev_ebitda": ("ev_ebitda", "max"),
    "min_roe": ("roe", "min"),
    "min_margin": ("profit_margin", "min"),
    "min_upside": ("analyst_upside", "min"),
}


def apply_value_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    """Keep only rows passing every active parameter threshold (missing => fails)."""
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for key, (col, direction) in VALUE_FILTERS.items():
        thr = filters.get(key)
        if thr is None or col not in df.columns:
            continue
        v = df[col]
        mask &= v.notna() & ((v <= thr) if direction == "max" else (v >= thr))
    return df[mask]


def _pct_rank(s: pd.Series, higher_better: bool) -> pd.Series:
    """Cross-sectional percentile in [0, 1] (1 = best); NaN stays NaN."""
    s = s.where(np.isfinite(s))
    return s.rank(pct=True, ascending=higher_better)


def composite_value_scores(df: pd.DataFrame,
                           weights: tuple[float, float, float] = DEFAULT_WEIGHTS) -> pd.DataFrame:
    """Add value_cheap / value_quality / value_upside / value_score columns.

    Cheapness metrics with non-positive values (a negative P/E isn't "cheap") are
    treated as missing. The blend is a per-row weighted mean over whichever metric
    groups are available.
    """
    df = df.copy()
    for c in CHEAP:
        if c in df:
            df[c] = df[c].where(df[c] > 0)

    def group_score(cols, higher_better):
        ranks = [_pct_rank(df[c], higher_better) for c in cols if c in df]
        if not ranks:
            return pd.Series(np.nan, index=df.index)
        return pd.concat(ranks, axis=1).mean(axis=1)   # ignores NaN per metric

    cheap = group_score(CHEAP, higher_better=False)
    quality = group_score(QUALITY, higher_better=True)
    upside = group_score(UPSIDE, higher_better=True)
    df["value_cheap"], df["value_quality"], df["value_upside"] = cheap, quality, upside

    groups = pd.DataFrame({"cheap": cheap, "quality": quality, "upside": upside})
    w = np.array(weights, dtype=float)
    num = (groups.fillna(0.0) * w).sum(axis=1)
    den = (groups.notna() * w).sum(axis=1)
    df["value_score"] = (num / den).where(den > 0)
    return df


def analyze_ticker(ticker: str, settings: Settings, *, verbose: bool = False) -> dict | None:
    snap = data.get_snapshot(ticker)
    if snap is None:
        _log(verbose, f"  {ticker}: no price/size data - skipped")
        return None
    if snap.size_usd < settings.min_market_cap:
        _log(verbose, f"  {ticker}: size ${snap.size_usd/1e9:.2f}B < floor - skipped")
        return None
    if settings.max_market_cap is not None and snap.size_usd > settings.max_market_cap:
        _log(verbose, f"  {ticker}: size ${snap.size_usd/1e9:.2f}B > ceiling - skipped")
        return None
    row = {"ticker": ticker, "quote_type": snap.quote_type,
           "size_b": round(snap.size_usd / 1e9, 2), "price": round(snap.price, 2)}
    row.update(fundamentals.compute(snap.info, snap.price))
    return row


def run(
    tickers: list[str],
    settings: Settings,
    *,
    sort_by: str = "value_score",
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    filters: dict | None = None,
    throttle: float = 0.0,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Rank names by composite value score (after parameter filters) and write a CSV."""
    if sort_by not in SORT_KEYS:
        raise ValueError(f"sort_by must be one of {SORT_KEYS}, got {sort_by!r}")
    rows: list[dict] = []
    for i, ticker in enumerate(tickers):
        try:
            row = analyze_ticker(ticker, settings, verbose=verbose)
            if row is not None:
                rows.append(row)
        except Exception as exc:
            _log(verbose, f"  {ticker}: error {exc!r} - skipped")
        if throttle and i < len(tickers) - 1:
            time.sleep(throttle)

    if not rows:
        _log(verbose, "No names with fundamentals.")
        return pd.DataFrame(columns=COLUMNS)

    raw = apply_value_filters(pd.DataFrame(rows), filters)   # hard parameter gates first
    if raw.empty:
        _log(verbose, "No names cleared the value filters.")
        return pd.DataFrame(columns=COLUMNS)
    df = composite_value_scores(raw, weights)                # then rank the survivors
    df = df.reindex(columns=COLUMNS)
    df = df.dropna(subset=["value_score"])               # drop names we can't value (e.g. ETFs)
    df = df.sort_values(sort_by, ascending=False, na_position="last").reset_index(drop=True)

    out_dir = out_dir or (REPO_ROOT / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"value_{stamp}.csv"
    df.to_csv(out_path, index=False)
    _log(verbose, f"\nWrote {len(df)} rows -> {out_path}")
    return df


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)
