"""Rules-based cross-sectional value/quality ranking of the underlyings.

This ranks names by how cheap *and* how high-quality they look right now, as a
"good value buys" screen. It's transparent and rules-based: each fundamental is
turned into a cross-sectional percentile (cheap = good for valuation metrics,
high = good for quality), then blended into a single ``value_score`` in [0, 1].

On top of that blend, several classic *undervaluation methods* (Graham, magic
formula, deep value, FCF yield, dividend, contrarian, analyst upside) each score
the same names their own way. A name earns a **vote** from a method when it lands
in that method's best ``TOP_FRAC`` (Graham: when it passes Graham's tests), and
``votes`` counts how many methods agree — one method liking a name is weak; several
is a stronger lead.

Stocks and ETFs are ranked in **separate peer sets**: an ETF's P/E is a holdings
average and lacks quality/upside data, so mixing them lets thinly-measured ETFs crowd
the top. Leveraged/inverse ETFs and ETFs without a P/E (bonds, commodities) are kept
in the output but not scored.

It uses **current** fundamentals (from the same ``fundamentals`` module the option
tools use), so it's valid for a *live* decision — you're ranking today's names
with today's data, no look-ahead. It is NOT a backtested or trained model: doing
that honestly needs point-in-time fundamentals (a future addition). Treat this as
a disciplined screen, not alpha.
"""

from __future__ import annotations

import datetime as dt
import re
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

METHODS = ("blend", "graham", "magic", "deep", "fcf", "dividend", "contrarian", "analyst")
METHOD_HELP = {
    "blend": "the value_score blend: cheapness 50%, quality 30%, analyst upside 20%",
    "graham": "P/E <= 15, P/B <= 1.5, P/E x P/B <= 22.5; ranked by P/E x P/B and price/Graham number",
    "magic": "Greenblatt-style: earnings yield (EBITDA/EV) + return on equity",
    "deep": "lowest P/E and P/B, quality ignored",
    "fcf": "highest free-cash-flow yield (FCF / market cap)",
    "dividend": "highest dividend yield among profitable payers with payout < 100%",
    "contrarian": "furthest below the 52-week high while still cheap and profitable",
    "analyst": "largest upside to the mean analyst target",
}
TOP_FRAC = 0.2       # a method votes for names in its best 20% (of the peer set)
MIN_PEERS = 5        # smaller country groups are pooled into "Other" for --peer country

# Names of geared / inverse products — their "valuation" is meaningless as a value buy.
_LEVERAGED = re.compile(r"\b(?:[123]x|-[123]x|ultra\w*|leveraged|inverse|bull|bear|short|daily)\b",
                        re.IGNORECASE)

NUMERIC = [
    "size_b", "price", "trailing_pe", "forward_pe", "peg", "ev_ebitda", "price_to_book",
    "roe", "profit_margin", "fcf_yield", "dividend_yield", "payout_ratio", "price_to_graham",
    "nav_discount", "analyst_upside", "pct_off_52w_high",
]
COLUMNS = [
    "ticker", "name", "asset", "quote_type", "country", "category", "currency",
    "size_b", "price",
    "trailing_pe", "forward_pe", "peg", "ev_ebitda", "price_to_book",
    "roe", "profit_margin", "fcf_yield", "dividend_yield", "payout_ratio", "price_to_graham",
    "nav_discount", "analyst_upside", "pct_off_52w_high",
    "value_cheap", "value_quality", "value_upside", "value_score",
    *[f"m_{m}" for m in METHODS],
    "votes", "methods_hit", "excluded", "currency_mismatch",
]
SORT_KEYS = ("value_score", "votes", "value_cheap", "value_quality", "value_upside",
             *[f"m_{m}" for m in METHODS if m != "blend"])

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


# --- undervaluation methods -------------------------------------------------- #

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[name], errors="coerce")


def _pos(df: pd.DataFrame, name: str) -> pd.Series:
    s = _col(df, name)
    return s.where(s > 0)


def _mean_ranks(*ranks: pd.Series, require_all: bool = True) -> pd.Series:
    m = pd.concat(ranks, axis=1)
    out = m.mean(axis=1)
    return out.where(m.notna().all(axis=1)) if require_all else out


def _top(score: pd.Series) -> pd.Series:
    """True for names in the best TOP_FRAC of those the method could score."""
    return (score.rank(pct=True) > 1 - TOP_FRAC).fillna(False)


def method_scores(df: pd.DataFrame, methods=METHODS) -> pd.DataFrame:
    """Per-method score in [0, 1] (``m_<method>``, 1 = most undervalued) plus a vote each.

    ``df`` is one peer set and must already carry ``value_score``. Returns the
    ``m_*`` columns, ``votes`` (how many selected methods vote for the name) and
    ``methods_hit`` (which ones).
    """
    pe, pb = _pos(df, "trailing_pe"), _pos(df, "price_to_book")
    roe, payout = _col(df, "roe"), _col(df, "payout_ratio")
    lo, hi = False, True                     # _pct_rank's higher_better flag

    def graham():
        ok = (pe <= 15) & (pb <= 1.5) & (pe * pb <= 22.5)
        score = _mean_ranks(_pct_rank(pe * pb, lo), _pct_rank(_pos(df, "price_to_graham"), lo),
                            require_all=False).where(pe.notna() & pb.notna())
        return score, ok.fillna(False)

    def magic():
        ey = 1.0 / _pos(df, "ev_ebitda")
        return _mean_ranks(_pct_rank(ey, hi), _pct_rank(roe, hi)), None

    def deep():
        return _mean_ranks(_pct_rank(pe, lo), _pct_rank(pb, lo)), None

    def fcf():
        return _pct_rank(_pos(df, "fcf_yield"), hi), None

    def dividend():
        ok = pe.notna() & (payout.isna() | (payout < 1.0))
        return _pct_rank(_pos(df, "dividend_yield").where(ok), hi), None

    def contrarian():
        ok = pe.notna() & (roe.isna() | (roe > 0))
        return _mean_ranks(_pct_rank(_col(df, "pct_off_52w_high").where(ok), lo),
                           _pct_rank(pe.where(ok), lo)), None

    def analyst():
        return _pct_rank(_col(df, "analyst_upside"), hi), None

    def blend():
        return _col(df, "value_score"), None

    fns = {"blend": blend, "graham": graham, "magic": magic, "deep": deep, "fcf": fcf,
           "dividend": dividend, "contrarian": contrarian, "analyst": analyst}
    out = pd.DataFrame(index=df.index)
    hits = pd.DataFrame(index=df.index)
    for m in methods:
        score, vote = fns[m]()
        out[f"m_{m}"] = score
        hits[m] = _top(score) if vote is None else vote
    out["votes"] = hits.sum(axis=1).astype(int)
    out["methods_hit"] = hits.apply(lambda r: ",".join(m for m in hits.columns if r[m]), axis=1)
    return out


def etf_exclusion(name, trailing_pe) -> str | None:
    """Why an ETF can't be value-scored, or None if it can."""
    if _LEVERAGED.search(str(name or "")):
        return "leveraged/inverse ETF"
    if pd.isna(trailing_pe) or trailing_pe <= 0:
        return "ETF without P/E (bonds, commodities, ...)"
    return None


def _peer_keys(df: pd.DataFrame, peer: str) -> pd.Series:
    """Peer-set label per row: ETFs vs stocks, and (``peer='country'``) stocks by country."""
    keys = df["asset"].astype(str)
    if peer == "country":
        country = df["country"].fillna("Unknown").astype(str)
        counts = country[df["asset"] == "stock"].value_counts()
        country = country.where(country.map(counts).fillna(0) >= MIN_PEERS, "Other")
        keys = keys.where(df["asset"] != "stock", "stock:" + country)
    return keys


def score_universe(df: pd.DataFrame, methods=METHODS, peer: str = "all",
                   weights: tuple[float, float, float] = DEFAULT_WEIGHTS) -> pd.DataFrame:
    """Score every peer set separately; unscorable ETFs stay in, flagged in ``excluded``."""
    if peer not in ("all", "country"):
        raise ValueError(f"peer must be 'all' or 'country', got {peer!r}")
    df = df.copy()
    if "asset" not in df:
        df["asset"] = "stock"
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    names = df["name"] if "name" in df else pd.Series(None, index=df.index)
    pes = _col(df, "trailing_pe")
    df["excluded"] = [etf_exclusion(n, p) if a == "etf" else None
                      for a, n, p in zip(df["asset"], names, pes)]

    scorable = df[df["excluded"].isna()]
    parts = []
    for _, group in scorable.groupby(_peer_keys(scorable, peer), sort=False):
        scored = composite_value_scores(group, weights)
        parts.append(pd.concat([scored, method_scores(scored, methods)], axis=1))
    return pd.concat([*parts, df[df["excluded"].notna()]])


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
           "asset": "etf" if snap.quote_type in ("ETF", "MUTUALFUND") else "stock",
           "size_b": round(snap.size_usd / 1e9, 2), "price": round(snap.price, 2)}
    row.update(fundamentals.compute(snap.info, snap.price))
    row.update(fundamentals.compute_extended(snap.info, snap.price))
    if row["currency_mismatch"]:
        row["ev_ebitda"] = None          # EV in quote currency vs EBITDA in reporting currency
    pm = row.get("profit_margin")
    if pm is not None and abs(pm) > 1:   # >100% margins are bad data, not great businesses
        row["profit_margin"] = None
    return row


def run(
    tickers: list[str],
    settings: Settings,
    *,
    sort_by: str = "value_score",
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    filters: dict | None = None,
    methods=METHODS,
    peer: str = "all",
    assets=("stock", "etf"),
    throttle: float = 0.0,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Rank names by value score and methods (after parameter filters) and write a CSV.

    Stocks and ETFs are scored in separate peer sets; the returned frame holds both
    (filter on ``asset``). Unscorable rows (``excluded``) sort last.
    """
    if sort_by not in SORT_KEYS:
        raise ValueError(f"sort_by must be one of {SORT_KEYS}, got {sort_by!r}")
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods {sorted(unknown)}; choose from {METHODS}")
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

    raw = pd.DataFrame(rows)
    raw = raw[raw["asset"].isin(assets)]
    raw = apply_value_filters(raw, filters)                  # hard parameter gates first
    if raw.empty:
        _log(verbose, "No names cleared the value filters.")
        return pd.DataFrame(columns=COLUMNS)
    df = score_universe(raw, methods, peer, weights)         # then rank the survivors
    df = df.reindex(columns=COLUMNS)
    df = df[df["value_score"].notna() | df["excluded"].notna()]   # drop silent no-data rows
    df["votes"] = df["votes"].astype("Int64")                     # NaN for unscored rows
    keys = [sort_by] if sort_by == "value_score" else [sort_by, "value_score"]
    df = df.sort_values(keys, ascending=False, na_position="last").reset_index(drop=True)

    out_dir = out_dir or (REPO_ROOT / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"value_{stamp}.csv"
    df.to_csv(out_path, index=False)
    n_ex = int(df["excluded"].notna().sum())
    _log(verbose, f"\nWrote {len(df)} rows -> {out_path}"
                  + (f" ({n_ex} unscored: see 'excluded')" if n_ex else ""))
    return df


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)
