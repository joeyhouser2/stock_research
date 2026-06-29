"""As-of feature panel — the model-ready bridge from filings/prices to features.

Produces one row per (ticker, rebalance date) where **every feature uses only
data available on that date**: fundamentals filed by then (TTM / latest via
:mod:`edgar`), prices up to then, and macro as-of (:mod:`fred`). A forward return
column is attached as the training label (the only thing that looks ahead — and
it's the target, never a feature).

Features: valuation yields (earnings/book/sales/cash-flow), quality (ROE, margin,
asset turnover), YoY growth, momentum (12-1), 3-month realized vol, size, and the
macro context. This is the dataset a cross-sectional value model trains on; the
model + walk-forward validation are the next step.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import edgar, fred
from .config import REPO_ROOT

PANEL_DIR = REPO_ROOT / "data" / "panel"

FEATURE_COLUMNS = [
    "ticker", "as_of", "price", "market_cap_b", "log_size",
    "earnings_yield", "book_yield", "sales_yield", "cfo_yield",
    "roe", "profit_margin", "asset_turnover", "rev_growth_yoy", "ni_growth_yoy",
    "mom_12_1", "vol_3m", "rate_10y", "rate_3m", "term_spread", "rate_10y_chg_3m",
    "fwd_return",
]


def _ratio(a, b):
    """a / b, but only when both exist and the denominator is positive."""
    if a is None or b is None or b <= 0:
        return None
    return round(float(a) / float(b), 6)


def _growth(now, prev):
    if now is None or prev is None or prev <= 0:
        return None
    return round(float(now) / float(prev) - 1.0, 6)


def _asof_idx(closes: pd.Series, as_of: pd.Timestamp) -> int | None:
    """Position of the last close on or before ``as_of`` (closes sorted ascending).

    yfinance indexes are timezone-aware; drop the tz so the comparison against a
    naive rebalance date doesn't raise.
    """
    idx = closes.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    n = int((idx <= as_of).sum())
    return n - 1 if n > 0 else None


def as_of_features(fund_long: pd.DataFrame, closes: pd.Series, macro: pd.DataFrame,
                   as_of, *, horizon_days: int = 126) -> dict | None:
    """One point-in-time feature row, or None if data isn't available as of ``as_of``."""
    as_of = pd.Timestamp(as_of)
    f = edgar.as_of_snapshot(fund_long, as_of)
    if not f:
        return None
    idx = _asof_idx(closes, as_of)
    if idx is None:
        return None
    prices = np.asarray(closes.values, dtype=float)
    price = float(prices[idx])
    if not (price > 0):
        return None

    shares = f.get("shares_diluted")
    mktcap = price * shares if (shares and shares > 0) else None
    rev, ni, eq = f.get("revenue"), f.get("net_income"), f.get("equity")
    assets, cfo = f.get("assets"), f.get("cash_from_ops")
    prev = edgar.as_of_snapshot(fund_long, as_of - pd.Timedelta(days=365))

    feats = {
        "as_of": as_of.date().isoformat(),
        "price": round(price, 4),
        "market_cap_b": round(mktcap / 1e9, 3) if mktcap else None,
        "log_size": round(float(np.log(mktcap)), 4) if mktcap else None,
        "earnings_yield": _ratio(ni, mktcap),
        "book_yield": _ratio(eq, mktcap),
        "sales_yield": _ratio(rev, mktcap),
        "cfo_yield": _ratio(cfo, mktcap),
        "roe": _ratio(ni, eq),
        "profit_margin": _ratio(ni, rev),
        "asset_turnover": _ratio(rev, assets),
        "rev_growth_yoy": _growth(rev, prev.get("revenue")),
        "ni_growth_yoy": _growth(ni, prev.get("net_income")),
        "mom_12_1": round(float(prices[idx - 21] / prices[idx - 252] - 1.0), 6) if idx >= 252 else None,
        "vol_3m": None,
        "fwd_return": None,
    }
    if idx >= 63:
        r = np.diff(np.log(prices[idx - 63:idx + 1]))
        if r.size > 1:
            feats["vol_3m"] = round(float(np.std(r, ddof=1) * np.sqrt(252.0)), 6)
    feats.update(fred.as_of_macro(macro, as_of))

    hsteps = max(1, round(horizon_days * 252.0 / 365.0))
    if idx + hsteps < len(prices):
        feats["fwd_return"] = round(float(prices[idx + hsteps] / price - 1.0), 6)
    return feats


def build_panel(data_by_ticker: dict, macro: pd.DataFrame, dates,
                *, horizon_days: int = 126) -> pd.DataFrame:
    """Assemble the panel across tickers × rebalance dates."""
    rows = []
    for ticker, d in data_by_ticker.items():
        fund, closes = d.get("fundamentals"), d.get("closes")
        if fund is None or fund.empty or closes is None or len(closes) < 2:
            continue
        for as_of in dates:
            feats = as_of_features(fund, closes, macro, as_of, horizon_days=horizon_days)
            if feats is not None:
                feats["ticker"] = ticker
                rows.append(feats)
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return pd.DataFrame(rows).reindex(columns=FEATURE_COLUMNS)


def run(
    tickers: list[str],
    settings=None,
    *,
    start=None,
    end=None,
    freq_months: int = 3,
    horizon_days: int = 126,
    history_days: int = 3650,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch EDGAR + prices + macro and build the as-of panel; write a CSV."""
    from . import data

    end = pd.Timestamp(end) if end else pd.Timestamp(dt.date.today())
    start = pd.Timestamp(start) if start else end - pd.DateOffset(years=5)
    dates = pd.date_range(start, end, freq=f"{freq_months}MS")

    macro = fred.macro_frame()
    data_by: dict[str, dict] = {}
    for ticker in tickers:
        ticker = ticker.upper()
        try:
            facts = edgar.company_facts(ticker)
            fund = edgar.point_in_time_fundamentals(facts)
        except Exception as exc:
            _log(verbose, f"  {ticker}: EDGAR {exc!r} - skipped")
            continue
        closes = data.close_history(ticker, history_days)
        data_by[ticker] = {"fundamentals": fund, "closes": closes}
        _log(verbose, f"  {ticker}: {len(fund)} facts, {len(closes)} closes")

    df = build_panel(data_by, macro, dates, horizon_days=horizon_days)
    if df.empty:
        _log(verbose, "Empty panel.")
        return df

    out_dir = out_dir or PANEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"panel_{stamp}.csv"
    df.to_csv(out_path, index=False)
    _log(verbose, f"\n{len(df)} rows ({df['ticker'].nunique()} names x {len(dates)} dates) "
                  f"-> {out_path}")
    return df


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)
