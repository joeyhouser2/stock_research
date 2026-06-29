"""Daily option-chain snapshot logger — accumulate per-name implied-vol history.

Free data has no *historical* implied volatility, which blocks a true IV-path
model. The fix is to start collecting it ourselves: this appends a daily snapshot
of the call chains (within a DTE window) to ``data/chains/chains_YYYYMMDD.csv``.
Run it on a schedule (cron / Task Scheduler); after enough days accumulate, the
per-contract IV series unlocks training/validating the joint (price, IV) model.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from . import data
from .config import REPO_ROOT

CHAINS_DIR = REPO_ROOT / "data" / "chains"
FIELDS = ["date", "ticker", "spot", "expiry", "dte", "strike", "bid", "ask",
          "last", "iv", "open_interest", "volume", "contract"]


def snapshot_ticker(ticker: str, *, min_dte: int, max_dte: int,
                    today: dt.date | None = None) -> list[dict]:
    """Rows for one ticker's call chains within the DTE window. Empty on failure."""
    today = today or dt.date.today()
    snap = data.get_snapshot(ticker)
    if snap is None:
        return []
    rows: list[dict] = []
    for expiry in data.list_expirations(ticker):
        dte = data.days_to_expiry(expiry, today)
        if dte < min_dte or dte > max_dte:
            continue
        chain = data.get_call_chain(ticker, expiry)
        if chain.empty:
            continue
        for _, c in chain.iterrows():
            rows.append({
                "date": today.isoformat(), "ticker": ticker, "spot": round(snap.price, 4),
                "expiry": expiry, "dte": dte, "strike": _f(c.get("strike")),
                "bid": _f(c.get("bid")), "ask": _f(c.get("ask")), "last": _f(c.get("lastPrice")),
                "iv": _f(c.get("impliedVolatility")), "open_interest": _i(c.get("openInterest")),
                "volume": _i(c.get("volume")), "contract": c.get("contractSymbol"),
            })
    return rows


def log_chains(tickers: list[str], *, min_dte: int = 1, max_dte: int = 120,
               out_dir: Path | None = None, throttle: float = 0.0,
               verbose: bool = True) -> Path:
    """Snapshot every ticker's chains and append to today's CSV. Returns the path."""
    import time
    out_dir = out_dir or CHAINS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"chains_{dt.date.today().strftime('%Y%m%d')}.csv"
    new = not path.exists()
    total = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for i, ticker in enumerate(tickers):
            try:
                rows = snapshot_ticker(ticker, min_dte=min_dte, max_dte=max_dte)
                writer.writerows(rows)
                total += len(rows)
                if verbose:
                    print(f"  {ticker}: {len(rows)} contracts")
            except Exception as exc:
                if verbose:
                    print(f"  {ticker}: error {exc!r} - skipped")
            if throttle and i < len(tickers) - 1:
                time.sleep(throttle)
    if verbose:
        print(f"\nLogged {total} contract rows -> {path}")
    return path


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return ""


def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return ""
