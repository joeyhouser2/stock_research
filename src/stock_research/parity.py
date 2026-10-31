"""Put-call parity check: the option market's implied forward price vs the
textbook one.

Put-call parity (European options, continuous dividend yield q) says::

    C - P = S*e^(-qT) - K*e^(-rT)

which rearranges to a forward price for the underlying implied purely by the
matched call/put quotes at a strike, that should be constant across strikes for
a given expiry::

    F_implied = (C - P) * e^(rT) + K

Comparing that to the textbook forward ``F_theo = S*e^((r-q)T)`` gives a basis:
persistent deviation flags a bad dividend/rate assumption, a borrow-cost or
early-exercise premium (real American-style listed options, not the European
model), or just noisy/illiquid quotes. It is a no-arbitrage cross-check on the
option chain, not a prediction of where the stock is headed.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd

from . import data, metrics
from .config import Settings

ROW_COLUMNS = [
    "expiry", "exp_type", "dte", "strike", "call_mid", "put_mid",
    "implied_forward", "theo_forward", "basis_pct",
]
SUMMARY_COLUMNS = ["expiry", "dte", "n_strikes", "implied_forward", "theo_forward", "basis_pct"]


def implied_forward(call_mid: float, put_mid: float, strike: float, t: float, r: float) -> float:
    """Market-implied forward price for the underlying, from put-call parity."""
    return (call_mid - put_mid) * math.exp(r * t) + strike


def theoretical_forward(spot: float, t: float, r: float, q: float = 0.0) -> float:
    """Textbook no-arbitrage forward: spot compounded at the cost-of-carry rate."""
    return spot * math.exp((r - q) * t)


def parity_row(
    *,
    call_row: pd.Series,
    put_row: pd.Series,
    spot: float,
    t: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
) -> dict | None:
    """One matched (call, put) pair at a strike -> parity stats, or None if unpriced."""
    strike = _num(call_row.get("strike"))
    if strike is None:
        return None
    call_mid = metrics.mid_price(_num(call_row.get("bid")), _num(call_row.get("ask")),
                                 _num(call_row.get("lastPrice")))
    put_mid = metrics.mid_price(_num(put_row.get("bid")), _num(put_row.get("ask")),
                                _num(put_row.get("lastPrice")))
    if call_mid is None or put_mid is None or t <= 0:
        return None

    f_implied = implied_forward(call_mid, put_mid, strike, t, risk_free_rate)
    f_theo = theoretical_forward(spot, t, risk_free_rate, dividend_yield)
    return {
        "strike": strike,
        "call_mid": round(call_mid, 4),
        "put_mid": round(put_mid, 4),
        "implied_forward": round(f_implied, 4),
        "theo_forward": round(f_theo, 4),
        "basis_pct": round(f_implied / f_theo - 1, 5) if f_theo else None,
    }


def build(ticker: str, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (per-strike grid, per-expiry summary, header) for one ticker.

    Strikes are kept within ``settings.min_otm``/``max_otm`` of spot (the same
    moneyness band the screener uses), applied symmetrically since a strike
    that's OTM for the call is ITM for the put and vice versa.
    """
    snap = data.get_snapshot(ticker)
    if snap is None:
        raise ValueError(f"No price/size data for {ticker!r}.")

    today = dt.date.today()
    rows: list[dict] = []

    for expiry in data.list_expirations(ticker):
        dte = data.days_to_expiry(expiry, today)
        if dte < settings.min_dte or dte > settings.max_dte:
            continue
        exp_type = data.classify_expiry(expiry)
        if settings.expiry_type != "any" and exp_type != settings.expiry_type:
            continue

        calls, puts = data.get_option_chain(ticker, expiry)
        if calls.empty or puts.empty:
            continue

        t = max(dte, 0) / metrics.DAYS_PER_YEAR
        merged = calls.merge(puts, on="strike", suffixes=("_call", "_put"))

        for _, pair in merged.iterrows():
            strike = float(pair["strike"])
            moneyness = abs(strike - snap.price) / snap.price
            if not (settings.min_otm <= moneyness <= settings.max_otm):
                continue
            call_row = pair.rename({"bid_call": "bid", "ask_call": "ask",
                                    "lastPrice_call": "lastPrice"})
            put_row = pair.rename({"bid_put": "bid", "ask_put": "ask",
                                   "lastPrice_put": "lastPrice"})
            stat = parity_row(call_row=call_row, put_row=put_row, spot=snap.price, t=t,
                              risk_free_rate=settings.risk_free_rate,
                              dividend_yield=snap.dividend_yield)
            if stat is None:
                continue
            stat["expiry"] = expiry
            stat["exp_type"] = exp_type
            stat["dte"] = dte
            rows.append(stat)

    grid = pd.DataFrame(rows).reindex(columns=ROW_COLUMNS) if rows else pd.DataFrame(columns=ROW_COLUMNS)
    if not grid.empty:
        grid = grid.sort_values(["expiry", "strike"]).reset_index(drop=True)

    header = {
        "ticker": ticker,
        "price": snap.price,
        "dividend_yield": snap.dividend_yield,
        "risk_free_rate": settings.risk_free_rate,
    }
    return grid, summarize(grid), header


def summarize(grid: pd.DataFrame) -> pd.DataFrame:
    """Per-expiry median implied forward vs the textbook forward."""
    if grid.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows = []
    for expiry, grp in grid.groupby("expiry", sort=False):
        rows.append({
            "expiry": expiry,
            "dte": int(grp["dte"].iloc[0]),
            "n_strikes": len(grp),
            "implied_forward": round(grp["implied_forward"].median(), 4),
            "theo_forward": round(grp["theo_forward"].iloc[0], 4),
            "basis_pct": round(grp["basis_pct"].median(), 5) if grp["basis_pct"].notna().any() else None,
        })
    return pd.DataFrame(rows).reindex(columns=SUMMARY_COLUMNS)


def render_text(grid: pd.DataFrame, summary: pd.DataFrame, header: dict) -> str:
    """Format the grid + summary + header as a console-friendly string."""
    h = header
    lines = [
        f"\n{h['ticker']}   spot ${h['price']:.2f}   div {h['dividend_yield'] * 100:.2f}%   "
        f"rf {h['risk_free_rate'] * 100:.2f}%"
    ]
    if grid.empty:
        lines.append("  No matched call/put strikes in the configured DTE / strike window.")
        return "\n".join(lines)
    lines.append("\nImplied forward per expiry (median across matched strikes) vs textbook forward:")
    lines.append(summary.to_string(index=False))
    lines.append("\nPer-strike detail:")
    lines.append(grid.to_string(index=False))
    return "\n".join(lines)


def _num(val) -> float | None:
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None
