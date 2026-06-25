"""Valuation / quality metrics for the underlying.

Covered-call writing can leave you holding (or assigned out of) the shares, so it
helps to know whether the underlying itself is decent value. These all come from
the yfinance ``.info`` dict already fetched for the snapshot, so enabling them
costs no extra network calls. Anything Yahoo doesn't supply comes back as ``None``
(common for ETFs, which lack single-company multiples).
"""

from __future__ import annotations

import math

# Columns appended to the output when the value flag is set, in display order.
VALUE_COLUMNS = [
    "trailing_pe", "forward_pe", "peg", "price_to_book", "ev_ebitda",
    "profit_margin", "roe", "pct_off_52w_high", "pct_above_52w_low", "analyst_upside",
]


def compute(info: dict, price: float) -> dict:
    """Return the value-metric columns for one underlying."""
    high = _num(info, "fiftyTwoWeekHigh")
    low = _num(info, "fiftyTwoWeekLow")
    target = _num(info, "targetMeanPrice")

    return {
        "trailing_pe": _r(_num(info, "trailingPE")),
        "forward_pe": _r(_num(info, "forwardPE")),
        "peg": _r(_num(info, "trailingPegRatio", "pegRatio")),
        "price_to_book": _r(_num(info, "priceToBook")),
        "ev_ebitda": _r(_num(info, "enterpriseToEbitda")),
        "profit_margin": _r(_num(info, "profitMargins"), 4),
        "roe": _r(_num(info, "returnOnEquity"), 4),
        # Negative pct_off_52w_high = trading below its high (more room / cheaper).
        "pct_off_52w_high": _r((price - high) / high, 4) if high else None,
        "pct_above_52w_low": _r((price - low) / low, 4) if low else None,
        # Mean analyst target vs. spot: positive = perceived upside.
        "analyst_upside": _r((target - price) / price, 4) if target else None,
    }


def empty() -> dict:
    """A row of None value-metrics, for when fundamentals can't be computed."""
    return {col: None for col in VALUE_COLUMNS}


def _num(info: dict, *keys: str) -> float | None:
    for key in keys:
        val = info.get(key)
        if val is None:
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            return f
    return None


def _r(val: float | None, ndigits: int = 2) -> float | None:
    return round(val, ndigits) if val is not None else None
