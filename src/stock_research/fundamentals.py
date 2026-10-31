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


#: Quote currencies Yahoo reports in minor units (pence, cents, agorot) -> (major, divisor).
MINOR_UNITS = {"GBp": ("GBP", 100), "GBX": ("GBP", 100), "ZAc": ("ZAR", 100),
               "ZAC": ("ZAR", 100), "ILA": ("ILS", 100)}


def major_currency(code: str | None) -> tuple[str, int]:
    """(ISO major currency, divisor from quote units) — ``"GBp"`` -> ``("GBP", 100)``.

    Checked before upper-casing: ``GBp`` (pence) and ``GBP`` (pounds) differ only by case.
    """
    if not code:
        return "USD", 1
    if code in MINOR_UNITS:
        return MINOR_UNITS[code]
    return code.upper(), 1


def compute_extended(info: dict, price: float) -> dict:
    """Extra fields for the value scan: identity, currency, FCF/dividend/Graham, ETF NAV.

    Per-share fundamentals (EPS, book) are in the *reporting* currency while price is
    in the *quote* currency. When those differ (e.g. a USD-listed ADR of a company
    reporting in CNY) Yahoo's cross-currency ratios are unreliable, so the ratios
    built here from raw figures are left as None and ``currency_mismatch`` is set.
    """
    quote_ccy, divisor = major_currency(info.get("currency"))
    fin_ccy = str(info.get("financialCurrency") or quote_ccy).upper()
    mismatch = fin_ccy != quote_ccy
    px = price / divisor                 # major units, comparable to EPS / book

    eps, bvps = _num(info, "trailingEps"), _num(info, "bookValue")
    graham = math.sqrt(22.5 * eps * bvps) if eps and bvps and eps > 0 and bvps > 0 else None
    fcf, mcap = _num(info, "freeCashflow"), _num(info, "marketCap")

    # ETFs report `yield`; equities `trailingAnnualDividendYield` (0.0 for non-payers).
    dy = _num(info, "trailingAnnualDividendYield") or _num(info, "yield")
    if dy is not None and not (0.0 <= dy <= 0.25):     # bare percents / junk
        dy = None

    nav = _num(info, "navPrice")
    nav_disc = (price - nav) / nav if nav else None
    if nav_disc is not None and abs(nav_disc) > 0.5:   # unit mismatch, not a real premium
        nav_disc = None

    return {
        "name": info.get("longName") or info.get("shortName"),
        "country": info.get("country"),
        "category": info.get("category"),
        "currency": quote_ccy,
        "financial_currency": fin_ccy,
        "currency_mismatch": mismatch,
        "dividend_yield": _r(dy, 4),
        "payout_ratio": _r(_num(info, "payoutRatio"), 4),
        "fcf_yield": _r(fcf / mcap, 4) if fcf is not None and mcap and not mismatch else None,
        "price_to_graham": _r(px / graham, 3) if graham and not mismatch else None,
        "nav_discount": _r(nav_disc, 4),
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
