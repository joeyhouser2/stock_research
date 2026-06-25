"""Per-contract statistics for selling an OTM call at market prices.

The input is a single row of a yfinance call chain plus the underlying context
(spot price, days to expiry, realized vol, dividend yield, risk-free rate). The
output is a flat dict ready to drop into a DataFrame row.
"""

from __future__ import annotations

import math

from . import blackscholes as bs

DAYS_PER_YEAR = 365.0


def mid_price(bid: float, ask: float, last: float) -> float | None:
    """Best estimate of where a sale fills: bid/ask mid, falling back to last."""
    bid = bid if bid and bid > 0 else None
    ask = ask if ask and ask > 0 else None
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if last and last > 0:
        return float(last)
    return bid or ask


def spread_pct(bid: float, ask: float) -> float | None:
    """(ask - bid) / mid; a liquidity proxy. None if quotes are missing."""
    if not bid or not ask or bid <= 0 or ask <= 0:
        return None
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid if mid > 0 else None


def annualize(period_return: float, days: int) -> float:
    """Scale a holding-period return to a year (simple, not compounded)."""
    if days <= 0:
        return 0.0
    return period_return * (DAYS_PER_YEAR / days)


def compute(
    *,
    row,
    spot: float,
    dte: int,
    risk_free_rate: float,
    hv: float | None,
    dividend_yield: float = 0.0,
) -> dict | None:
    """Compute the full stat set for one call contract.

    ``row`` is a pandas Series from a yfinance call chain. Returns ``None`` if the
    contract is in/at the money or has no usable price (it can't be an OTM sale).
    """
    strike = float(row.get("strike", float("nan")))
    if not math.isfinite(strike) or strike <= spot:
        return None  # not out of the money

    bid = _num(row.get("bid"))
    ask = _num(row.get("ask"))
    last = _num(row.get("lastPrice"))
    premium = mid_price(bid, ask, last)
    if premium is None or premium <= 0:
        return None

    iv = _num(row.get("impliedVolatility"))
    iv = iv if iv and iv > 0 else None
    t = max(dte, 0) / DAYS_PER_YEAR

    pct_otm = (strike - spot) / spot
    static_return = premium / spot
    if_called_return = (premium + (strike - spot)) / spot

    delta = prob_otm = None
    if iv is not None and t > 0:
        delta = bs.call_delta(spot, strike, t, iv, risk_free_rate, dividend_yield)
        prob_otm = bs.prob_otm_call(spot, strike, t, iv, risk_free_rate, dividend_yield)

    # Probability-weighted annualized yield: the static income return scaled by the
    # chance you actually keep it (call expires worthless). Balances fat premiums
    # against high assignment risk. None when IV is missing so we can't estimate odds.
    annual_yield = annualize(static_return, dte)
    score = round(annual_yield * prob_otm, 4) if prob_otm is not None else None

    return {
        "contract": row.get("contractSymbol"),
        "strike": strike,
        "spot": round(spot, 4),
        "dte": dte,
        "bid": bid,
        "ask": ask,
        "mid": round(premium, 4),
        "pct_otm": round(pct_otm, 4),
        "static_yield": round(static_return, 4),
        "annual_yield": round(annual_yield, 4),
        "score": score,
        "if_called_yield": round(annualize(if_called_return, dte), 4),
        "prob_otm": round(prob_otm, 4) if prob_otm is not None else None,
        "delta": round(delta, 4) if delta is not None else None,
        "downside_cushion": round(static_return, 4),   # premium / spot
        "breakeven": round(spot - premium, 4),
        "iv": round(iv, 4) if iv is not None else None,
        "hv": round(hv, 4) if hv is not None else None,
        "iv_hv": round(iv / hv, 3) if iv and hv else None,
        "open_interest": _int(row.get("openInterest")),
        "volume": _int(row.get("volume")),
        "spread_pct": _round(spread_pct(bid, ask), 4),
    }


def passes_liquidity(stat: dict, *, min_oi: int, min_volume: int, max_spread: float) -> bool:
    """True if the contract clears the open-interest, volume and spread filters."""
    oi = stat.get("open_interest") or 0
    vol = stat.get("volume") or 0
    spread = stat.get("spread_pct")
    if oi < min_oi:
        return False
    if vol < min_volume:
        return False
    if spread is not None and spread > max_spread:
        return False
    return True


def _num(val) -> float | None:
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _int(val) -> int:
    f = _num(val)
    return int(f) if f is not None else 0


def _round(val, ndigits):
    return round(val, ndigits) if val is not None else None
