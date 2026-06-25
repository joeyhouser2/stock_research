"""Black-Scholes helpers for European call options.

These are used to estimate assignment probability (delta) and the probability the
call finishes out of the money, both derived from the option's implied volatility.
We treat options as European; American early-exercise on liquid index/large-cap
calls is rare except around dividends, which is acceptable for screening.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def _d1(spot: float, strike: float, t: float, vol: float, r: float, q: float = 0.0) -> float:
    return (math.log(spot / strike) + (r - q + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))


def _d2(spot: float, strike: float, t: float, vol: float, r: float, q: float = 0.0) -> float:
    return _d1(spot, strike, t, vol, r, q) - vol * math.sqrt(t)


def call_delta(spot: float, strike: float, t: float, vol: float, r: float, q: float = 0.0) -> float:
    """Black-Scholes call delta in [0, 1]; a common proxy for P(assignment)."""
    if t <= 0 or vol <= 0:
        return 1.0 if spot > strike else 0.0
    return math.exp(-q * t) * norm.cdf(_d1(spot, strike, t, vol, r, q))


def prob_otm_call(spot: float, strike: float, t: float, vol: float, r: float, q: float = 0.0) -> float:
    """Risk-neutral P(S_T < K) = N(-d2): the chance the call expires worthless.

    Uses the risk-neutral drift (r - q). With no better estimate of real-world
    drift this is the standard, conservative choice for an income writer.
    """
    if t <= 0 or vol <= 0:
        return 1.0 if spot < strike else 0.0
    return float(norm.cdf(-_d2(spot, strike, t, vol, r, q)))


def call_price(spot: float, strike: float, t: float, vol: float, r: float, q: float = 0.0) -> float:
    """Black-Scholes fair value of a European call (used only for sanity checks)."""
    if t <= 0 or vol <= 0:
        return max(spot - strike, 0.0)
    d1 = _d1(spot, strike, t, vol, r, q)
    d2 = d1 - vol * math.sqrt(t)
    return spot * math.exp(-q * t) * norm.cdf(d1) - strike * math.exp(-r * t) * norm.cdf(d2)
