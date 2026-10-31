"""Model-free risk-neutral probabilities extracted directly from call prices.

Black-Scholes ``prob_otm`` (see :mod:`blackscholes`) assumes one IV per strike
plugged into a lognormal spot distribution. Breeden & Litzenberger (1978) showed
you don't need a model at all: the risk-neutral CDF is recoverable straight from
the *shape* of observed call prices across strikes,

    Q(S_T > K) = -e^{rT} * dC/dK

so differentiating the market's own call-price curve gives the market's own
consensus probability distribution for the underlying at expiry, with no
volatility model in between. This is the natural benchmark for the GARCH/BS
estimates elsewhere in this package: where they disagree with the market's
implied distribution is exactly where a sell-side screen is finding real edge
(or is about to get run over by one).
"""

from __future__ import annotations

import math

import pandas as pd

from . import metrics


def implied_prob_otm(chain: pd.DataFrame, dte: int, risk_free_rate: float) -> dict[float, float]:
    """Market-implied P(S_T <= K) per strike, via finite differences on call mid prices.

    Uses every strike in ``chain`` with a usable bid/ask/last (not just the OTM
    subset), since the derivative at strike K needs its neighbors on both sides.
    Interior strikes use a central difference; the two end strikes use a
    one-sided difference and are less reliable (no price data beyond the
    sampled range). Returns {} if fewer than two strikes have usable prices.
    """
    if dte <= 0 or chain.empty:
        return {}
    t = dte / 365.0

    points: list[tuple[float, float]] = []
    seen_strikes: set[float] = set()
    for _, row in chain.iterrows():
        strike = _num(row.get("strike"))
        if strike is None or strike in seen_strikes:
            continue
        mid = metrics.mid_price(_num(row.get("bid")), _num(row.get("ask")), _num(row.get("lastPrice")))
        if mid is None or mid <= 0:
            continue
        seen_strikes.add(strike)
        points.append((strike, mid))
    if len(points) < 2:
        return {}
    points.sort(key=lambda p: p[0])

    strikes = [p[0] for p in points]
    prices = [p[1] for p in points]
    disc = math.exp(risk_free_rate * t)
    n = len(points)

    out: dict[float, float] = {}
    for i in range(n):
        if i == 0:
            lo, hi = 0, 1
        elif i == n - 1:
            lo, hi = n - 2, n - 1
        else:
            lo, hi = i - 1, i + 1
        dcdk = (prices[hi] - prices[lo]) / (strikes[hi] - strikes[lo])
        q_above = max(0.0, min(1.0, -disc * dcdk))
        out[strikes[i]] = round(1.0 - q_above, 4)
    return out


def _num(val) -> float | None:
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None
