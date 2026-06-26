"""Valuation-conditioned expected return — the real-world drift for simulation.

The Monte-Carlo engine needs an annualized drift ``mu``. Feeding it a flat
risk-free rate ignores everything we know about the underlying; this module
estimates a *forward-looking* drift from fundamentals so a richly-valued name
drifts differently from a cheap one.

The workhorse is the **Grinold-Kroner** decomposition of equity return:

    E[R]  =  income (dividend yield)
           + earnings growth
           + valuation re-rating  (P/E mean-reversion toward an anchor)

Two alternatives are offered: an **analyst-target-implied** return, and a
**blend** of the two. Everything is reported component-by-component so the drift
is never a black box.

Honesty note: at *short* horizons (days to weeks — the options use case) the
drift term is tiny next to volatility (drift scales with t, vol with sqrt(t)),
so this mostly matters at the margin and grows in importance with the horizon.
Single-name fundamental drift is also noisy; ETFs and loss-making names lack the
inputs and degrade gracefully to a market baseline (risk-free + equity premium).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Sane bounds so a bad data point can't produce an absurd drift. Live testing
# showed the unbounded model overstating drift on hypergrowth names (a +60%
# earnings-growth read flowed straight through) and over-penalizing low-growth
# quality / index names (the PEG=1 anchor mispriced them), so growth and the
# reversion term are both capped.
_MU_FLOOR, _MU_CAP = -0.50, 1.00
_GROWTH_FLOOR, _GROWTH_CAP = -0.25, 0.25
_REVERSION_CAP = 0.15          # max annual tailwind/drag from P/E reversion
_ANCHOR_FLOOR, _ANCHOR_CAP = 8.0, 40.0
_DEFAULT_MARKET_PE = 18.0


@dataclass
class DriftEstimate:
    """An annualized drift plus the components that produced it."""

    annual_drift: float
    model: str
    components: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        c = self.components
        parts = []
        if "income" in c:
            parts.append(f"income {c['income']:+.1%}")
        if "growth" in c:
            parts.append(f"growth {c['growth']:+.1%}")
        if "reversion" in c:
            parts.append(f"reversion {c['reversion']:+.1%} "
                         f"(P/E {c.get('trailing_pe', float('nan')):.1f} vs anchor "
                         f"{c.get('pe_anchor', float('nan')):.1f})")
        if "analyst_implied" in c:
            parts.append(f"analyst {c['analyst_implied']:+.1%}")
        return f"mu={self.annual_drift:+.1%}/yr [{self.model}]" + (
            "  =  " + " + ".join(parts) if parts else "")


# --------------------------------------------------------------------------- #
# Component helpers.
# --------------------------------------------------------------------------- #
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


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def earnings_growth(info: dict) -> float | None:
    """Best available annual earnings-growth estimate, clamped to a sane band.

    Prefers forward-vs-trailing EPS, then Yahoo's reported growth fields.
    """
    feps, teps = _num(info, "forwardEps"), _num(info, "trailingEps")
    if feps is not None and teps is not None and teps > 0:
        return _clamp((feps - teps) / teps, _GROWTH_FLOOR, _GROWTH_CAP)
    for key in ("earningsGrowth", "earningsQuarterlyGrowth", "revenueGrowth"):
        g = _num(info, key)
        if g is not None:
            return _clamp(g, _GROWTH_FLOOR, _GROWTH_CAP)
    return None


def _pe_anchor(growth: float | None, override: float | None) -> tuple[float, str]:
    """The 'normal' P/E the current multiple is assumed to revert toward."""
    if override is not None:
        return override, "user"
    if growth is not None and growth > 0:
        # PEG = 1 fair value: a g% grower 'deserves' a P/E of ~100g.
        return _clamp(100.0 * growth, _ANCHOR_FLOOR, _ANCHOR_CAP), "peg1"
    return _DEFAULT_MARKET_PE, "market"


# --------------------------------------------------------------------------- #
# Models.
# --------------------------------------------------------------------------- #
def grinold_kroner(
    info: dict,
    dividend_yield: float,
    *,
    rf: float,
    erp: float = 0.045,
    pe_anchor: float | None = None,
    reversion_years: float = 5.0,
    shrink: float = 1.0,
) -> DriftEstimate:
    """Grinold-Kroner expected return: income + growth + P/E re-rating.

    ``shrink`` scales only the (noisy, weak-at-short-horizon) reversion term.
    When earnings growth is unavailable the growth term degrades so the drift
    sans-reversion equals the market baseline ``rf + erp``; when there's no P/E
    at all (e.g. an ETF) the reversion term is zero.
    """
    notes: list[str] = []
    income = float(dividend_yield or 0.0)
    g = earnings_growth(info)

    # Growth term: real estimate if we have one, else a baseline stand-in so the
    # no-reversion drift lands at rf + erp rather than just the dividend yield.
    if g is not None:
        growth = g
    else:
        growth = (rf + erp) - income
        notes.append("no earnings-growth data -> growth set to market baseline (rf + erp)")

    trailing_pe = _num(info, "trailingPE")
    anchor, anchor_src = _pe_anchor(g, pe_anchor)
    if trailing_pe is not None and trailing_pe > 0 and reversion_years > 0:
        # Exponential reversion of the multiple toward the anchor over the
        # reversion horizon; annualized. Above anchor -> drag, below -> tailwind.
        raw_reversion = -math.log(trailing_pe / anchor) / reversion_years * shrink
        reversion = _clamp(raw_reversion, -_REVERSION_CAP, _REVERSION_CAP)
        if reversion != raw_reversion:
            notes.append(f"reversion clamped to {reversion:+.0%}/yr "
                         f"(raw {raw_reversion:+.0%} from P/E {trailing_pe:.1f} vs anchor {anchor:.1f})")
    else:
        reversion = 0.0
        trailing_pe = float("nan")
        notes.append("no trailing P/E (e.g. ETF) -> no valuation-reversion term")

    raw = income + growth + reversion
    mu = _clamp(raw, _MU_FLOOR, _MU_CAP)
    return DriftEstimate(
        annual_drift=mu,
        model="fundamental",
        components={
            "income": income, "growth": growth, "reversion": reversion,
            "trailing_pe": trailing_pe, "pe_anchor": anchor, "anchor_source": anchor_src,
            "shrink": shrink, "raw": raw,
        },
        notes=notes,
    )


def analyst_implied(
    info: dict,
    price: float,
    *,
    rf: float,
    erp: float = 0.045,
    shrink: float = 0.5,
) -> DriftEstimate | None:
    """Annualized return implied by the mean analyst target, shrunk toward the
    market baseline (targets are optimistically biased). ``None`` if no target."""
    target = _num(info, "targetMeanPrice")
    if target is None or price <= 0:
        return None
    implied = target / price - 1.0          # ~12-month horizon
    baseline = rf + erp
    mu = _clamp(baseline + shrink * (implied - baseline), _MU_FLOOR, _MU_CAP)
    return DriftEstimate(
        annual_drift=mu,
        model="analyst",
        components={"analyst_implied": implied, "baseline": baseline, "shrink": shrink},
        notes=[],
    )


def estimate(
    info: dict,
    price: float,
    dividend_yield: float,
    *,
    model: str = "fundamental",
    rf: float = 0.04,
    erp: float = 0.045,
    pe_anchor: float | None = None,
    reversion_years: float = 5.0,
    shrink: float = 1.0,
) -> DriftEstimate:
    """Dispatch to the requested drift model.

    ``model`` is one of ``fixed`` (baseline rf + erp), ``fundamental``,
    ``analyst`` or ``blend``. Unavailable models fall back gracefully.
    """
    if model == "fixed":
        return DriftEstimate(rf + erp, "fixed",
                             {"baseline": rf + erp}, ["flat market baseline (rf + erp)"])

    fund = grinold_kroner(info, dividend_yield, rf=rf, erp=erp, pe_anchor=pe_anchor,
                          reversion_years=reversion_years, shrink=shrink)
    if model == "fundamental":
        return fund

    ana = analyst_implied(info, price, rf=rf, erp=erp)
    if model == "analyst":
        if ana is None:
            fund.notes.append("no analyst target -> fell back to fundamental model")
            return fund
        return ana

    if model == "blend":
        if ana is None:
            fund.notes.append("no analyst target -> blend used fundamental only")
            return fund
        mu = 0.5 * (fund.annual_drift + ana.annual_drift)
        return DriftEstimate(
            annual_drift=_clamp(mu, _MU_FLOOR, _MU_CAP),
            model="blend",
            components={"fundamental": fund.annual_drift, "analyst": ana.annual_drift,
                        **{f"f_{k}": v for k, v in fund.components.items()}},
            notes=fund.notes,
        )

    raise ValueError(f"unknown drift model {model!r}")
