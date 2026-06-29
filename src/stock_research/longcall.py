"""Vega-aware P&L simulation for *buying* a call (the long side).

Selling calls earns the volatility risk premium; buying calls pays it, so a long
call only makes sense when you have a directional view or the option's vol is
genuinely cheap. This module gives the honest odds for that bet.

It simulates the joint (price, volatility) path with the GARCH stochastic-vol
model (already fit and validated elsewhere), then values the call along each path:

* **Held to expiry** → terminal payoff ``max(S_T − K, 0)`` (vega irrelevant).
* **Exited early** → Black–Scholes value with the *simulated* volatility, so the
  P&L captures vega / vol expansion.

Because free data has no historical implied vol, we anchor the IV *level* to the
contract's IV observed right now (today's IV/RV ratio) and let it evolve with the
simulated vol process — an explicit calibration, not a forecast of IV. (GARCH is
the vol engine here; a neural-SV / TCN engine can be swapped in later.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from . import simulate

TRADING_DAYS = 252.0


def _bs_call_vec(S, K, t_years, vol, r, q=0.0):
    """Vectorized Black–Scholes call value. ``S`` and ``vol`` are arrays."""
    S = np.asarray(S, dtype=float)
    vol = np.asarray(vol, dtype=float)
    intrinsic = np.maximum(S - K, 0.0)
    if t_years <= 0:
        return intrinsic
    valid = (vol > 0) & (S > 0)
    out = intrinsic.copy()
    if np.any(valid):
        Sv, vv = S[valid], vol[valid]
        srt = vv * math.sqrt(t_years)
        d1 = (np.log(Sv / K) + (r - q + 0.5 * vv * vv) * t_years) / srt
        d2 = d1 - srt
        out[valid] = (Sv * math.exp(-q * t_years) * norm.cdf(d1)
                      - K * math.exp(-r * t_years) * norm.cdf(d2))
    return out


@dataclass
class LongCallResult:
    spot: float
    strike: float
    premium: float
    dte: int
    hold_days: int
    days_remaining: int
    iv0: float
    rv0: float
    iv_rv: float
    r: float
    backend: str
    call_value: np.ndarray        # simulated exit value per path

    def prob_profit(self) -> float:
        return float(np.mean(self.call_value > self.premium))

    def prob_multiple(self, x: float) -> float:
        return float(np.mean(self.call_value >= x * self.premium))

    def prob_total_loss(self, frac: float = 0.1) -> float:
        """Probability of losing ~everything (exit value <= frac of premium)."""
        return float(np.mean(self.call_value <= frac * self.premium))

    def expected_return(self) -> float:
        return float(np.mean(self.call_value) / self.premium - 1.0)

    def mean_value(self) -> float:
        return float(self.call_value.mean())

    def value_quantiles(self, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
        return {q: float(np.quantile(self.call_value, q)) for q in qs}

    @property
    def breakeven_spot(self) -> float:
        return self.strike + self.premium


def simulate_long_call(
    *,
    spot: float,
    strike: float,
    premium: float,
    dte: int,
    current_iv: float | None,
    returns: np.ndarray | None,
    hold_days: int | None = None,
    mu: float = 0.04,
    r: float = 0.04,
    dividend_yield: float = 0.0,
    garch=None,
    n_paths: int = 50_000,
    seed: int = 0,
    force_cpu: bool = False,
) -> LongCallResult:
    """Simulate the P&L distribution of buying one call to ``hold_days`` (default: expiry)."""
    if premium is None or premium <= 0:
        raise ValueError("need a positive premium")
    hold_days = dte if hold_days is None else min(hold_days, dte)
    hold_steps = max(1, round(hold_days * TRADING_DAYS / 365.0))
    days_remaining = max(dte - hold_days, 0)

    rv0 = _annualized_vol(returns)
    iv0 = float(current_iv) if (current_iv and current_iv > 0) else rv0
    iv_rv = min(max(iv0 / rv0, 0.3), 3.0) if rv0 > 0 else 1.0   # IV/RV anchor, clamped

    if garch is None and returns is not None and len(returns):
        garch = simulate.fit_garch(returns)

    b = simulate.select_backend(seed, force_cpu=force_cpu)
    price = b.full(n_paths, spot)
    mu_daily = mu / TRADING_DAYS
    if garch is not None:
        var = b.full(n_paths, garch.last_var)
        eps2 = b.full(n_paths, garch.last_eps2)
        for _ in range(hold_steps):
            var = garch.omega + garch.alpha * eps2 + garch.beta * var
            z = b.student_t(n_paths, garch.nu) if garch.dist != "normal" else b.normal(n_paths)
            eps = b.sqrt(var) * z
            eps2 = eps * eps
            price = price * b.exp(mu_daily + eps)
        s_exit = b.to_numpy(price)
        vol_exit = np.sqrt(b.to_numpy(var) * TRADING_DAYS)      # conditional vol at exit
    else:                                                       # constant-vol fallback
        daily = rv0 / math.sqrt(TRADING_DAYS)
        for _ in range(hold_steps):
            price = price * b.exp(mu_daily - 0.5 * daily * daily + daily * b.normal(n_paths))
        s_exit = b.to_numpy(price)
        vol_exit = np.full(n_paths, rv0)

    iv_exit = vol_exit * iv_rv                                  # anchor the level to today's IV
    if days_remaining <= 0:
        call_value = np.maximum(s_exit - strike, 0.0)           # at expiry: pure intrinsic
    else:
        call_value = _bs_call_vec(s_exit, strike, days_remaining / 365.0, iv_exit, r, dividend_yield)

    return LongCallResult(
        spot=spot, strike=strike, premium=premium, dte=dte, hold_days=hold_days,
        days_remaining=days_remaining, iv0=iv0, rv0=rv0, iv_rv=iv_rv, r=r,
        backend=b.name, call_value=call_value,
    )


def _annualized_vol(returns) -> float:
    if returns is None or len(returns) < 2:
        return 0.30
    s = float(np.std(np.asarray(returns, dtype=float), ddof=1))
    return s * math.sqrt(TRADING_DAYS) if math.isfinite(s) and s > 0 else 0.30


# --------------------------------------------------------------------------- #
# Orchestration + reporting.
# --------------------------------------------------------------------------- #
def run(
    ticker: str,
    settings,
    *,
    expiry: str,
    strike: float,
    hold_days: int | None = None,
    n_paths: int = 50_000,
    seed: int = 0,
    force_cpu: bool = False,
) -> tuple[LongCallResult, dict]:
    """Look up the live contract, then simulate buying it."""
    from . import data, expected_return, metrics

    snap = data.get_snapshot(ticker)
    if snap is None:
        raise ValueError(f"No price/size data for {ticker!r}.")
    chain = data.get_call_chain(ticker, expiry)
    if chain.empty:
        raise ValueError(f"No call chain for {ticker} {expiry}.")
    row = chain[np.isclose(chain["strike"].astype(float), float(strike))]
    if row.empty:
        raise ValueError(f"No strike {strike} in {ticker} {expiry}.")
    row = row.iloc[0]
    premium = metrics.mid_price(row.get("bid"), row.get("ask"), row.get("lastPrice"))
    if premium is None or premium <= 0:
        raise ValueError(f"No usable premium for {ticker} {expiry} {strike}.")
    iv = float(row.get("impliedVolatility") or 0.0)
    dte = data.days_to_expiry(expiry)
    returns = data.daily_log_returns(ticker, 504)
    drift = expected_return.estimate(
        snap.info, snap.price, snap.dividend_yield, model=settings.drift_model,
        rf=settings.risk_free_rate, erp=settings.equity_risk_premium,
        pe_anchor=settings.pe_anchor, reversion_years=settings.pe_reversion_years,
        shrink=settings.pe_reversion_shrink)

    result = simulate_long_call(
        spot=snap.price, strike=float(strike), premium=float(premium), dte=dte,
        current_iv=iv, returns=returns, hold_days=hold_days, mu=drift.annual_drift,
        r=settings.risk_free_rate, dividend_yield=snap.dividend_yield, n_paths=n_paths,
        seed=seed, force_cpu=force_cpu)
    report = {"ticker": ticker, "expiry": expiry, "drift": drift,
              "contract": row.get("contractSymbol")}
    return result, report


def render_text(result: LongCallResult, report: dict) -> str:
    rr = result
    rich_cheap = ("CHEAP" if rr.iv_rv < 0.95 else "RICH" if rr.iv_rv > 1.05 else "fair")
    lines = [
        f"\n{report['ticker']}  buy {rr.strike:g} call exp {report['expiry']}  "
        f"({report.get('contract','')})",
        f"  spot ${rr.spot:.2f}   premium ${rr.premium:.2f}   breakeven ${rr.breakeven_spot:.2f}   "
        f"dte {rr.dte}d   hold {rr.hold_days}d",
        f"  IV {rr.iv0:.1%}  vs realized {rr.rv0:.1%}  ->  IV/RV {rr.iv_rv:.2f} [{rich_cheap}]   "
        f"backend {rr.backend}",
    ]
    if report.get("drift") is not None:
        lines.append(f"  drift: {report['drift'].summary()}")
    q = result.value_quantiles()
    lines.append(
        f"\n  P&L of buying it:\n"
        f"    P(profit) {rr.prob_profit():.1%}   P(2x) {rr.prob_multiple(2):.1%}   "
        f"P(lose >=90%) {rr.prob_total_loss():.1%}\n"
        f"    expected return {rr.expected_return():+.1%}   "
        f"(max loss -100% = ${rr.premium:.2f})\n"
        f"    exit value  p05 ${q[0.05]:.2f}  median ${q[0.5]:.2f}  p95 ${q[0.95]:.2f}   "
        f"(premium ${rr.premium:.2f})")
    return "\n".join(lines)
