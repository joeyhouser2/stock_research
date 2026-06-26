"""Walk-forward validation of the simulation's probability and risk forecasts.

Before trusting a model you have to ask: when it says "30% chance of touching
+5%", does that happen ~30% of the time? This harness answers that with a
no-look-ahead backtest:

    for each rebalance date t (using only data up to t):
        fit the model on the trailing window, simulate the horizon, record the
        forecast (touch/terminal probabilities, VaR, return quantiles)
    after the horizon elapses, observe what actually happened
    score the pooled forecast/outcome pairs

Scoring uses the standard tools for probabilistic forecasts:

* **Calibration** — reliability table + Brier score + expected calibration error
  (ECE) for the binary touch/terminal forecasts.
* **VaR coverage** — Kupiec proportion-of-failures test: does the realized return
  breach the VaR forecast at the nominal rate?
* **Sharpness/accuracy** — pinball (quantile) loss on the predicted return
  quantiles, a proper scoring rule.

Honesty constraint: yfinance exposes only *current* fundamentals, so the
valuation drift can't be reconstructed point-in-time without look-ahead. The
backtest therefore runs drift at a flat baseline and validates the **price
model** (GBM / t / GARCH / bootstrap touch, terminal and tail forecasts). The
realized touch is measured on daily closes, so simulated touch is read WITHOUT
the continuity correction — an apples-to-apples discrete-monitoring comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

from . import simulate

TRADING_DAYS = 252.0
DEFAULT_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def _n_steps(horizon_days: int) -> int:
    return max(1, round(horizon_days * TRADING_DAYS / 365.0))


def backtest_series(
    closes: pd.Series,
    *,
    horizon_days: int,
    target_pct: float = 0.05,
    model: str = "garch",
    lookback: int = 504,
    step: int | None = None,
    n_paths: int = 20_000,
    mu: float = 0.085,
    sigma: float | None = None,
    block: int = 5,
    var_level: float = 0.95,
    quantile_levels=DEFAULT_QUANTILES,
    seed: int = 12345,
    force_cpu: bool = False,
    ticker: str = "",
    simulate_fn=None,
) -> list[dict]:
    """Roll the model through ``closes`` and return forecast/outcome records.

    ``closes`` is a date-indexed price series. Windows are non-overlapping by
    default (``step`` = horizon in trading days), which keeps the pooled records
    statistically independent — no purging/embargo needed.

    ``simulate_fn(spot, train_returns, horizon_days, seed) -> Simulation`` swaps in
    a custom path model (e.g. a trained generative model) instead of the built-in
    ``simulate.simulate`` engine, so any model that yields a Simulation is scored
    on identical windows.
    """
    prices = np.asarray(closes, dtype=float)
    prices = prices[np.isfinite(prices)]
    h = _n_steps(horizon_days)
    step = step or h
    records: list[dict] = []

    i = lookback
    while i + h < len(prices):
        spot = prices[i]
        train = prices[i - lookback:i + 1]
        train_rets = np.diff(np.log(train))

        if simulate_fn is not None:
            sim = simulate_fn(spot, train_rets, horizon_days, seed + i)
        else:
            sim = simulate.simulate(
                spot=spot, returns=train_rets, horizon_days=horizon_days, model=model,
                mu=mu, sigma=sigma, block=block, n_paths=n_paths, seed=seed + i,
                force_cpu=force_cpu,
            )

        up, dn = spot * (1 + target_pct), spot * (1 - target_pct)
        future = prices[i:i + h + 1]              # includes spot at index 0
        realized_return = future[-1] / spot - 1.0
        run_max = np.maximum.accumulate(future)
        realized_mdd = float((1.0 - future / run_max).max())

        records.append({
            "ticker": ticker,
            "asof": closes.index[i] if i < len(closes.index) else i,
            "spot": float(spot),
            # forecasts (touch read discrete, to match daily-close realized touch)
            "prob_touch_up": sim.prob_touch(up, continuity_correct=False),
            "prob_touch_down": sim.prob_touch(dn, continuity_correct=False),
            "prob_term_up": sim.prob_terminal_above(up),
            "var": sim.var(var_level),
            "q_pred": {q: float(np.quantile(sim.returns, q)) for q in quantile_levels},
            # realized outcomes
            "realized_return": float(realized_return),
            "touched_up": bool(future.max() >= up),
            "touched_down": bool(future.min() <= dn),
            "finished_up": bool(realized_return >= target_pct),
            "realized_mdd": realized_mdd,
        })
        i += step

    return records


# --------------------------------------------------------------------------- #
# Scoring.
# --------------------------------------------------------------------------- #
@dataclass
class Reliability:
    n: int
    brier: float
    ece: float
    table: list[dict]      # per-bin: mean_pred, obs_freq, count


def reliability(records: list[dict], prob_key: str, outcome_key: str,
                n_bins: int = 10) -> Reliability:
    """Calibration of a probability forecast against a binary outcome."""
    preds = np.array([r[prob_key] for r in records], dtype=float)
    obs = np.array([float(r[outcome_key]) for r in records], dtype=float)
    n = len(preds)
    if n == 0:
        return Reliability(0, float("nan"), float("nan"), [])

    brier = float(np.mean((preds - obs) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(preds, edges[1:-1]), 0, n_bins - 1)
    table, ece = [], 0.0
    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        if not c:
            continue
        mp, of = float(preds[mask].mean()), float(obs[mask].mean())
        ece += c / n * abs(mp - of)
        table.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                      "mean_pred": mp, "obs_freq": of, "count": c})
    return Reliability(n, brier, float(ece), table)


def var_coverage(records: list[dict], *, var_key: str = "var",
                 ret_key: str = "realized_return", level: float = 0.95) -> dict:
    """Kupiec proportion-of-failures test for the VaR forecast.

    An exception is a realized loss worse than the forecast VaR. The expected
    exception rate is ``1 - level``; Kupiec tests whether the observed rate
    differs significantly (chi-square, 1 dof).
    """
    ret = np.array([r[ret_key] for r in records], dtype=float)
    var = np.array([r[var_key] for r in records], dtype=float)
    n = len(ret)
    exceptions = int(np.sum(ret < -var))
    p_exp = 1.0 - level
    if n == 0:
        return {"n": 0, "exceptions": 0, "expected_rate": p_exp,
                "observed_rate": float("nan"), "kupiec_lr": float("nan"),
                "kupiec_p": float("nan")}

    pi = exceptions / n
    lr, p_value = _kupiec(exceptions, n, p_exp)
    return {"n": n, "exceptions": exceptions, "expected_rate": p_exp,
            "observed_rate": pi, "kupiec_lr": lr, "kupiec_p": p_value}


def _kupiec(x: int, n: int, p: float) -> tuple[float, float]:
    def term(a, b):
        return a * math.log(b) if a > 0 and b > 0 else 0.0
    pi = x / n
    ll_null = term(n - x, 1 - p) + term(x, p)
    ll_alt = term(n - x, 1 - pi) + term(x, pi)
    lr = -2.0 * (ll_null - ll_alt)
    lr = max(lr, 0.0)
    return float(lr), float(1.0 - chi2.cdf(lr, 1))


def pinball_loss(records: list[dict], quantile_levels=DEFAULT_QUANTILES,
                 ret_key: str = "realized_return") -> dict:
    """Average pinball (quantile) loss per level and overall — lower is better."""
    per_level: dict[float, float] = {}
    for q in quantile_levels:
        losses = []
        for r in records:
            f = r["q_pred"].get(q)
            if f is None:
                continue
            y = r[ret_key]
            diff = y - f
            losses.append(max(q * diff, (q - 1.0) * diff))
        if losses:
            per_level[q] = float(np.mean(losses))
    overall = float(np.mean(list(per_level.values()))) if per_level else float("nan")
    return {"per_level": per_level, "overall": overall}


def score(records: list[dict], *, var_level: float = 0.95,
          quantile_levels=DEFAULT_QUANTILES) -> dict:
    """Full scorecard for a set of forecast/outcome records."""
    return {
        "n": len(records),
        "touch_up": reliability(records, "prob_touch_up", "touched_up"),
        "touch_down": reliability(records, "prob_touch_down", "touched_down"),
        "term_up": reliability(records, "prob_term_up", "finished_up"),
        "var": var_coverage(records, level=var_level),
        "pinball": pinball_loss(records, quantile_levels),
    }


# --------------------------------------------------------------------------- #
# Orchestration + reporting.
# --------------------------------------------------------------------------- #
def run(
    tickers: list[str],
    settings,
    *,
    horizon_days: int | None = None,
    target_pct: float = 0.05,
    model: str = "garch",
    lookback: int = 504,
    step: int | None = None,
    n_paths: int = 20_000,
    history_days: int = 1825,
    mu: float | None = None,
    var_level: float = 0.95,
    seed: int = 12345,
    force_cpu: bool = False,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Fetch history for each ticker, backtest, and return (records, scorecard).

    Records are pooled across tickers for calibration statistics. Drift defaults
    to a flat baseline (rf + equity premium) because point-in-time fundamentals
    aren't available — see the module docstring.
    """
    from . import data  # lazy: keeps backtest_series/scoring importable w/o network

    horizon_days = horizon_days or settings.max_dte
    if mu is None:
        mu = settings.risk_free_rate + settings.equity_risk_premium

    records: list[dict] = []
    for ticker in tickers:
        closes = data.close_history(ticker, history_days)
        if len(closes) < lookback + _n_steps(horizon_days) + 1:
            _log(verbose, f"  {ticker}: only {len(closes)} closes, need "
                          f"{lookback + _n_steps(horizon_days) + 1} - skipped")
            continue
        recs = backtest_series(
            closes, horizon_days=horizon_days, target_pct=target_pct, model=model,
            lookback=lookback, step=step, n_paths=n_paths, mu=mu, var_level=var_level,
            seed=seed, force_cpu=force_cpu, ticker=ticker,
        )
        _log(verbose, f"  {ticker}: {len(recs)} windows")
        records.extend(recs)

    return records, score(records, var_level=var_level)


def render_text(scores: dict, *, model: str, horizon_days: int, target_pct: float,
                mu: float) -> str:
    lines = [
        f"\nBacktest scorecard   model={model}   horizon={horizon_days}d   "
        f"touch band +/-{target_pct:.0%}   drift mu={mu:.1%}/yr (flat baseline)   "
        f"windows={scores['n']}",
    ]
    if scores["n"] == 0:
        lines.append("  No windows — not enough history.")
        return "\n".join(lines)

    for key, label in (("touch_up", "P(touch +band)"),
                       ("touch_down", "P(touch -band)"),
                       ("term_up", "P(finish above +band)")):
        rel = scores[key]
        lines.append(f"\n  {label} calibration   Brier {rel.brier:.4f}   "
                     f"ECE {rel.ece:.4f}   (lower = better)")
        lines.append(f"    {'pred bin':>10} {'mean_pred':>10} {'obs_freq':>10} {'count':>7}")
        for row in rel.table:
            lines.append(f"    {row['bin']:>10} {row['mean_pred']:>10.3f} "
                         f"{row['obs_freq']:>10.3f} {row['count']:>7}")

    v = scores["var"]
    flag = "OK" if (v["kupiec_p"] == v["kupiec_p"] and v["kupiec_p"] > 0.05) else "REJECT"
    lines.append(f"\n  VaR({1-v['expected_rate']:.0%}) coverage   "
                 f"expected {v['expected_rate']:.1%}   observed {v['observed_rate']:.1%}   "
                 f"({v['exceptions']}/{v['n']})   Kupiec p={v['kupiec_p']:.3f} [{flag}]")

    pb = scores["pinball"]
    lines.append(f"\n  Pinball loss (return quantiles)   overall {pb['overall']:.4f}")
    lines.append("    " + "  ".join(f"q{int(q*100):02d} {loss:.4f}"
                                     for q, loss in pb["per_level"].items()))
    return "\n".join(lines)


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)
