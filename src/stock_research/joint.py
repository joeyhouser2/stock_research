"""Cross-asset joint return model — where a generative model beats per-name GARCH.

GARCH is fit one name at a time, so a portfolio built from independent GARCH
samples assumes the names are independent. They aren't: large-caps crash together.
That independence assumption **understates joint tail risk** — it over-credits
diversification — so an independent-GARCH portfolio VaR is breached too often.

This module models the **joint** distribution with a dynamic factor model:

    r_it = drift + beta_i * f_t + idio_it

where ``f_t`` is a common market factor whose fat-tailed, vol-clustering dynamics
are learned by the TCN (:mod:`generative`), ``beta_i`` is each name's loading, and
``idio_it`` is a per-name Student-t idiosyncratic shock. The shared ``f_t`` induces
realistic cross-sectional correlation and tail co-movement that independent GARCH
cannot produce — so the portfolio tails come out right.

``compare_joint`` trains the factor model on early history (frozen) and scores it
against a per-name refit GARCH on held-out windows, on two scorecards:

* **Portfolio** — VaR coverage (Kupiec) and pinball on the basket return: joint
  model vs independent GARCH. This is the decisive comparison.
* **Single-name (terminal)** — VaR / pinball / terminal calibration of the joint
  model's marginals vs GARCH, to confirm the joint model doesn't sacrifice the
  per-name fit it ties on.

Drift is held flat and identical for both models, so the only thing under test is
the dependence structure. Requires PyTorch (via :mod:`generative`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import backtest, generative, simulate

TRADING_DAYS = 252.0


@dataclass
class JointFactorModel:
    tickers: list[str]
    factor_weights: np.ndarray      # weights defining the common factor
    betas: np.ndarray               # per-name factor loading
    idio_std: np.ndarray            # per-name idiosyncratic daily std
    idio_dof: np.ndarray            # per-name idiosyncratic Student-t dof
    factor_tcn: generative.TCNModel
    index: dict                     # ticker -> column position

    @property
    def device(self) -> str:
        return self.factor_tcn.device

    @property
    def final_loss(self) -> float:
        return self.factor_tcn.final_loss


def _dof_from_kurtosis(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    sd = x.std()
    if sd <= 0:
        return 30.0
    g = float(np.mean(((x - x.mean()) / sd) ** 4) - 3.0)
    if g <= 0.1:
        return 30.0
    return float(min(max(6.0 / g + 4.0, 3.0), 30.0))


def fit_factor_model(
    train_rets: np.ndarray,
    tickers: list[str],
    *,
    factor_weights: np.ndarray | None = None,
    seed: int = 0,
    force_cpu: bool = False,
    tcn_hp: dict | None = None,
    verbose: bool = False,
) -> JointFactorModel:
    """Fit betas, idiosyncratic tails, and the TCN factor dynamics on a return matrix.

    ``train_rets`` is (T, N) daily log returns. The common factor is the
    (equal- or supplied-weight) cross-sectional average return.
    """
    n = train_rets.shape[1]
    w = np.full(n, 1.0 / n) if factor_weights is None else np.asarray(factor_weights, float)
    factor = train_rets @ w
    var_f = float(np.var(factor))
    var_f = var_f if var_f > 0 else 1e-12

    betas = np.array([float(np.cov(train_rets[:, j], factor)[0, 1] / var_f)
                      for j in range(n)])
    resid = train_rets - np.outer(factor, betas)
    idio_std = resid.std(axis=0)
    idio_std[idio_std <= 0] = 1e-8
    idio_dof = np.array([_dof_from_kurtosis(resid[:, j]) for j in range(n)])

    factor_tcn = generative.fit_tcn(factor, seed=seed, force_cpu=force_cpu,
                                    verbose=verbose, **(tcn_hp or {}))
    if verbose:
        print(f"  factor model: {n} names, mean beta {betas.mean():.2f}, "
              f"factor vol {factor.std()*math.sqrt(TRADING_DAYS):.1%}/yr")
    return JointFactorModel(tickers=list(tickers), factor_weights=w, betas=betas,
                            idio_std=idio_std, idio_dof=idio_dof, factor_tcn=factor_tcn,
                            index={t: i for i, t in enumerate(tickers)})


def sample_joint_terminals(
    model: JointFactorModel,
    context_rets: np.ndarray,
    horizon_days: int,
    n_paths: int,
    *,
    mu: float = 0.085,
    seed: int = 0,
) -> np.ndarray:
    """Sample joint terminal simple returns, shape (n_paths, n_names).

    The shared factor path is sampled once per path and loaded onto every name
    (``beta_i``); idiosyncratic shocks are independent across names.
    """
    h = max(1, round(horizon_days * TRADING_DAYS / 365.0))
    factor_ctx = context_rets @ model.factor_weights
    fpaths = generative.sample_paths(model.factor_tcn, factor_ctx, h, n_paths,
                                     mu=0.0, seed=seed)           # (n_paths, h)
    fsum = fpaths.sum(axis=1)                                     # common shock per path

    rng = np.random.default_rng(seed)
    n = len(model.tickers)
    drift = mu / TRADING_DAYS * h
    term_log = np.empty((n_paths, n))
    for j in range(n):
        dof = model.idio_dof[j]
        eps = (rng.standard_t(dof, size=(n_paths, h))
               * math.sqrt((dof - 2.0) / dof) * model.idio_std[j])
        term_log[:, j] = drift + model.betas[j] * fsum + eps.sum(axis=1)
    return np.exp(term_log) - 1.0


def _indep_terminals(context_rets, spots, horizon_days, n_paths, mu, seed, force_cpu):
    """Independent per-name GARCH terminal simple returns, (n_paths, n_names)."""
    n = context_rets.shape[1]
    out = np.empty((n_paths, n))
    for j in range(n):
        sim = simulate.simulate(spot=float(spots[j]), returns=context_rets[:, j],
                                horizon_days=horizon_days, model="garch", mu=mu,
                                n_paths=n_paths, seed=seed + j, force_cpu=force_cpu)
        out[:, j] = sim.terminal / float(spots[j]) - 1.0
    return out


# --------------------------------------------------------------------------- #
# Scoring helpers (reuse the backtest scorers on terminal-return records).
# --------------------------------------------------------------------------- #
_Q = backtest.DEFAULT_QUANTILES


def _terminal_records(sample_simple, realized_simple, *, target_pct=None):
    """One record per path-sample column set: forecast quantiles/VaR + realized."""
    rec = {
        "realized_return": float(realized_simple),
        "var": float(-np.quantile(sample_simple, 0.05)),
        "q_pred": {q: float(np.quantile(sample_simple, q)) for q in _Q},
    }
    if target_pct is not None:
        rec["prob_term_up"] = float(np.mean(sample_simple >= target_pct))
        rec["finished_up"] = bool(realized_simple >= target_pct)
    return rec


def _score_portfolio(records, var_level):
    return {"var": backtest.var_coverage(records, level=var_level),
            "pinball": backtest.pinball_loss(records)}


def _score_single(records, var_level):
    return {"var": backtest.var_coverage(records, level=var_level),
            "pinball": backtest.pinball_loss(records),
            "term_up": backtest.reliability(records, "prob_term_up", "finished_up")}


# --------------------------------------------------------------------------- #
# The decisive comparison.
# --------------------------------------------------------------------------- #
def compare_joint(
    closes_by_ticker: dict,
    *,
    weights: np.ndarray | None = None,
    train_frac: float = 0.7,
    horizon_days: int,
    target_pct: float = 0.05,
    lookback: int = 504,
    n_paths: int = 20_000,
    mu: float = 0.085,
    var_level: float = 0.95,
    seed: int = 0,
    force_cpu: bool = False,
    tcn_hp: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Train the joint factor model on early history, then score it (frozen) vs a
    per-name refit GARCH on held-out windows — portfolio and single-name scorecards.
    """
    df = pd.DataFrame(closes_by_ticker).dropna()       # align on common dates
    if df.shape[1] < 2 or len(df) < lookback + 30:
        raise ValueError("need >= 2 names and enough overlapping history")
    tickers = list(df.columns)
    prices = df.to_numpy()
    dates = df.index
    rets = np.diff(np.log(prices), axis=0)             # (T-1, N); rets[k] ends at date k+1
    n = len(tickers)
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, float)

    split_pos = int(np.searchsorted(dates, dates[0] + (dates[-1] - dates[0]) * train_frac))
    if split_pos <= lookback:
        raise ValueError("train portion too short for the lookback")
    model = fit_factor_model(rets[:split_pos - 1], tickers, factor_weights=w,
                             seed=seed, force_cpu=force_cpu, tcn_hp=tcn_hp, verbose=verbose)

    h = max(1, round(horizon_days * TRADING_DAYS / 365.0))
    port = {"joint": [], "indep": []}
    single = {"joint": [], "garch": []}

    i = max(split_pos, lookback)
    n_windows = 0
    while i + h < len(prices):
        ctx = rets[i - lookback:i]                     # (lookback, N) context per name
        spots = prices[i]
        realized = prices[i + h] / prices[i] - 1.0     # (N,) realized simple returns

        joint_t = sample_joint_terminals(model, ctx, horizon_days, n_paths, mu=mu, seed=seed + i)
        indep_t = _indep_terminals(ctx, spots, horizon_days, n_paths, mu, seed + i, force_cpu)

        port["joint"].append(_terminal_records(joint_t @ w, realized @ w))
        port["indep"].append(_terminal_records(indep_t @ w, realized @ w))
        for j in range(n):
            single["joint"].append(_terminal_records(joint_t[:, j], realized[j], target_pct=target_pct))
            single["garch"].append(_terminal_records(indep_t[:, j], realized[j], target_pct=target_pct))

        n_windows += 1
        i += h
        if verbose:
            print(f"  window {n_windows}: asof {str(dates[i-h])[:10]}  "
                  f"port realized {float(realized @ w):+.2%}")

    return {
        "model": model,
        "n_tickers": n,
        "windows": n_windows,
        "portfolio": {"independent": _score_portfolio(port["indep"], var_level),
                      "joint": _score_portfolio(port["joint"], var_level)},
        "single_name": {"garch": _score_single(single["garch"], var_level),
                        "joint": _score_single(single["joint"], var_level)},
    }
