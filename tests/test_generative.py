"""Tests for the autoregressive TCN generative model.

Skipped entirely when PyTorch isn't installed. Models are kept tiny and trained
on CPU for a few hundred steps so the suite stays fast.
"""

import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from stock_research import generative as gen  # noqa: E402


def _garch_returns(n=4000, omega=2e-6, alpha=0.08, beta=0.90, seed=0):
    """Synthetic GARCH(1,1)-normal returns: fat tails + volatility clustering."""
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    var = omega / (1 - alpha - beta)
    eps = 0.0
    for i in range(n):
        var = omega + alpha * eps**2 + beta * var
        eps = math.sqrt(var) * rng.standard_normal()
        r[i] = eps
    return r


_HP = dict(channels=16, dilations=(1, 2, 4, 8), window=48, batch=64, steps=300)


def test_fit_reduces_loss():
    r = _garch_returns(seed=1)
    # Compare final training NLL to the loss of the untrained net on the same data.
    model = gen.fit_tcn(r, steps=1, seed=0, force_cpu=True, **{k: v for k, v in _HP.items() if k != "steps"})
    untrained = model.final_loss
    trained = gen.fit_tcn(r, seed=0, force_cpu=True, **_HP).final_loss
    assert trained < untrained
    assert math.isfinite(trained)


def test_sample_paths_shape_and_finiteness():
    r = _garch_returns(seed=2)
    model = gen.fit_tcn(r, seed=0, force_cpu=True, **_HP)
    paths = gen.sample_paths(model, r, horizon=20, n_paths=512, mu=0.05, seed=3)
    assert paths.shape == (512, 20)
    assert np.all(np.isfinite(paths))
    # Daily moves are small and centered near the per-step drift, not exploding.
    assert abs(paths.mean()) < 0.02
    assert paths.std() < 0.2


def test_sample_simulation_integration():
    r = _garch_returns(seed=4)
    model = gen.fit_tcn(r, seed=0, force_cpu=True, **_HP)
    sim = gen.sample_simulation(model, spot=100.0, context_returns=r, horizon_days=30,
                                mu=0.05, n_paths=4000, seed=5)
    assert sim.model == "tcn"
    assert sim.terminal.shape == (4000,)
    # The Simulation API works on the learned paths.
    assert 0.0 <= sim.prob_touch(105.0) <= 1.0
    assert sim.prob_touch(105.0) >= sim.prob_terminal_above(105.0)   # touch >= terminal
    assert sim.var(0.95) > 0
    assert 0.0 < sim.mean_max_drawdown() < 1.0


def test_learns_fat_tails():
    # Trained on fat-tailed GARCH data, sampled one-step innovations should be
    # leptokurtic (excess kurtosis > 0), not Gaussian.
    r = _garch_returns(n=5000, seed=6)
    model = gen.fit_tcn(r, seed=0, force_cpu=True, **dict(_HP, steps=500))
    paths = gen.sample_paths(model, r, horizon=1, n_paths=20000, mu=0.0, seed=7)
    x = paths[:, 0]
    x = (x - x.mean()) / x.std()
    excess_kurt = float(np.mean(x**4) - 3.0)
    assert excess_kurt > 0.3


def test_short_context_is_padded():
    r = _garch_returns(seed=8)
    model = gen.fit_tcn(r, seed=0, force_cpu=True, **_HP)
    # Fewer returns than the receptive field -> left-padded, still works.
    paths = gen.sample_paths(model, r[:5], horizon=10, n_paths=64, seed=9)
    assert paths.shape == (64, 10) and np.all(np.isfinite(paths))


# --- pooled training -------------------------------------------------------- #

def test_pooled_fit_stores_scalers_and_respects_them():
    a = _garch_returns(n=2500, omega=2e-6, seed=10)   # ~1% daily vol
    b = _garch_returns(n=2500, omega=8e-6, seed=11)   # ~2% daily vol
    model = gen.fit_tcn_pooled({"A": a, "B": b}, seed=0, force_cpu=True,
                               **dict(_HP, steps=300))
    assert set(model.scalers) == {"A", "B"}
    assert model.scalers["B"][1] > model.scalers["A"][1]   # B is the higher-vol name

    # Same shared net, but each name's scaler scales the output: B paths are wider.
    pa = gen.sample_paths(model, a, horizon=20, n_paths=1000,
                          scaler=model.scalers["A"], seed=1)
    pb = gen.sample_paths(model, b, horizon=20, n_paths=1000,
                          scaler=model.scalers["B"], seed=1)
    assert np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))
    assert pb.std() > 1.5 * pa.std()


def _synth_closes(n=1200, vol=0.012, drift=0.0003, s0=100.0, seed=0):
    rng = np.random.default_rng(seed)
    r = drift + vol * rng.standard_normal(n)
    prices = s0 * np.exp(np.cumsum(r))
    return pd.Series(prices, index=pd.bdate_range("2015-01-01", periods=n))


def test_compare_universe_runs_and_scores():
    closes = {"A": _synth_closes(seed=1), "B": _synth_closes(vol=0.018, seed=2)}
    res = gen.compare_universe(
        closes, train_frac=0.6, horizon_days=20, lookback=150, n_paths=2000,
        mu=0.05, seed=0, force_cpu=True, tcn_hp=dict(_HP, steps=120))
    assert res["n_tickers"] == 2 and res["windows"] > 20
    for key in ("garch", "tcn"):
        assert 0.0 <= res[key]["var"]["observed_rate"] <= 1.0
        assert math.isfinite(res[key]["pinball"]["overall"])
