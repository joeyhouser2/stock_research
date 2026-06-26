"""Unit tests for the Monte-Carlo risk / simulation module — no network access.

All tests force the NumPy (CPU) backend with a fixed seed so they're deterministic
regardless of whether PyTorch / CUDA is installed.
"""

import math

import numpy as np
import pytest

from stock_research import simulate as sim


# --- closed-form GBM references -------------------------------------------- #

def test_first_passage_at_spot_is_certain():
    # The path starts at spot, so it has already "touched" it.
    assert sim.first_passage_prob_gbm(100, 100, 0.5, 0.04, 0.3) == pytest.approx(1.0)


def test_first_passage_decreases_with_distance():
    near = sim.first_passage_prob_gbm(100, 105, 0.25, 0.04, 0.3)
    far = sim.first_passage_prob_gbm(100, 140, 0.25, 0.04, 0.3)
    assert 0.0 < far < near < 1.0


def test_touch_exceeds_terminal():
    # P(ever reaching K) must be >= P(finishing above K).
    spot, k, t, mu, vol = 100, 115, 0.5, 0.04, 0.3
    touch = sim.first_passage_prob_gbm(spot, k, t, mu, vol)
    terminal = sim.terminal_prob_above_gbm(spot, k, t, mu, vol)
    assert touch > terminal
    assert touch == pytest.approx(2 * terminal, rel=0.35)  # ~2x near the money


def test_first_passage_symmetry_under_zero_log_drift():
    # With mu = 0.5*sigma^2 the log-price drift is zero, so an up-barrier and a
    # down-barrier the same log-distance away are equally likely to be touched.
    vol = 0.3
    mu = 0.5 * vol * vol
    up = sim.first_passage_prob_gbm(100, 100 * math.exp(0.2), 0.5, mu, vol)
    down = sim.first_passage_prob_gbm(100, 100 * math.exp(-0.2), 0.5, mu, vol)
    assert up == pytest.approx(down, abs=1e-6)


def test_zero_time_degenerates():
    assert sim.terminal_prob_above_gbm(120, 100, 0.0, 0.04, 0.3) == 1.0
    assert sim.terminal_prob_above_gbm(90, 100, 0.0, 0.04, 0.3) == 0.0


# --- Monte-Carlo convergence to the closed form ---------------------------- #

def _gbm_sim(**kw):
    base = dict(spot=100.0, returns=None, horizon_days=60, model="gbm",
                mu=0.05, sigma=0.30, n_paths=200_000, seed=7, force_cpu=True)
    base.update(kw)
    return sim.simulate(**base)

def test_mc_gbm_matches_closed_form_probabilities():
    s = _gbm_sim()
    for k in (95, 105, 120):
        assert s.prob_touch(k) == pytest.approx(
            sim.first_passage_prob_gbm(s.spot, k, s.t_years, s.mu, s.sigma), abs=0.015)
        assert s.prob_terminal_above(k) == pytest.approx(
            sim.terminal_prob_above_gbm(s.spot, k, s.t_years, s.mu, s.sigma), abs=0.015)


def test_mc_gbm_expected_terminal():
    s = _gbm_sim()
    # E[S_T] = spot * exp(mu * T) for GBM.
    assert s.expected_terminal() == pytest.approx(
        s.spot * math.exp(s.mu * s.t_years), rel=0.01)


def test_var_cvar_conventions():
    s = _gbm_sim()
    v95, v99 = s.var(0.95), s.var(0.99)
    assert v95 > 0                      # a 60-day 30%-vol name has real downside
    assert v99 > v95                    # deeper tail -> larger loss
    assert s.cvar(0.95) >= v95          # expected shortfall is at least the VaR


def test_drawdown_in_unit_range():
    s = _gbm_sim()
    assert np.all(s.max_dd >= 0.0) and np.all(s.max_dd < 1.0)
    assert s.mean_max_drawdown() > 0.0


# --- alternative path models ----------------------------------------------- #

def test_bootstrap_constant_returns_is_deterministic():
    # Every historical return is the same constant; with recentering off, every
    # bootstrapped path is identical and terminal price is exactly determined.
    c = 0.001
    rets = np.full(300, c)
    s = sim.simulate(spot=100.0, returns=rets, horizon_days=30, model="bootstrap",
                     mu=0.0, recenter=False, block=5, n_paths=5_000, seed=3,
                     force_cpu=True)
    expected = 100.0 * math.exp(c * s.n_steps)
    assert s.terminal.std() == pytest.approx(0.0, abs=1e-6)
    assert float(s.terminal.mean()) == pytest.approx(expected, rel=1e-5)


def test_student_t_model_runs_and_fattens_tails():
    rng = np.random.default_rng(0)
    rets = rng.standard_normal(400) * 0.02
    s = sim.simulate(spot=100.0, returns=rets, horizon_days=30, model="t",
                     mu=0.04, n_paths=50_000, seed=1, force_cpu=True)
    assert s.model == "t"
    assert s.terminal.min() < s.spot < s.terminal.max()


def test_garch_falls_back_when_no_returns():
    # No history -> GARCH can't be fit -> model downgrades to Student-t.
    s = sim.simulate(spot=100.0, returns=None, horizon_days=30, model="garch",
                     mu=0.04, sigma=0.25, n_paths=10_000, seed=1, force_cpu=True)
    assert s.model == "t"


# --- GARCH fit -------------------------------------------------------------- #

def test_garch_recovers_synthetic_parameters():
    # Generate a GARCH(1,1)-normal series with known dynamics and check the fit
    # recovers the persistence and unconditional variance (loosely — MLE is noisy).
    omega, alpha, beta = 2e-6, 0.08, 0.90
    rng = np.random.default_rng(42)
    n = 4000
    r = np.empty(n)
    var = omega / (1 - alpha - beta)
    eps = 0.0
    for i in range(n):
        var = omega + alpha * eps**2 + beta * var
        eps = math.sqrt(var) * rng.standard_normal()
        r[i] = eps
    fit = sim.fit_garch(r, dist="normal")
    assert fit is not None
    assert fit.persistence == pytest.approx(alpha + beta, abs=0.05)
    uncond = omega / (1 - alpha - beta)
    assert fit.uncond_var == pytest.approx(uncond, rel=0.5)


def test_garch_fit_needs_enough_data():
    assert sim.fit_garch(np.array([0.01, -0.01, 0.02])) is None


# --- shared helpers (used by riskscan / screener) -------------------------- #

def test_summarize_name_row():
    from stock_research import expected_return as er

    s = _gbm_sim(spot=100.0, mu=0.06, sigma=0.25, horizon_days=30)
    drift = er.DriftEstimate(0.06, "fixed", {"baseline": 0.06}, [])
    row = sim.summarize_name(s, drift, target_pct=0.05, rf=0.04)

    assert row["drift"] == pytest.approx(0.06)
    assert row["sigma"] == pytest.approx(0.25)
    # Sharpe = (drift - rf) / sigma.
    assert row["sharpe"] == pytest.approx((0.06 - 0.04) / 0.25, abs=1e-3)
    # Touch is the running-extreme probability; terminal-above is the endpoint.
    assert 0.0 < row["prob_term_up"] < row["prob_up"] <= 1.0
    assert row["var"] > 0 and row["cvar"] >= row["var"]
    assert 0.0 < row["mdd"] < 1.0
    assert row["horizon_days"] == 30 and row["model"] == "gbm"
