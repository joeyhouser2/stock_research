"""Unit tests for the long-call P&L simulator — no network access."""

import math

import numpy as np
import pytest

from stock_research import blackscholes as bs
from stock_research import longcall as lc


def _returns(n=600, vol=0.02, seed=0):
    return np.random.default_rng(seed).standard_normal(n) * vol


def test_bs_vec_matches_scalar():
    v = lc._bs_call_vec(np.array([100.0]), 100.0, 1.0, np.array([0.2]), 0.05)[0]
    assert v == pytest.approx(bs.call_price(100, 100, 1.0, 0.2, 0.05), abs=1e-6)


def test_bs_vec_zero_time_is_intrinsic():
    out = lc._bs_call_vec(np.array([120.0, 90.0]), 100.0, 0.0, np.array([0.3, 0.3]), 0.04)
    assert out[0] == pytest.approx(20.0) and out[1] == pytest.approx(0.0)


def _sim(strike, premium, **kw):
    base = dict(spot=100.0, strike=strike, premium=premium, dte=30, current_iv=0.30,
                returns=_returns(seed=1), mu=0.05, r=0.04, n_paths=40_000, seed=2,
                force_cpu=True)
    base.update(kw)
    return lc.simulate_long_call(**base)


def test_probabilities_in_range():
    r = _sim(105, 2.0)
    assert 0.0 <= r.prob_profit() <= 1.0
    assert 0.0 <= r.prob_total_loss() <= 1.0
    assert r.prob_multiple(2) <= r.prob_profit()       # 2x is rarer than any profit
    assert np.all(r.call_value >= 0.0)


def test_total_loss_rises_with_strike():
    # Hold to expiry, fixed premium: a higher strike is more likely to expire worthless.
    lo = _sim(95, 2.0).prob_total_loss()
    mid = _sim(105, 2.0).prob_total_loss()
    hi = _sim(120, 2.0).prob_total_loss()
    assert lo < mid < hi


def test_early_exit_has_time_value():
    # Exiting before expiry leaves time value, cushioning total loss vs holding to expiry.
    expiry = _sim(100, 3.0, hold_days=30)
    early = _sim(100, 3.0, hold_days=10)
    assert early.days_remaining == 20 and expiry.days_remaining == 0
    assert early.prob_total_loss() < expiry.prob_total_loss()


def test_iv_rv_anchor_and_clamp():
    rich = _sim(100, 3.0, current_iv=0.60)     # IV well above realized
    cheap = _sim(100, 3.0, current_iv=0.12)    # IV below realized
    assert rich.iv_rv > 1.05
    assert cheap.iv_rv < 0.95
    crazy = _sim(100, 3.0, current_iv=9.0)
    assert crazy.iv_rv == pytest.approx(3.0)   # clamped


def test_rejects_bad_premium():
    with pytest.raises(ValueError):
        _sim(100, 0.0)
