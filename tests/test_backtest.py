"""Tests for the walk-forward validation harness — no network access.

The key tests drive synthetic price series with a KNOWN data-generating process:
a model that matches the data should come out well-calibrated, and a deliberately
misspecified model should be caught (VaR over-exceedance). That validates the
harness itself, not just its plumbing.
"""

import math

import numpy as np
import pandas as pd
import pytest

from stock_research import backtest as bt


def _gbm_closes(n=3000, mu=0.08, sigma=0.25, s0=100.0, seed=0):
    """A synthetic daily close series from geometric Brownian motion."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    incr = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rng.standard_normal(n)
    prices = s0 * np.exp(np.cumsum(incr))
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(prices, index=idx)


# --- scoring units ---------------------------------------------------------- #

def test_reliability_perfect_calibration():
    # Predictions equal to outcomes -> Brier and ECE ~ 0.
    recs = [{"p": 1.0, "y": True} for _ in range(50)] + \
           [{"p": 0.0, "y": False} for _ in range(50)]
    rel = bt.reliability(recs, "p", "y", n_bins=10)
    assert rel.brier == pytest.approx(0.0)
    assert rel.ece == pytest.approx(0.0)
    assert rel.n == 100


def test_reliability_detects_miscalibration():
    # Always predict 0.9 but it only happens half the time.
    rng = np.random.default_rng(1)
    recs = [{"p": 0.9, "y": bool(rng.random() < 0.5)} for _ in range(400)]
    rel = bt.reliability(recs, "p", "y")
    assert rel.ece > 0.3            # ~0.4 gap between predicted and observed


def test_var_coverage_kupiec():
    # Construct returns so exactly 5% breach a constant VaR of 0.10.
    n = 200
    rets = [0.02] * n
    for k in range(10):            # 10/200 = 5% exceptions
        rets[k] = -0.20
    recs = [{"realized_return": r, "var": 0.10} for r in rets]
    cov = bt.var_coverage(recs, level=0.95)
    assert cov["exceptions"] == 10
    assert cov["observed_rate"] == pytest.approx(0.05)
    assert cov["kupiec_p"] > 0.9   # observed == expected -> not rejected


def test_var_coverage_flags_underestimated_risk():
    # 20% breaches against an expected 5% -> Kupiec should reject.
    n = 200
    rets = [0.02] * n
    for k in range(40):
        rets[k] = -0.20
    recs = [{"realized_return": r, "var": 0.10} for r in rets]
    cov = bt.var_coverage(recs, level=0.95)
    assert cov["observed_rate"] == pytest.approx(0.20)
    assert cov["kupiec_p"] < 0.01


def test_pinball_loss_minimized_at_truth():
    # Predicting the exact realized value gives zero loss; biased predictions cost.
    recs = [{"realized_return": 0.05, "q_pred": {0.5: 0.05}}]
    good = bt.pinball_loss(recs, quantile_levels=(0.5,))
    recs2 = [{"realized_return": 0.05, "q_pred": {0.5: 0.00}}]
    worse = bt.pinball_loss(recs2, quantile_levels=(0.5,))
    assert good["overall"] == pytest.approx(0.0)
    assert worse["overall"] > good["overall"]


# --- end-to-end on synthetic data ------------------------------------------ #

def test_wellspecified_model_is_calibrated():
    closes = _gbm_closes(n=3500, mu=0.08, sigma=0.25, seed=7)
    recs = bt.backtest_series(
        closes, horizon_days=30, target_pct=0.05, model="gbm",
        lookback=252, n_paths=20_000, mu=0.08, sigma=0.25, seed=11, force_cpu=True)
    assert len(recs) > 25
    scores = bt.score(recs, var_level=0.95)

    # A correctly-specified GBM model should be well-calibrated and pass Kupiec.
    assert scores["touch_up"].ece < 0.12
    assert scores["term_up"].ece < 0.12
    v = scores["var"]
    assert abs(v["observed_rate"] - v["expected_rate"]) < 0.06
    assert v["kupiec_p"] > 0.05


def test_underestimated_vol_overexceeds_var():
    # Data has 35% vol; the model assumes 18%. VaR should breach far too often.
    closes = _gbm_closes(n=3500, mu=0.05, sigma=0.35, seed=3)
    recs = bt.backtest_series(
        closes, horizon_days=30, model="gbm",
        lookback=252, n_paths=20_000, mu=0.05, sigma=0.18, seed=5, force_cpu=True)
    v = bt.var_coverage(recs, level=0.95)
    assert v["observed_rate"] > v["expected_rate"]      # risk understated
    assert v["kupiec_p"] < 0.05                         # detectably so


def test_empty_when_insufficient_history():
    closes = _gbm_closes(n=200, seed=0)
    recs = bt.backtest_series(closes, horizon_days=30, lookback=252, force_cpu=True)
    assert recs == []
