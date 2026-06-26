"""Tests for the cross-asset joint factor model. Skipped without PyTorch."""

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from stock_research import joint  # noqa: E402

_HP = dict(channels=16, dilations=(1, 2, 4, 8), window=48, batch=64, steps=200)


def _factor_returns(n_days=2500, n_names=6, betas=None, factor_vol=0.011,
                    idio_vol=0.008, seed=0):
    """Synthetic returns with a strong common factor: r_it = beta_i f_t + idio."""
    rng = np.random.default_rng(seed)
    betas = np.linspace(0.7, 1.3, n_names) if betas is None else np.asarray(betas)
    f = factor_vol * rng.standard_normal(n_days)
    idio = idio_vol * rng.standard_normal((n_days, n_names))
    return f[:, None] * betas[None, :] + idio, betas


def test_fit_recovers_betas():
    rets, betas = _factor_returns(seed=1)
    tickers = [f"T{i}" for i in range(rets.shape[1])]
    model = joint.fit_factor_model(rets, tickers, force_cpu=True, tcn_hp=_HP, seed=0)
    # Equal-weight factor != the true factor exactly, but betas should track.
    assert np.corrcoef(model.betas, betas)[0, 1] > 0.9
    assert model.idio_std.shape == (len(tickers),)


def test_joint_sampler_shape_and_correlation():
    rets, _ = _factor_returns(seed=2)
    tickers = [f"T{i}" for i in range(rets.shape[1])]
    model = joint.fit_factor_model(rets, tickers, force_cpu=True, tcn_hp=_HP, seed=0)
    ctx = rets[-504:]
    term = joint.sample_joint_terminals(model, ctx, horizon_days=20, n_paths=4000,
                                        mu=0.05, seed=3)
    assert term.shape == (4000, len(tickers))
    assert np.all(np.isfinite(term))
    # Shared factor -> names are positively correlated across paths.
    corr = np.corrcoef(term.T)
    off_diag = corr[np.triu_indices(len(tickers), k=1)]
    assert off_diag.mean() > 0.3


def _synth_closes(rets, seed=0, s0=100.0):
    prices = s0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2014-01-01", periods=prices.shape[0])
    return {f"T{i}": pd.Series(prices[:, i], index=idx) for i in range(prices.shape[1])}


def test_joint_portfolio_var_better_than_independent():
    # Strongly-correlated names: independent GARCH should UNDER-cover the portfolio
    # VaR (too many breaches); the joint model should be closer to nominal.
    rets, _ = _factor_returns(n_days=2200, n_names=6, factor_vol=0.015,
                              idio_vol=0.004, seed=5)
    closes = _synth_closes(rets, seed=5)
    res = joint.compare_joint(closes, train_frac=0.6, horizon_days=20, lookback=200,
                              n_paths=4000, mu=0.05, seed=0, force_cpu=True,
                              tcn_hp=dict(_HP, steps=150))
    indep = res["portfolio"]["independent"]["var"]
    jnt = res["portfolio"]["joint"]["var"]
    exp = indep["expected_rate"]
    # Independent ignores correlation -> portfolio tail too thin -> breaches too often.
    assert indep["observed_rate"] > jnt["observed_rate"]
    # Joint sits closer to the nominal exception rate.
    assert abs(jnt["observed_rate"] - exp) < abs(indep["observed_rate"] - exp)
    assert res["windows"] > 10
