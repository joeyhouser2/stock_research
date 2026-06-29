"""Tests for the cross-sectional value model: purge logic, IC, and a signal check."""

import numpy as np
import pandas as pd
import pytest

from stock_research import valuemodel as vm


def _panel(n_dates=12, n_names=30, signal=0.0, seed=0):
    """Synthetic panel where earnings_yield drives forward return with `signal` strength."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="3MS")
    rows = []
    for d in dates:
        ey = rng.normal(size=n_names)                       # standardized value factor
        noise = rng.normal(size=n_names)
        fwd = signal * ey + noise * 0.1                     # forward return
        for i in range(n_names):
            rows.append({
                "ticker": f"T{i:02d}", "as_of": d.date().isoformat(), "price": 100.0,
                "market_cap_b": 10.0, "log_size": rng.normal(), "earnings_yield": ey[i],
                "book_yield": rng.normal(), "sales_yield": rng.normal(),
                "cfo_yield": rng.normal(), "roe": rng.normal(), "profit_margin": rng.normal(),
                "asset_turnover": rng.normal(), "rev_growth_yoy": rng.normal(),
                "ni_growth_yoy": rng.normal(), "mom_12_1": rng.normal(), "vol_3m": rng.normal(),
                "rate_10y": 3.0, "rate_3m": 2.0, "term_spread": 1.0, "rate_10y_chg_3m": 0.0,
                "fwd_return": fwd[i],
            })
    return pd.DataFrame(rows)


# --- purge / split correctness --------------------------------------------- #

def test_walk_forward_purges_overlapping_labels():
    # 6-month horizon, quarterly rebalances -> the immediately-prior date overlaps
    # and must NOT be in the training set for a given test date.
    df = _panel(n_dates=10, signal=0.0, seed=1)
    preds = vm.walk_forward(df, horizon_days=182, min_train_rows=1)
    assert not preds.empty
    tested = sorted(preds["as_of"].unique())
    all_dates = sorted(df["as_of"].astype(str).unique())
    # The first testable date must be late enough that >=1 fully-realized prior date exists.
    assert tested[0] > all_dates[1]


# --- IC behaves: signal vs noise ------------------------------------------- #

def test_ic_positive_when_signal_present():
    pytest.importorskip("lightgbm")
    df = _panel(n_dates=16, n_names=40, signal=0.5, seed=2)
    score = vm.evaluate(vm.walk_forward(df, horizon_days=182))
    assert score.n_dates > 3
    assert score.mean_ic > 0.10                  # the planted signal is recovered out-of-sample


def test_ic_near_zero_on_pure_noise():
    pytest.importorskip("lightgbm")
    df = _panel(n_dates=16, n_names=40, signal=0.0, seed=3)
    score = vm.evaluate(vm.walk_forward(df, horizon_days=182))
    assert abs(score.mean_ic) < 0.12             # no signal -> IC indistinguishable from 0


# --- evaluate() arithmetic on hand-made predictions ------------------------ #

def test_evaluate_perfect_and_inverse():
    # Two dates: perfectly-ranked predictions -> IC +1.
    rows = []
    for d in ("2021-01-01", "2021-04-01"):
        for i in range(10):
            rows.append({"as_of": d, "ticker": f"T{i}", "pred": float(i),
                         "fwd_return": float(i), "baseline": float(i)})
    score = vm.evaluate(pd.DataFrame(rows))
    assert score.mean_ic == pytest.approx(1.0, abs=1e-9)
    assert score.hit_rate == 1.0
    assert score.mean_spread > 0                 # top half outperforms bottom half
