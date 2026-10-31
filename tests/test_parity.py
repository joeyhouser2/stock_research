"""Unit tests for put-call parity math — no network access required."""

import math

import pandas as pd
import pytest

from stock_research import parity


def test_theoretical_forward_matches_known_value():
    # S=100, r=0.04, q=0.01, T=1 -> 100 * e^0.03
    f = parity.theoretical_forward(spot=100, t=1.0, r=0.04, q=0.01)
    assert f == pytest.approx(100 * math.exp(0.03))


def test_implied_forward_recovers_theoretical_when_priced_off_parity():
    # If C-P were exactly priced by parity off a known forward, we should recover it.
    spot, k, t, r, q = 100.0, 105.0, 0.25, 0.04, 0.01
    f_theo = parity.theoretical_forward(spot, t, r, q)
    c_minus_p = (f_theo - k) * math.exp(-r * t)
    f_implied = parity.implied_forward(call_mid=c_minus_p + 2.0, put_mid=2.0, strike=k, t=t, r=r)
    assert f_implied == pytest.approx(f_theo, rel=1e-6)


def test_parity_row_computes_basis():
    call_row = pd.Series({"strike": 100, "bid": 5.9, "ask": 6.1, "lastPrice": 6.0})
    put_row = pd.Series({"strike": 100, "bid": 1.9, "ask": 2.1, "lastPrice": 2.0})
    stat = parity.parity_row(call_row=call_row, put_row=put_row, spot=100, t=0.25,
                             risk_free_rate=0.04, dividend_yield=0.0)
    assert stat is not None
    assert stat["call_mid"] == pytest.approx(6.0)
    assert stat["put_mid"] == pytest.approx(2.0)
    assert stat["implied_forward"] == pytest.approx(
        parity.implied_forward(6.0, 2.0, 100, 0.25, 0.04))
    assert stat["basis_pct"] == pytest.approx(
        stat["implied_forward"] / stat["theo_forward"] - 1, abs=1e-6)


def test_parity_row_none_without_quotes():
    call_row = pd.Series({"strike": 100, "bid": 0, "ask": 0, "lastPrice": 0})
    put_row = pd.Series({"strike": 100, "bid": 1.9, "ask": 2.1, "lastPrice": 2.0})
    assert parity.parity_row(call_row=call_row, put_row=put_row, spot=100, t=0.25,
                             risk_free_rate=0.04) is None


def test_parity_row_none_at_expiry():
    call_row = pd.Series({"strike": 100, "bid": 5.9, "ask": 6.1, "lastPrice": 6.0})
    put_row = pd.Series({"strike": 100, "bid": 1.9, "ask": 2.1, "lastPrice": 2.0})
    assert parity.parity_row(call_row=call_row, put_row=put_row, spot=100, t=0.0,
                             risk_free_rate=0.04) is None


def test_summarize_empty_grid():
    grid = pd.DataFrame(columns=parity.ROW_COLUMNS)
    summary = parity.summarize(grid)
    assert summary.empty
    assert list(summary.columns) == parity.SUMMARY_COLUMNS


def test_summarize_groups_by_expiry_with_median():
    grid = pd.DataFrame([
        {"expiry": "2026-08-21", "exp_type": "monthly", "dte": 30, "strike": 95,
         "call_mid": 7.0, "put_mid": 2.0, "implied_forward": 101.0, "theo_forward": 100.0,
         "basis_pct": 0.01},
        {"expiry": "2026-08-21", "exp_type": "monthly", "dte": 30, "strike": 105,
         "call_mid": 2.0, "put_mid": 7.0, "implied_forward": 103.0, "theo_forward": 100.0,
         "basis_pct": 0.03},
    ])
    summary = parity.summarize(grid)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_strikes"] == 2
    assert row["implied_forward"] == pytest.approx(102.0)   # median of 101, 103
    assert row["basis_pct"] == pytest.approx(0.02)
