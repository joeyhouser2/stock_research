"""Unit tests for the option math — no network access required."""

import math

import pandas as pd
import pytest

from stock_research import blackscholes as bs
from stock_research import metrics


# --- Black-Scholes ---------------------------------------------------------

def test_delta_bounds_and_monotonicity():
    # Deep ITM -> delta near 1; deep OTM -> delta near 0.
    deep_itm = bs.call_delta(spot=100, strike=50, t=0.25, vol=0.3, r=0.04)
    deep_otm = bs.call_delta(spot=100, strike=200, t=0.25, vol=0.3, r=0.04)
    assert 0.95 < deep_itm <= 1.0
    assert 0.0 <= deep_otm < 0.05


def test_prob_otm_complements_itm():
    # An ATM call: prob OTM should be a bit above 0.5 (negative d2 drift term small).
    p = bs.prob_otm_call(spot=100, strike=100, t=0.25, vol=0.3, r=0.04)
    assert 0.4 < p < 0.6


def test_prob_otm_rises_with_strike():
    near = bs.prob_otm_call(100, 102, 0.1, 0.3, 0.04)
    far = bs.prob_otm_call(100, 130, 0.1, 0.3, 0.04)
    assert far > near
    assert far > 0.9


def test_call_price_matches_known_value():
    # Textbook case: S=K=100, T=1, vol=0.2, r=0.05 -> ~10.45.
    price = bs.call_price(100, 100, 1.0, 0.2, 0.05)
    assert price == pytest.approx(10.45, abs=0.05)


def test_zero_time_is_intrinsic():
    assert bs.call_price(120, 100, 0.0, 0.3, 0.04) == pytest.approx(20.0)
    assert bs.call_delta(120, 100, 0.0, 0.3, 0.04) == 1.0
    assert bs.prob_otm_call(120, 100, 0.0, 0.3, 0.04) == 0.0


# --- metrics helpers -------------------------------------------------------

def test_mid_price_prefers_bidask_then_last():
    assert metrics.mid_price(1.0, 3.0, 5.0) == 2.0
    assert metrics.mid_price(0, 0, 4.2) == 4.2      # no quotes -> last
    assert metrics.mid_price(None, None, None) is None


def test_spread_pct():
    assert metrics.spread_pct(1.0, 1.10) == pytest.approx(0.10 / 1.05, rel=1e-6)
    assert metrics.spread_pct(0, 2.0) is None


def test_annualize_scales_by_year_fraction():
    # 1% over ~73 days annualizes to ~5%.
    assert metrics.annualize(0.01, 73) == pytest.approx(0.05, abs=0.001)
    assert metrics.annualize(0.05, 0) == 0.0


def test_compute_rejects_non_otm():
    itm = pd.Series({"strike": 90, "bid": 11, "ask": 12, "lastPrice": 11.5,
                     "impliedVolatility": 0.3, "openInterest": 100, "volume": 10})
    assert metrics.compute(row=itm, spot=100, dte=30, risk_free_rate=0.04, hv=0.25) is None


def test_compute_full_row():
    row = pd.Series({
        "contractSymbol": "TEST", "strike": 110, "bid": 2.0, "ask": 2.2,
        "lastPrice": 2.1, "impliedVolatility": 0.30, "openInterest": 500, "volume": 40,
    })
    stat = metrics.compute(row=row, spot=100, dte=30, risk_free_rate=0.04, hv=0.25,
                           dividend_yield=0.01)
    assert stat is not None
    assert stat["mid"] == pytest.approx(2.1)
    assert stat["pct_otm"] == pytest.approx(0.10)
    assert stat["downside_cushion"] == pytest.approx(0.021)
    assert stat["breakeven"] == pytest.approx(97.9)
    assert stat["iv_hv"] == pytest.approx(1.2, abs=0.01)
    # annual yield = (2.1/100) * (365/30)
    assert stat["annual_yield"] == pytest.approx(0.021 * 365 / 30, abs=1e-4)
    assert 0.0 < stat["prob_otm"] < 1.0
    assert 0.0 < stat["delta"] < 1.0
    # score is yield weighted by the probability of keeping it.
    assert stat["score"] == pytest.approx(stat["annual_yield"] * stat["prob_otm"], abs=1e-4)
    assert stat["score"] < stat["annual_yield"]


def test_score_adj_weights_by_iv_hv():
    # IV (0.30) rich vs HV (0.25) -> iv_hv 1.2 -> score_adj = score * 1.2 > score.
    row = pd.Series({
        "contractSymbol": "RICH", "strike": 110, "bid": 2.0, "ask": 2.2,
        "lastPrice": 2.1, "impliedVolatility": 0.30, "openInterest": 500, "volume": 40,
    })
    stat = metrics.compute(row=row, spot=100, dte=30, risk_free_rate=0.04, hv=0.25)
    assert stat["iv_hv"] == pytest.approx(1.2, abs=0.01)
    assert stat["score_adj"] == pytest.approx(stat["score"] * 1.2, abs=1e-4)
    assert stat["score_adj"] > stat["score"]

    # IV cheap vs HV -> score_adj penalized below score.
    cheap = metrics.compute(row=row, spot=100, dte=30, risk_free_rate=0.04, hv=0.40)
    assert cheap["score_adj"] < cheap["score"]


def test_score_adj_none_without_iv():
    row = pd.Series({
        "contractSymbol": "NOIV", "strike": 110, "bid": 2.0, "ask": 2.2,
        "lastPrice": 2.1, "impliedVolatility": 0.0, "openInterest": 500, "volume": 40,
    })
    stat = metrics.compute(row=row, spot=100, dte=30, risk_free_rate=0.04, hv=0.25)
    assert stat["score_adj"] is None


def test_score_is_none_without_iv():
    row = pd.Series({
        "contractSymbol": "NOIV", "strike": 110, "bid": 2.0, "ask": 2.2,
        "lastPrice": 2.1, "impliedVolatility": 0.0, "openInterest": 500, "volume": 40,
    })
    stat = metrics.compute(row=row, spot=100, dte=30, risk_free_rate=0.04, hv=0.25)
    assert stat is not None
    assert stat["prob_otm"] is None
    assert stat["score"] is None
    assert stat["annual_yield"] > 0   # yield still computable without IV


def test_passes_liquidity():
    good = {"open_interest": 100, "volume": 5, "spread_pct": 0.05}
    assert metrics.passes_liquidity(good, min_oi=50, min_volume=1, max_spread=0.15)

    thin = {"open_interest": 10, "volume": 5, "spread_pct": 0.05}
    assert not metrics.passes_liquidity(thin, min_oi=50, min_volume=1, max_spread=0.15)

    wide = {"open_interest": 100, "volume": 5, "spread_pct": 0.40}
    assert not metrics.passes_liquidity(wide, min_oi=50, min_volume=1, max_spread=0.15)
