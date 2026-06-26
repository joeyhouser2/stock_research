"""Unit tests for the valuation-conditioned drift model — no network access."""

import math

import pytest

from stock_research import expected_return as er

RF, ERP = 0.04, 0.045
BASELINE = RF + ERP


# --- component helpers ------------------------------------------------------ #

def test_earnings_growth_from_eps():
    g = er.earnings_growth({"forwardEps": 5.0, "trailingEps": 4.0})
    assert g == pytest.approx(0.25)


def test_earnings_growth_clamped():
    # 10x forward EPS would imply 900% growth -> clamped to the cap.
    g = er.earnings_growth({"forwardEps": 50.0, "trailingEps": 5.0})
    assert g == pytest.approx(er._GROWTH_CAP)


def test_earnings_growth_falls_back_to_reported_fields():
    assert er.earnings_growth({"earningsGrowth": 0.12}) == pytest.approx(0.12)
    assert er.earnings_growth({}) is None


# --- Grinold-Kroner --------------------------------------------------------- #

def _gk(info, div=0.0, **kw):
    return er.grinold_kroner(info, div, rf=RF, erp=ERP, **kw)

def test_cheap_drifts_higher_than_expensive():
    # Same growth (g=0.25 -> PEG=1 anchor ~ 25); one trades at 10x, one at 40x.
    eps = {"forwardEps": 5.0, "trailingEps": 4.0}
    cheap = _gk({**eps, "trailingPE": 10.0})
    rich = _gk({**eps, "trailingPE": 40.0})
    assert cheap.annual_drift > rich.annual_drift
    assert cheap.components["reversion"] > 0 > rich.components["reversion"]


def test_reversion_zero_at_anchor():
    info = {"forwardEps": 5.0, "trailingEps": 4.0, "trailingPE": 20.0}
    est = _gk(info, pe_anchor=20.0)
    assert est.components["reversion"] == pytest.approx(0.0, abs=1e-12)


def test_components_sum_to_drift():
    info = {"forwardEps": 5.0, "trailingEps": 4.0, "trailingPE": 22.0}
    est = _gk(info, div=0.015)
    c = est.components
    assert est.annual_drift == pytest.approx(c["income"] + c["growth"] + c["reversion"])
    assert c["income"] == pytest.approx(0.015)


def test_reversion_magnitude_matches_formula():
    info = {"forwardEps": 5.0, "trailingEps": 4.0, "trailingPE": 40.0}
    est = _gk(info, reversion_years=5.0)
    anchor = est.components["pe_anchor"]
    expected = -math.log(40.0 / anchor) / 5.0
    assert est.components["reversion"] == pytest.approx(expected)


def test_shrink_scales_reversion():
    info = {"forwardEps": 5.0, "trailingEps": 4.0, "trailingPE": 40.0}
    full = _gk(info, shrink=1.0)
    half = _gk(info, shrink=0.5)
    none = _gk(info, shrink=0.0)
    assert half.components["reversion"] == pytest.approx(0.5 * full.components["reversion"])
    assert none.components["reversion"] == pytest.approx(0.0)


def test_etf_without_pe_falls_back_to_baseline():
    # No EPS, no P/E (typical ETF): drift should land at the market baseline.
    est = _gk({}, div=0.015)
    assert est.annual_drift == pytest.approx(BASELINE)
    assert est.components["reversion"] == 0.0
    assert any("no trailing P/E" in n for n in est.notes)
    assert any("earnings-growth" in n for n in est.notes)


def test_absurd_pe_reversion_is_clamped():
    # A 1,000,000x P/E would imply a huge reversion drag; it's capped instead of
    # blowing through the floor. Growth here is 0, income 0, so drift == -cap.
    info = {"forwardEps": 1.0, "trailingEps": 1.0, "trailingPE": 1_000_000.0}
    est = _gk(info)
    assert est.components["reversion"] == pytest.approx(-er._REVERSION_CAP)
    assert est.annual_drift == pytest.approx(-er._REVERSION_CAP)
    assert any("reversion clamped" in n for n in est.notes)


def test_reversion_cap_bounds_low_growth_quality():
    # KO-style: trades at 25x but only grows ~9% -> PEG anchor ~9 -> big raw drag,
    # capped to -15%/yr rather than the unbounded ~-20%.
    info = {"earningsGrowth": 0.09, "trailingPE": 25.0}
    est = _gk(info, div=0.03)
    assert est.components["reversion"] == pytest.approx(-er._REVERSION_CAP)


# --- analyst-implied -------------------------------------------------------- #

def test_analyst_implied_positive_and_shrunk():
    est = er.analyst_implied({"targetMeanPrice": 120.0}, 100.0, rf=RF, erp=ERP, shrink=0.5)
    assert est is not None
    # implied +20%, shrunk halfway toward the 8.5% baseline.
    assert est.annual_drift == pytest.approx(BASELINE + 0.5 * (0.20 - BASELINE))


def test_analyst_implied_none_without_target():
    assert er.analyst_implied({}, 100.0, rf=RF, erp=ERP) is None


# --- dispatcher ------------------------------------------------------------- #

def test_estimate_fixed_is_baseline():
    est = er.estimate({}, 100.0, 0.0, model="fixed", rf=RF, erp=ERP)
    assert est.annual_drift == pytest.approx(BASELINE)
    assert est.model == "fixed"


def test_estimate_analyst_falls_back_to_fundamental():
    info = {"forwardEps": 5.0, "trailingEps": 4.0, "trailingPE": 20.0}
    est = er.estimate(info, 100.0, 0.0, model="analyst", rf=RF, erp=ERP)
    assert est.model == "fundamental"
    assert any("no analyst target" in n for n in est.notes)


def test_estimate_blend_averages():
    info = {"forwardEps": 5.0, "trailingEps": 4.0, "trailingPE": 20.0,
            "targetMeanPrice": 120.0}
    fund = er.estimate(info, 100.0, 0.0, model="fundamental", rf=RF, erp=ERP)
    ana = er.analyst_implied(info, 100.0, rf=RF, erp=ERP)
    blend = er.estimate(info, 100.0, 0.0, model="blend", rf=RF, erp=ERP)
    assert blend.model == "blend"
    assert blend.annual_drift == pytest.approx(0.5 * (fund.annual_drift + ana.annual_drift))


def test_estimate_rejects_unknown_model():
    with pytest.raises(ValueError):
        er.estimate({}, 100.0, 0.0, model="bogus")
