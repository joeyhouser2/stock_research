"""Tests for the S&P 500 eligibility gates — pure arithmetic, no network."""

import datetime as dt

import pytest

from stock_research.spx import rules


@pytest.fixture
def criteria():
    return rules.load_criteria()


def _fields(**overrides):
    """A candidate that clears every gate; override one field to break one gate."""
    base = {
        "market_cap": 40e9,
        "shares_outstanding": 400e6,
        "float_shares": 380e6,             # IWF 0.95
        "annual_dollar_volume": 60e9,      # / 38e9 float cap = 1.58 liquidity ratio
        "monthly_volumes": [5e6] * 6,
        "quarterly_earnings": [100e6, 120e6, 90e6, 150e6],
        "first_trade_date": dt.date(2010, 1, 4),
        "exchange": "NASDAQ",
        "country": "United States",
    }
    base.update(overrides)
    return base


# --- threshold table -------------------------------------------------------- #

def test_threshold_is_the_row_in_force_on_the_date(criteria):
    cap, _ = criteria.min_market_cap(dt.date(2024, 6, 1))
    assert cap == pytest.approx(15.8e9)
    cap, _ = criteria.min_market_cap(dt.date(2025, 8, 1))
    assert cap == pytest.approx(22.7e9)


def test_threshold_before_table_start_warns(criteria):
    cap, warnings = criteria.min_market_cap(dt.date(2015, 1, 1))
    assert cap == pytest.approx(9.8e9)                 # earliest known row
    assert any("precedes the threshold table" in w for w in warnings)


def test_unverified_threshold_row_warns(criteria):
    _, warnings = criteria.min_market_cap(dt.date(2023, 6, 1))
    assert any("unverified" in w for w in warnings)


def test_stale_table_warns(criteria):
    """A date far past the newest row means we probably missed a quarterly revision."""
    newest = criteria.cap_schedule[-1]["effective"]
    _, warnings = criteria.min_market_cap(newest + dt.timedelta(days=400))
    assert any("may be superseded" in w for w in warnings)


# --- gates ------------------------------------------------------------------ #

def test_clean_candidate_is_eligible(criteria):
    ev = rules.evaluate("AAA", _fields(), dt.date(2025, 8, 1), criteria)
    assert ev.eligible is True
    assert ev.hard_blocks == []
    assert ev.unknown == []
    assert ev.readiness == 1.0


def test_market_cap_below_threshold_sets_readiness_to_the_shortfall(criteria):
    # Half the 22.7B threshold -> the binding gate sits at 0.5.
    ev = rules.evaluate("AAA", _fields(market_cap=11.35e9, annual_dollar_volume=30e9),
                        dt.date(2025, 8, 1), criteria)
    assert ev.eligible is False
    assert ev.binding.name == "market_cap"
    assert ev.readiness == pytest.approx(0.5)


def test_loss_in_latest_quarter_is_a_hard_block(criteria):
    """Positive four-quarter sum doesn't rescue a negative most-recent quarter."""
    ev = rules.evaluate("AAA", _fields(quarterly_earnings=[300e6, 200e6, 100e6, -50e6]),
                        dt.date(2025, 8, 1), criteria)
    assert ev.gate("viability").passed is False
    assert ev.hard_blocks == ["viability"]
    assert ev.readiness == 0.0


def test_negative_four_quarter_sum_is_a_hard_block(criteria):
    ev = rules.evaluate("AAA", _fields(quarterly_earnings=[-500e6, 50e6, 50e6, 50e6]),
                        dt.date(2025, 8, 1), criteria)
    assert ev.gate("viability").passed is False


def test_profitable_latest_quarter_and_sum_passes(criteria):
    ev = rules.evaluate("AAA", _fields(quarterly_earnings=[-50e6, 60e6, 60e6, 60e6]),
                        dt.date(2025, 8, 1), criteria)
    assert ev.gate("viability").passed is True


def test_missing_quarters_is_unknown_not_a_failure(criteria):
    ev = rules.evaluate("AAA", _fields(quarterly_earnings=[100e6, 100e6]),
                        dt.date(2025, 8, 1), criteria)
    assert ev.gate("viability").passed is None
    assert "viability" in ev.unknown
    assert ev.eligible is None                 # unresolved, not rejected
    assert ev.readiness_is_upper_bound


def test_monthly_volume_gate_scores_the_weakest_month(criteria):
    vols = [5e6, 5e6, 5e6, 125_000, 5e6, 5e6]      # one month at half the floor
    ev = rules.evaluate("AAA", _fields(monthly_volumes=vols), dt.date(2025, 8, 1), criteria)
    gate = ev.gate("monthly_volume")
    assert gate.passed is False
    assert gate.ratio == pytest.approx(0.5)


def test_monthly_volume_needs_six_months(criteria):
    ev = rules.evaluate("AAA", _fields(monthly_volumes=[5e6] * 4),
                        dt.date(2025, 8, 1), criteria)
    assert ev.gate("monthly_volume").passed is None


def test_seasoning_counts_months_listed(criteria):
    ev = rules.evaluate("AAA", _fields(first_trade_date=dt.date(2025, 3, 1)),
                        dt.date(2025, 8, 1), criteria)
    assert ev.gate("seasoning").passed is False
    ev = rules.evaluate("AAA", _fields(first_trade_date=dt.date(2024, 7, 1)),
                        dt.date(2025, 8, 1), criteria)
    assert ev.gate("seasoning").passed is True


def test_foreign_domicile_and_bad_exchange_block(criteria):
    ev = rules.evaluate("AAA", _fields(country="Ireland"), dt.date(2025, 8, 1), criteria)
    assert "domicile" in ev.hard_blocks
    ev = rules.evaluate("AAA", _fields(exchange="OTC"), dt.date(2025, 8, 1), criteria)
    assert "exchange" in ev.hard_blocks


def test_low_float_fails_both_float_gates(criteria):
    """A 5% float breaks the IWF floor and drags the float-adjusted cap under."""
    ev = rules.evaluate("AAA", _fields(float_shares=20e6), dt.date(2025, 8, 1), criteria)
    assert ev.gate("iwf").passed is False
    assert ev.gate("float_market_cap").passed is False


def test_iwf_derived_from_shares_when_not_given(criteria):
    ev = rules.evaluate("AAA", _fields(), dt.date(2025, 8, 1), criteria)
    assert ev.gate("iwf").value == pytest.approx(0.95)


def test_explicit_iwf_wins_over_share_counts(criteria):
    ev = rules.evaluate("AAA", _fields(iwf=0.4), dt.date(2025, 8, 1), criteria)
    assert ev.gate("iwf").value == pytest.approx(0.4)


def test_liquidity_ratio_is_dollar_volume_over_float_cap(criteria):
    ev = rules.evaluate("AAA", _fields(annual_dollar_volume=19e9),
                        dt.date(2025, 8, 1), criteria)
    # float cap = 40e9 * 0.95 = 38e9 -> ratio 0.5, under the 0.75 floor.
    assert ev.gate("liquidity_ratio").value == pytest.approx(0.5)
    assert ev.gate("liquidity_ratio").passed is False


def test_readiness_clips_per_gate_so_a_huge_cap_hides_nothing(criteria):
    """A trillion-dollar cap must not paper over a barely-passing liquidity gate."""
    ev = rules.evaluate("AAA", _fields(market_cap=1e12, annual_dollar_volume=713e9),
                        dt.date(2025, 8, 1), criteria)
    assert ev.readiness <= 1.0
    assert ev.binding.name == "liquidity_ratio"


# --- deletion pressure ------------------------------------------------------ #

def test_deletion_risk_zero_when_still_add_eligible(criteria):
    _, risk = rules.deletion_pressure("AAA", _fields(), dt.date(2025, 8, 1), criteria)
    assert risk == 0.0


def test_deletion_risk_maxes_at_the_review_fraction(criteria):
    """At half the addition threshold the heuristic is fully triggered."""
    _, risk = rules.deletion_pressure("AAA", _fields(market_cap=22.7e9 * 0.5),
                                      dt.date(2025, 8, 1), criteria)
    assert risk == pytest.approx(1.0)


def test_deletion_risk_scales_between(criteria):
    _, risk = rules.deletion_pressure("AAA", _fields(market_cap=22.7e9 * 0.75),
                                      dt.date(2025, 8, 1), criteria)
    assert risk == pytest.approx(0.5)


def test_viability_break_forces_elevated_deletion_risk(criteria):
    _, risk = rules.deletion_pressure(
        "AAA", _fields(market_cap=100e9, quarterly_earnings=[10e6, 10e6, 10e6, -900e6]),
        dt.date(2025, 8, 1), criteria)
    assert risk >= 0.75


# --- row rendering ---------------------------------------------------------- #

def test_to_row_is_flat_and_carries_the_summary(criteria):
    ev = rules.evaluate("AAA", _fields(), dt.date(2025, 8, 1), criteria)
    row = ev.to_row()
    assert row["ticker"] == "AAA"
    assert row["as_of"] == "2025-08-01"
    assert row["eligible"] is True
    assert row["ratio_market_cap"] == pytest.approx(40e9 / 22.7e9, rel=1e-3)
    assert all(not isinstance(v, (list, dict)) for v in row.values())
