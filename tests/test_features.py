"""Tests for expiry classification and fundamentals — both network-free."""

import pytest

from stock_research import fundamentals
from stock_research.data import classify_expiry, _third_friday


# --- expiry classification -------------------------------------------------

def test_third_friday_known_dates():
    import datetime as dt
    # Verified against a calendar.
    assert _third_friday(2026, 6) == dt.date(2026, 6, 19)
    assert _third_friday(2026, 1) == dt.date(2026, 1, 16)
    assert _third_friday(2026, 12) == dt.date(2026, 12, 18)


def test_classify_monthly_vs_weekly():
    assert classify_expiry("2026-06-19") == "monthly"   # 3rd Friday
    assert classify_expiry("2026-06-26") == "weekly"     # 4th Friday
    assert classify_expiry("2026-06-12") == "weekly"     # 2nd Friday


# --- fundamentals ----------------------------------------------------------

def test_fundamentals_full():
    info = {
        "trailingPE": 30.5, "forwardPE": 25.1, "trailingPegRatio": 1.8,
        "priceToBook": 12.3, "enterpriseToEbitda": 20.0,
        "profitMargins": 0.34, "returnOnEquity": 0.45,
        "fiftyTwoWeekHigh": 400.0, "fiftyTwoWeekLow": 250.0,
        "targetMeanPrice": 380.0,
    }
    v = fundamentals.compute(info, price=352.83)
    assert v["trailing_pe"] == 30.5
    assert v["forward_pe"] == 25.1
    assert v["peg"] == 1.8
    assert v["profit_margin"] == 0.34
    # below the 52w high -> negative
    assert v["pct_off_52w_high"] == pytest.approx((352.83 - 400) / 400, abs=1e-4)
    assert v["pct_off_52w_high"] < 0
    # analyst target above spot -> positive upside
    assert v["analyst_upside"] == pytest.approx((380 - 352.83) / 352.83, abs=1e-4)
    assert v["analyst_upside"] > 0


def test_fundamentals_missing_fields_are_none():
    v = fundamentals.compute({}, price=100.0)
    assert set(v) == set(fundamentals.VALUE_COLUMNS)
    assert all(val is None for val in v.values())


def test_fundamentals_peg_fallback_key():
    # Falls back to the older 'pegRatio' key when 'trailingPegRatio' is absent.
    v = fundamentals.compute({"pegRatio": 2.5}, price=100.0)
    assert v["peg"] == 2.5


def test_empty_matches_columns():
    assert set(fundamentals.empty()) == set(fundamentals.VALUE_COLUMNS)
