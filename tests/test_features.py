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


# --- value filters ---------------------------------------------------------

def test_pe_filter_passes_and_fails():
    from stock_research.config import Settings
    from stock_research.screener import _passes_value_filters, _value_active

    cheap = {"trailing_pe": 18.0, "forward_pe": 15.0, "peg": 1.2}
    rich = {"trailing_pe": 45.0, "forward_pe": 40.0, "peg": 3.5}
    missing = {"trailing_pe": None, "forward_pe": None, "peg": None}

    capped = Settings(max_pe=25.0)
    assert _value_active(capped, with_value=False) is True
    assert _passes_value_filters(cheap, capped) is True
    assert _passes_value_filters(rich, capped) is False
    # No P/E available (e.g. ETF) fails a P/E cap rather than sneaking through.
    assert _passes_value_filters(missing, capped) is False

    # With no caps set, everything passes and value isn't forced on.
    none = Settings()
    assert _value_active(none, with_value=False) is False
    assert _passes_value_filters(rich, none) is True


# --- extended fundamentals / currency --------------------------------------

def test_major_currency_minor_units():
    assert fundamentals.major_currency("GBp") == ("GBP", 100)
    assert fundamentals.major_currency("GBP") == ("GBP", 1)
    assert fundamentals.major_currency("jpy") == ("JPY", 1)
    assert fundamentals.major_currency(None) == ("USD", 1)


def test_extended_same_currency():
    info = {"currency": "JPY", "financialCurrency": "JPY", "trailingEps": 350.0,
            "bookValue": 3150.0, "freeCashflow": 2e12, "marketCap": 40e12,
            "trailingAnnualDividendYield": 0.03, "country": "Japan", "longName": "Toyota"}
    v = fundamentals.compute_extended(info, price=3000.0)
    assert not v["currency_mismatch"]
    assert v["fcf_yield"] == pytest.approx(0.05)
    graham = (22.5 * 350 * 3150) ** 0.5
    assert v["price_to_graham"] == pytest.approx(3000 / graham, abs=1e-3)
    assert v["dividend_yield"] == 0.03 and v["name"] == "Toyota"


def test_extended_pence_quote_uses_pounds_for_per_share_ratios():
    info = {"currency": "GBp", "financialCurrency": "GBP", "trailingEps": 1.0, "bookValue": 10.0}
    v = fundamentals.compute_extended(info, price=1500.0)        # 1500p = £15
    assert v["currency"] == "GBP" and not v["currency_mismatch"]
    assert v["price_to_graham"] == pytest.approx(15 / (22.5 * 10) ** 0.5, abs=1e-3)


def test_extended_currency_mismatch_drops_raw_ratios():
    info = {"currency": "USD", "financialCurrency": "CNY", "trailingEps": 4.0,
            "bookValue": 67.0, "freeCashflow": 1e10, "marketCap": 2.4e11}
    v = fundamentals.compute_extended(info, price=150.0)
    assert v["currency_mismatch"]
    assert v["fcf_yield"] is None and v["price_to_graham"] is None


def test_extended_etf_yield_and_nav():
    v = fundamentals.compute_extended({"yield": 0.04, "navPrice": 32.0}, price=32.32)
    assert v["dividend_yield"] == 0.04
    assert v["nav_discount"] == pytest.approx(0.01)
