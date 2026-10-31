"""Tests for the option-chain logger's offline pieces."""

import pandas as pd

from stock_research import chainlog


def test_field_coercion():
    assert chainlog._f("1.5") == 1.5
    assert chainlog._f(None) == ""
    assert chainlog._f("x") == ""
    assert chainlog._i(3.0) == 3
    assert chainlog._i(None) == ""


def test_snapshot_skips_when_no_data(monkeypatch):
    # No price/size -> empty, no crash (defensive like the rest of the data layer).
    monkeypatch.setattr(chainlog.data, "get_snapshot", lambda t: None)
    assert chainlog.snapshot_ticker("NOPE", min_dte=1, max_dte=120) == []


def test_fields_cover_iv_and_keys():
    for col in ("date", "ticker", "expiry", "strike", "side", "iv", "contract"):
        assert col in chainlog.FIELDS


def test_snapshot_logs_both_sides(monkeypatch):
    snap = chainlog.data.TickerSnapshot(
        ticker="ABC", price=100.0, quote_type="EQUITY", size_usd=1e9,
        dividend_yield=0.0, info={})
    monkeypatch.setattr(chainlog.data, "get_snapshot", lambda t: snap)
    monkeypatch.setattr(chainlog.data, "list_expirations", lambda t: ["2099-01-15"])
    monkeypatch.setattr(chainlog.data, "days_to_expiry", lambda e, today=None: 30)

    calls = pd.DataFrame([{"strike": 110.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1,
                           "impliedVolatility": 0.3, "openInterest": 10, "volume": 5,
                           "contractSymbol": "ABC_C"}])
    puts = pd.DataFrame([{"strike": 90.0, "bid": 0.9, "ask": 1.1, "lastPrice": 1.0,
                          "impliedVolatility": 0.35, "openInterest": 8, "volume": 3,
                          "contractSymbol": "ABC_P"}])
    monkeypatch.setattr(chainlog.data, "get_option_chain", lambda t, e: (calls, puts))

    rows = chainlog.snapshot_ticker("ABC", min_dte=1, max_dte=120)
    sides = {r["side"] for r in rows}
    assert sides == {"call", "put"}
    assert len(rows) == 2
