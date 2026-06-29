"""Tests for the option-chain logger's offline pieces."""

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
    for col in ("date", "ticker", "expiry", "strike", "iv", "contract"):
        assert col in chainlog.FIELDS
