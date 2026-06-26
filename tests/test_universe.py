"""Tests for the Cboe weeklys parser and the market-cap cache — network-free."""

from stock_research import cboe
from stock_research.cache import MarketCapCache


# A trimmed sample mirroring the real Cboe CSV layout: schedule preamble first,
# then the two labelled sections.
SAMPLE_CSV = '''\
"Standard","06/26/26","",""
"Equity/ETF/ETN (MON)","06/29/26","07/06/26"
"SPXW (MON)","06/29/26","07/06/26"
"Available Weeklys - Exchange Traded Products (ETFs and ETNs)"
"SPY","SPDR S&P 500 ETF TRUST"
"QQQ","INVESCO QQQ TRUST"
"Available Weeklys - Equity"
"AAPL","APPLE INC"
"MSFT","MICROSOFT CORP"
"BRK.B","BERKSHIRE HATHAWAY INC-CL B"
'''


def test_parse_skips_preamble_and_tags_sections():
    recs = cboe.parse_weeklys(SAMPLE_CSV)
    by_symbol = {r["symbol"]: r for r in recs}
    assert set(by_symbol) == {"SPY", "QQQ", "AAPL", "MSFT", "BRK.B"}
    assert by_symbol["SPY"]["type"] == "ETF"
    assert by_symbol["AAPL"]["type"] == "EQUITY"
    assert by_symbol["AAPL"]["name"] == "APPLE INC"
    # Schedule rows (Standard, SPXW, ...) must not leak in as symbols.
    assert "SPXW" not in by_symbol
    assert "Standard" not in by_symbol


def test_parse_keeps_dotted_tickers():
    recs = cboe.parse_weeklys(SAMPLE_CSV)
    assert any(r["symbol"] == "BRK.B" for r in recs)


def test_save_and_load_roundtrip(tmp_path):
    recs = cboe.parse_weeklys(SAMPLE_CSV)
    path = tmp_path / "weeklys.csv"
    cboe.save_weeklys(recs, path)
    loaded = cboe.load_weeklys(path)
    assert [r["symbol"] for r in loaded] == [r["symbol"] for r in recs]


def test_get_symbols_dedupes(tmp_path):
    path = tmp_path / "weeklys.csv"
    cboe.save_weeklys(
        [{"symbol": "AAPL", "type": "EQUITY", "name": "a"},
         {"symbol": "AAPL", "type": "EQUITY", "name": "a"},
         {"symbol": "spy", "type": "ETF", "name": "b"}],
        path,
    )
    syms = cboe.get_symbols(path=path)
    assert syms == ["AAPL", "SPY"]


# --- cache -----------------------------------------------------------------

def test_cache_put_get_roundtrip(tmp_path):
    c = MarketCapCache(path=tmp_path / "mc.json", ttl_days=7, now=1000.0)
    c.put("AAPL", 4.0e12, "EQUITY")
    rec = c.get("aapl")           # case-insensitive
    assert rec["size_usd"] == 4.0e12
    assert rec["quote_type"] == "EQUITY"


def test_cache_expiry():
    fresh = MarketCapCache(path="/nonexistent.json", ttl_days=1, now=0.0)
    fresh.put("X", 1e9, "EQUITY")
    # Same instance advanced past TTL: simulate by reading with a later clock.
    fresh._now = 2 * 86400        # 2 days later, ttl is 1 day
    assert fresh.get("X") is None


def test_cache_persists(tmp_path):
    path = tmp_path / "mc.json"
    c1 = MarketCapCache(path=path, ttl_days=7, now=500.0)
    c1.put("MSFT", 3e12, "EQUITY")
    c1.save()
    c2 = MarketCapCache(path=path, ttl_days=7, now=500.0)
    assert c2.get("MSFT")["size_usd"] == 3e12
