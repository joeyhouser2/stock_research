"""Foreign listings: market cap is converted to USD before the size gate — no network."""

import pytest

from stock_research import data


class _FakeTicker:
    PRICES = {"7203.T": 3000.0, "SHEL.L": 3900.0, "JPYUSD=X": 0.0066, "GBPUSD=X": 1.30}
    INFOS = {
        "7203.T": {"quoteType": "EQUITY", "currency": "JPY", "marketCap": 36e12},
        "SHEL.L": {"quoteType": "EQUITY", "currency": "GBp", "marketCap": 226e9},
    }

    def __init__(self, sym):
        self.sym = sym
        self.fast_info = {"last_price": self.PRICES.get(sym)}

    def get_info(self):
        return self.INFOS.get(self.sym, {})


@pytest.fixture(autouse=True)
def _fake_yf(monkeypatch):
    monkeypatch.setattr(data.yf, "Ticker", _FakeTicker)
    data.fx_to_usd.cache_clear()
    yield
    data.fx_to_usd.cache_clear()


def test_yen_market_cap_converted_to_usd():
    snap = data.get_snapshot("7203.T")
    assert snap.size_usd == pytest.approx(36e12 * 0.0066)


def test_pence_quote_market_cap_is_in_pounds():
    # marketCap for a GBp quote is already in pounds: convert at GBP, not GBP/100.
    snap = data.get_snapshot("SHEL.L")
    assert snap.size_usd == pytest.approx(226e9 * 1.30)


def test_usd_needs_no_rate():
    assert data.fx_to_usd("USD") == 1.0
