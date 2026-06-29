"""Tests for the as-of feature panel and FRED parsing — no network access."""

import numpy as np
import pandas as pd

from stock_research import fred, panel


# --- FRED parsing ----------------------------------------------------------- #

def test_parse_fred_csv_handles_missing():
    text = "DATE,DGS10\n2022-01-03,1.63\n2022-01-04,.\n2022-01-05,1.71\n"
    s = fred.parse_fred_csv(text, "DGS10")
    assert list(s.values) == [1.63, 1.71]          # the "." row is dropped
    assert len(s) == 2


def test_as_of_macro_derives_spread():
    idx = pd.to_datetime(["2022-01-03", "2022-02-01", "2022-03-01"])
    macro = pd.DataFrame({"rate_10y": [1.6, 1.8, 2.0], "rate_3m": [0.1, 0.3, 0.5]}, index=idx)
    out = fred.as_of_macro(macro, "2022-02-15")
    assert out["rate_10y"] == 1.8                  # last obs on/before the date
    assert out["term_spread"] == 1.5               # 1.8 - 0.3


# --- panel features --------------------------------------------------------- #

def _fundamentals():
    """Point-in-time long table: TTM revenue 400, net income 40, equity 500, 100 shares."""
    def q(concept, tag, end, start, val, filed, unit_form="10-Q"):
        return {"concept": concept, "tag": tag, "start": start, "end": end, "val": val,
                "filed": filed, "form": unit_form, "fy": 2021, "fp": "Q1", "frame": None}
    rows = []
    for end, start, filed in [("2021-03-31", "2021-01-01", "2021-04-20"),
                              ("2021-06-30", "2021-04-01", "2021-07-20"),
                              ("2021-09-30", "2021-07-01", "2021-10-20"),
                              ("2021-12-31", "2021-10-01", "2022-02-15")]:
        rows.append(q("revenue", "Revenues", end, start, 100, filed))
        rows.append(q("net_income", "NetIncomeLoss", end, start, 10, filed))
    rows.append({"concept": "equity", "tag": "StockholdersEquity", "start": None,
                 "end": "2021-12-31", "val": 500, "filed": "2022-02-15", "form": "10-K",
                 "fy": 2021, "fp": "FY", "frame": None})
    rows.append({"concept": "shares_diluted", "tag": "WeightedAverageNumberOfDilutedSharesOutstanding",
                 "start": "2021-10-01", "end": "2021-12-31", "val": 100, "filed": "2022-02-15",
                 "form": "10-K", "fy": 2021, "fp": "FY", "frame": None})
    return pd.DataFrame(rows)


def _closes():
    idx = pd.bdate_range("2021-01-01", periods=400)
    prices = 50.0 * np.exp(np.cumsum(np.full(400, 0.0005)))   # gently rising
    return pd.Series(prices, index=idx)


def test_as_of_features_point_in_time_and_ratios():
    fund, closes = _fundamentals(), _closes()
    macro = pd.DataFrame({"rate_10y": [2.0], "rate_3m": [0.5]},
                         index=pd.to_datetime(["2021-01-01"]))
    # As of 2022-03-01: all four 2021 quarters + the 10-K are filed.
    feats = panel.as_of_features(fund, closes, macro, "2022-03-01", horizon_days=60)
    assert feats is not None
    price = feats["price"]
    mktcap = price * 100                                  # 100 shares
    # TTM net income 40 / market cap; book yield 500 / market cap.
    assert feats["earnings_yield"] == round(40.0 / mktcap, 6)
    assert feats["book_yield"] == round(500.0 / mktcap, 6)
    assert feats["roe"] == round(40.0 / 500.0, 6)
    assert feats["profit_margin"] == round(40.0 / 400.0, 6)
    assert feats["term_spread"] == 1.5


def test_as_of_features_none_before_filings():
    fund, closes = _fundamentals(), _closes()
    macro = pd.DataFrame({"rate_10y": [2.0]}, index=pd.to_datetime(["2021-01-01"]))
    # Before any filing -> no fundamentals -> no row.
    assert panel.as_of_features(fund, closes, macro, "2021-02-01") is None


def test_build_panel_shapes():
    fund, closes = _fundamentals(), _closes()
    macro = pd.DataFrame({"rate_10y": [2.0], "rate_3m": [0.5]},
                         index=pd.to_datetime(["2021-01-01"]))
    dates = pd.date_range("2022-03-01", "2022-06-01", freq="MS")
    df = panel.build_panel({"AAA": {"fundamentals": fund, "closes": closes}}, macro, dates)
    assert not df.empty
    assert set(["ticker", "as_of", "earnings_yield", "fwd_return"]).issubset(df.columns)
    assert (df["ticker"] == "AAA").all()
