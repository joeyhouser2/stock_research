"""Unit tests for the SEC EDGAR parsing layer — no network access."""

import pandas as pd

from stock_research import edgar


def test_cik10_pads():
    assert edgar.cik10(320193) == "0000320193"
    assert edgar.cik10("320193") == "0000320193"


def test_parse_cik_map():
    payload = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "msft", "title": "Microsoft"},
    }
    m = edgar.parse_cik_map(payload)
    assert m["AAPL"] == 320193
    assert m["MSFT"] == 789019            # upper-cased


_FACTS = {
    "entityName": "Test Co",
    "facts": {"us-gaap": {
        "NetIncomeLoss": {"units": {"USD": [
            {"end": "2022-12-31", "val": 100, "filed": "2023-02-01", "form": "10-K", "fy": 2022, "fp": "FY"},
            {"end": "2021-12-31", "val": 80, "filed": "2022-02-01", "form": "10-K", "fy": 2021, "fp": "FY"},
        ]}},
        "Revenues": {"units": {"USD": [
            {"end": "2022-12-31", "val": 1000, "filed": "2023-02-01", "form": "10-K", "fy": 2022, "fp": "FY"},
        ]}},
    }},
}


def test_concept_series_parses_and_sorts():
    df = edgar.concept_series(_FACTS, "NetIncomeLoss", "USD")
    assert list(df["val"]) == [80, 100]                 # sorted by filed date ascending
    assert "filed" in df.columns and len(df) == 2


def test_concept_series_missing_is_empty():
    df = edgar.concept_series(_FACTS, "DoesNotExist", "USD")
    assert df.empty and list(df.columns) == edgar._FACT_COLS


def _flow(end, start, val, filed):
    return {"concept": "revenue", "tag": "Revenues", "start": start, "end": end,
            "val": val, "filed": filed, "form": "10-Q", "fy": 2022, "fp": "Q1", "frame": None}


def test_as_of_snapshot_ttm_and_point_in_time():
    import pandas as pd
    # Four quarterly revenue figures (~90d each) plus one filed in the FUTURE.
    rows = [
        _flow("2022-03-31", "2022-01-01", 100, "2022-04-20"),
        _flow("2022-06-30", "2022-04-01", 110, "2022-07-20"),
        _flow("2022-09-30", "2022-07-01", 120, "2022-10-20"),
        _flow("2022-12-31", "2022-10-01", 130, "2023-02-15"),
        _flow("2023-03-31", "2023-01-01", 200, "2023-04-20"),   # future vs our as-of
    ]
    # A balance-sheet concept (instant: no start).
    rows.append({"concept": "equity", "tag": "StockholdersEquity", "start": None,
                 "end": "2022-12-31", "val": 5000, "filed": "2023-02-15",
                 "form": "10-K", "fy": 2022, "fp": "FY", "frame": None})
    long_df = pd.DataFrame(rows)

    # As of 2023-03-01: the 2023-Q1 row (filed 2023-04-20) must be EXCLUDED.
    snap = edgar.as_of_snapshot(long_df, "2023-03-01")
    assert snap["revenue"] == 100 + 110 + 120 + 130        # TTM of the 4 known quarters
    assert snap["equity"] == 5000

    # As of 2022-08-01: only Q1+Q2 filed -> annualized partial (2 quarters).
    early = edgar.as_of_snapshot(long_df, "2022-08-01")
    assert early["revenue"] == (100 + 110) * 4 / 2
    assert "equity" not in early                           # 10-K not filed yet


def test_as_of_snapshot_empty_before_any_filing():
    import pandas as pd
    long_df = pd.DataFrame([_flow("2022-03-31", "2022-01-01", 100, "2022-04-20")])
    assert edgar.as_of_snapshot(long_df, "2022-01-01") == {}


def test_point_in_time_picks_first_available_tag():
    concepts = [
        {"key": "revenue", "unit": "USD", "tags": ["SalesRevenueNet", "Revenues"]},  # 2nd tag wins
        {"key": "net_income", "unit": "USD", "tags": ["NetIncomeLoss"]},
    ]
    df = edgar.point_in_time_fundamentals(_FACTS, concepts)
    assert set(df["concept"]) == {"revenue", "net_income"}
    rev = df[df["concept"] == "revenue"]
    assert (rev["tag"] == "Revenues").all()             # fell through to the available tag
    assert rev.iloc[0]["val"] == 1000
