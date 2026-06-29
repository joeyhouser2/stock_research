"""Unit tests for the rules-based value ranking — no network access."""

import numpy as np
import pandas as pd
import pytest

from stock_research import valuescan as vs


def _df(rows):
    return pd.DataFrame(rows)


def test_cheap_high_quality_scores_highest():
    df = _df([
        {"ticker": "CHEAP_GOOD", "trailing_pe": 8, "peg": 0.8, "price_to_book": 1.0,
         "roe": 0.25, "profit_margin": 0.20, "analyst_upside": 0.30},
        {"ticker": "MID", "trailing_pe": 18, "peg": 1.5, "price_to_book": 3.0,
         "roe": 0.15, "profit_margin": 0.10, "analyst_upside": 0.05},
        {"ticker": "RICH_WEAK", "trailing_pe": 40, "peg": 3.0, "price_to_book": 9.0,
         "roe": 0.05, "profit_margin": 0.03, "analyst_upside": -0.10},
    ])
    out = vs.composite_value_scores(df).set_index("ticker")
    assert out.loc["CHEAP_GOOD", "value_score"] > out.loc["MID", "value_score"]
    assert out.loc["MID", "value_score"] > out.loc["RICH_WEAK", "value_score"]
    assert out["value_score"].between(0, 1).all()


def test_negative_pe_not_treated_as_cheap():
    # A negative P/E must not rank as "cheapest" — it's treated as missing.
    df = _df([
        {"ticker": "LOSS", "trailing_pe": -12, "roe": 0.10, "analyst_upside": 0.0},
        {"ticker": "CHEAP", "trailing_pe": 9, "roe": 0.10, "analyst_upside": 0.0},
        {"ticker": "RICH", "trailing_pe": 35, "roe": 0.10, "analyst_upside": 0.0},
    ])
    out = vs.composite_value_scores(df).set_index("ticker")
    # CHEAP (lowest *positive* P/E) should have the best cheap rank, not LOSS.
    assert out.loc["CHEAP", "value_cheap"] > out.loc["RICH", "value_cheap"]
    assert np.isnan(out.loc["LOSS", "value_cheap"])


def test_missing_groups_are_skipped():
    # Only quality available -> value_score equals the quality sub-score.
    df = _df([
        {"ticker": "A", "roe": 0.30},
        {"ticker": "B", "roe": 0.10},
    ])
    out = vs.composite_value_scores(df).set_index("ticker")
    assert out.loc["A", "value_score"] == pytest.approx(out.loc["A", "value_quality"])
    assert out.loc["A", "value_score"] > out.loc["B", "value_score"]


def test_run_rejects_bad_sort_key():
    with pytest.raises(ValueError):
        vs.run([], type("S", (), {"min_market_cap": 0})(), sort_by="bogus")


# --- value-picker parameter filters ---------------------------------------- #

def _filter_df():
    return _df([
        {"ticker": "A", "trailing_pe": 9, "peg": 0.8, "roe": 0.20, "analyst_upside": 0.30},
        {"ticker": "B", "trailing_pe": 30, "peg": 2.5, "roe": 0.05, "analyst_upside": -0.10},
        {"ticker": "C", "trailing_pe": 15, "peg": 1.2, "roe": 0.15, "analyst_upside": 0.10},
        {"ticker": "D", "trailing_pe": None, "peg": 1.0, "roe": 0.18, "analyst_upside": 0.05},
    ])


def test_apply_value_filters_caps_and_floors():
    out = vs.apply_value_filters(_filter_df(), {"max_pe": 20, "min_roe": 0.12})
    assert set(out["ticker"]) == {"A", "C"}        # B fails P/E & ROE; D has no P/E -> fails cap


def test_apply_value_filters_missing_fails_cap():
    # A None on a filtered column is excluded (we don't pass through unknowns).
    out = vs.apply_value_filters(_filter_df(), {"max_pe": 50})
    assert "D" not in set(out["ticker"])           # D's trailing_pe is None


def test_apply_value_filters_noop_without_filters():
    df = _filter_df()
    assert len(vs.apply_value_filters(df, None)) == len(df)
    assert len(vs.apply_value_filters(df, {})) == len(df)
