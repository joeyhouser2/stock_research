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


# --- undervaluation methods, votes, ETF peer set ---------------------------- #

def _universe():
    return _df([
        {"ticker": "CHEAP", "asset": "stock", "name": "Cheap Co", "trailing_pe": 8,
         "price_to_book": 1.0, "ev_ebitda": 4, "roe": 0.20, "fcf_yield": 0.12,
         "dividend_yield": 0.05, "payout_ratio": 0.4, "price_to_graham": 0.7,
         "pct_off_52w_high": -0.40, "analyst_upside": 0.40},
        *[{"ticker": f"MID{i}", "asset": "stock", "name": f"Mid {i}", "trailing_pe": 18 + i,
           "price_to_book": 3.0 + i, "ev_ebitda": 12 + i, "roe": 0.12, "fcf_yield": 0.04,
           "dividend_yield": 0.01, "payout_ratio": 0.3, "price_to_graham": 1.5,
           "pct_off_52w_high": -0.05, "analyst_upside": 0.05} for i in range(8)],
        {"ticker": "LOSS", "asset": "stock", "name": "Loss Co", "trailing_pe": -5,
         "price_to_book": 0.5, "roe": -0.3, "dividend_yield": 0.09, "payout_ratio": 3.0,
         "pct_off_52w_high": -0.80},
        {"ticker": "EWZ", "asset": "etf", "name": "iShares MSCI Brazil ETF",
         "trailing_pe": 9.4, "price_to_book": 0.9, "dividend_yield": 0.04},
        {"ticker": "SPY", "asset": "etf", "name": "SPDR S&P 500 ETF", "trailing_pe": 27,
         "price_to_book": 5.0, "dividend_yield": 0.012},
        {"ticker": "KORU", "asset": "etf", "name": "Direxion Daily South Korea Bull 3X",
         "trailing_pe": 5.6},
        {"ticker": "TLT", "asset": "etf", "name": "iShares 20+ Year Treasury Bond ETF",
         "price_to_book": 0.55},
    ])


def test_cheap_name_collects_votes_from_many_methods():
    out = vs.score_universe(_universe()).set_index("ticker")
    hit = set(out.loc["CHEAP", "methods_hit"].split(","))
    assert {"graham", "magic", "deep", "fcf", "dividend", "contrarian", "analyst"} <= hit
    assert out.loc["CHEAP", "votes"] == out["votes"].max()


def test_loss_maker_gets_no_cheapness_votes():
    out = vs.score_universe(_universe()).set_index("ticker")
    hit = set(str(out.loc["LOSS", "methods_hit"]).split(","))
    # Negative P/E and a >100% payout: not Graham/deep/dividend/contrarian value.
    assert not hit & {"graham", "deep", "dividend", "contrarian"}


def test_etfs_ranked_separately_and_junk_etfs_excluded():
    out = vs.score_universe(_universe()).set_index("ticker")
    assert out.loc["KORU", "excluded"] == "leveraged/inverse ETF"
    assert out.loc["TLT", "excluded"].startswith("ETF without P/E")
    assert np.isnan(out.loc["KORU", "value_score"]) and np.isnan(out.loc["TLT", "value_score"])
    # Two scorable ETFs -> ranks are among themselves only: the cheaper one is 1.0.
    assert out.loc["EWZ", "value_score"] == pytest.approx(1.0)
    assert out.loc["EWZ", "value_score"] > out.loc["SPY", "value_score"]


def test_selected_methods_only():
    out = vs.score_universe(_universe(), methods=("graham", "fcf"))
    assert "m_graham" in out and "m_fcf" in out and "m_magic" not in out
    assert out["votes"].max() <= 2


def test_graham_vote_requires_passing_the_tests():
    out = vs.score_universe(_universe(), methods=("graham",)).set_index("ticker")
    assert out.loc["CHEAP", "methods_hit"] == "graham"        # P/E 8, P/B 1.0
    assert out.loc["MID0", "methods_hit"] == ""               # P/E 18 fails


def test_country_peer_pools_small_countries():
    df = _df([{"ticker": f"US{i}", "asset": "stock", "country": "United States",
               "trailing_pe": 10 + i} for i in range(6)]
             + [{"ticker": "JP1", "asset": "stock", "country": "Japan", "trailing_pe": 30}])
    keys = vs._peer_keys(df, "country")
    assert set(keys) == {"stock:United States", "stock:Other"}
    out = vs.score_universe(df, peer="country").set_index("ticker")
    # JP1 is alone in "Other", so it's the best of its own (pooled) peer set.
    assert out.loc["JP1", "value_score"] == pytest.approx(1.0)


def test_run_rejects_unknown_method():
    with pytest.raises(ValueError):
        vs.run([], type("S", (), {"min_market_cap": 0})(), methods=("astrology",))
