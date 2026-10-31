"""Tests for the model-free risk-neutral probability extraction."""

import pandas as pd

from stock_research import riskneutral


def _chain(rows):
    return pd.DataFrame(rows)


def test_empty_or_short_chain_returns_empty():
    assert riskneutral.implied_prob_otm(pd.DataFrame(), dte=30, risk_free_rate=0.04) == {}
    one = _chain([{"strike": 100.0, "bid": 5.0, "ask": 5.2, "lastPrice": 5.1}])
    assert riskneutral.implied_prob_otm(one, dte=30, risk_free_rate=0.04) == {}


def test_zero_dte_returns_empty():
    chain = _chain([
        {"strike": 95.0, "bid": 6.0, "ask": 6.2, "lastPrice": 6.1},
        {"strike": 105.0, "bid": 2.0, "ask": 2.2, "lastPrice": 2.1},
    ])
    assert riskneutral.implied_prob_otm(chain, dte=0, risk_free_rate=0.04) == {}


def test_monotone_decreasing_calls_give_increasing_prob_otm():
    # Call prices strictly decreasing in strike -> monotone positive density proxy,
    # so P(S_T<=K) should rise as strike rises across the sampled strikes.
    chain = _chain([
        {"strike": 90.0, "bid": 11.5, "ask": 11.7, "lastPrice": 11.6},
        {"strike": 100.0, "bid": 5.0, "ask": 5.2, "lastPrice": 5.1},
        {"strike": 110.0, "bid": 1.5, "ask": 1.7, "lastPrice": 1.6},
        {"strike": 120.0, "bid": 0.3, "ask": 0.5, "lastPrice": 0.4},
    ])
    probs = riskneutral.implied_prob_otm(chain, dte=30, risk_free_rate=0.04)
    assert set(probs) == {90.0, 100.0, 110.0, 120.0}
    ordered = [probs[k] for k in sorted(probs)]
    assert ordered == sorted(ordered)
    for p in probs.values():
        assert 0.0 <= p <= 1.0


def test_duplicate_strikes_are_deduped():
    chain = _chain([
        {"strike": 100.0, "bid": 5.0, "ask": 5.2, "lastPrice": 5.1},
        {"strike": 100.0, "bid": 5.0, "ask": 5.2, "lastPrice": 5.1},
        {"strike": 110.0, "bid": 1.5, "ask": 1.7, "lastPrice": 1.6},
    ])
    probs = riskneutral.implied_prob_otm(chain, dte=30, risk_free_rate=0.04)
    assert set(probs) == {100.0, 110.0}


def test_rows_with_no_usable_price_are_skipped():
    chain = _chain([
        {"strike": 90.0, "bid": None, "ask": None, "lastPrice": None},
        {"strike": 100.0, "bid": 5.0, "ask": 5.2, "lastPrice": 5.1},
        {"strike": 110.0, "bid": 1.5, "ask": 1.7, "lastPrice": 1.6},
    ])
    probs = riskneutral.implied_prob_otm(chain, dte=30, risk_free_rate=0.04)
    assert set(probs) == {100.0, 110.0}
