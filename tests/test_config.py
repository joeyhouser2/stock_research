"""Config + universe loading tests against the shipped YAML files."""

from stock_research.config import load_settings, load_universe, override


def test_settings_load_from_yaml():
    s = load_settings()
    assert s.min_market_cap >= 1_000_000_000
    assert s.min_dte < s.max_dte
    assert 0 < s.min_otm < s.max_otm < 1


def test_override_ignores_none():
    s = load_settings()
    out = override(s, min_dte=10, max_dte=None)
    assert out.min_dte == 10
    assert out.max_dte == s.max_dte   # unchanged


def test_universe_is_deduped_and_upper():
    u = load_universe()
    assert len(u) == len(set(u))
    assert all(t == t.upper() for t in u)
    assert "SPY" in u
