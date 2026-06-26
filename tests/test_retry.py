"""Tests for the network retry/backoff helper (no real sleeps or network)."""

import pytest

from stock_research import data


def test_retry_succeeds_first_try():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "ok"

    assert data._retry(fn, base_delay=0) == "ok"
    assert calls["n"] == 1


def test_retry_recovers_after_failures(monkeypatch):
    monkeypatch.setattr(data.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("Too Many Requests: 429")
        return 42

    assert data._retry(flaky, attempts=4, base_delay=0) == 42
    assert calls["n"] == 3


def test_retry_reraises_after_exhausting(monkeypatch):
    monkeypatch.setattr(data.time, "sleep", lambda *_: None)

    def always_fail():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        data._retry(always_fail, attempts=3, base_delay=0)


def test_rate_limit_detection():
    assert data._is_rate_limit(RuntimeError("429 Too Many Requests"))
    assert data._is_rate_limit(Exception("YFRateLimitError"))
    assert not data._is_rate_limit(ValueError("bad ticker"))
