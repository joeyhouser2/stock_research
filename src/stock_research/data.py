"""Yahoo Finance access layer (via yfinance).

Everything that talks to the network lives here so the math modules stay pure and
testable. Yahoo data is delayed and sometimes incomplete, so each accessor is
defensive and returns ``None`` / empty rather than raising on missing fields.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

# Retry policy for transient Yahoo failures (rate limits, flaky connections).
RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY = 1.5      # seconds; doubles each attempt
RATE_LIMIT_FACTOR = 3       # extra backoff multiplier when it's clearly a rate limit


def _is_rate_limit(exc: Exception) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    compact = blob.replace(" ", "")
    return "ratelimit" in compact or "toomanyrequests" in compact or "429" in blob


def _retry(fn, *, attempts: int = RETRY_ATTEMPTS, base_delay: float = RETRY_BASE_DELAY):
    """Call ``fn`` with exponential backoff, longer when it's a rate limit.

    Re-raises the last exception if every attempt fails; callers keep their own
    try/except to fall back to None/empty after that.
    """
    delay = base_delay
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - transient; retried or re-raised
            last = exc
            if attempt == attempts - 1:
                break
            wait = delay * (RATE_LIMIT_FACTOR if _is_rate_limit(exc) else 1)
            time.sleep(wait)
            delay *= 2
    assert last is not None
    raise last


@dataclass
class TickerSnapshot:
    """A point-in-time view of a ticker used by the screener."""

    ticker: str
    price: float
    quote_type: str          # "ETF", "EQUITY", ...
    size_usd: float          # market cap for equities, total assets (AUM) for ETFs
    dividend_yield: float    # annual, as a fraction (0.015 = 1.5%); 0 if unknown
    info: dict               # raw yfinance .info, reused for fundamentals (no refetch)


def get_snapshot(ticker: str) -> TickerSnapshot | None:
    """Fetch price, size (market cap or AUM), quote type and dividend yield."""
    tk = yf.Ticker(ticker)

    price = _last_price(tk)
    if price is None or price <= 0:
        return None

    info: dict = {}
    try:
        info = _retry(tk.get_info) or {}
    except Exception:
        info = {}

    quote_type = (info.get("quoteType") or "").upper()

    # Equities report marketCap; ETFs report totalAssets (AUM).
    size = info.get("marketCap") or info.get("totalAssets")
    if size is None:
        # Last-ditch: shares * price for equities.
        shares = info.get("sharesOutstanding")
        size = shares * price if shares else None
    if size is None:
        return None

    # `trailingAnnualDividendYield` is a clean fraction; the `dividendYield` field
    # is unreliable (Yahoo has returned bare percents like 73, and even bogus 1.0
    # for MSFT), so only fall back to it when the trailing field is absent.
    div = info.get("trailingAnnualDividendYield")
    if not div:
        div = info.get("dividendYield") or 0.0
        if div > 1:                       # reported as a percent (e.g. 1.5 = 1.5%)
            div /= 100.0
    div = float(div)
    if not (0.0 <= div <= 0.25):          # implausible for a $1B+ name -> ignore
        div = 0.0

    return TickerSnapshot(
        ticker=ticker,
        price=float(price),
        quote_type=quote_type or "EQUITY",
        size_usd=float(size),
        dividend_yield=div,
        info=info,
    )


def get_size(ticker: str) -> tuple[float | None, str]:
    """Just (size_usd, quote_type) for the market-cap gate — cheap universe pass.

    Returns ``(None, "")`` when Yahoo has no size, so the caller drops the name.
    """
    try:
        info = _retry(yf.Ticker(ticker).get_info) or {}
    except Exception:
        return None, ""
    quote_type = (info.get("quoteType") or "").upper()
    size = info.get("marketCap") or info.get("totalAssets")
    if size is None:
        shares = info.get("sharesOutstanding")
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        size = shares * price if shares and price else None
    return (float(size) if size else None), quote_type


def _last_price(tk: "yf.Ticker") -> float | None:
    """Most-reliable current price across yfinance's shifting APIs."""
    try:
        fi = tk.fast_info
        for key in ("last_price", "lastPrice", "regular_market_price"):
            val = getattr(fi, key, None) if not isinstance(fi, dict) else fi.get(key)
            if val:
                return float(val)
    except Exception:
        pass
    try:
        hist = tk.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None


def historical_volatility(ticker: str, window: int = 30) -> float | None:
    """Annualized realized volatility from trailing daily log returns."""
    try:
        hist = _retry(lambda: yf.Ticker(ticker).history(period=f"{max(window * 2, 60)}d"))
    except Exception:
        return None
    closes = hist["Close"].dropna() if "Close" in hist else pd.Series(dtype=float)
    if len(closes) < window + 1:
        return None
    log_ret = np.log(closes / closes.shift(1)).dropna()
    daily = log_ret.tail(window).std()
    if not math.isfinite(daily) or daily <= 0:
        return None
    return float(daily * math.sqrt(252))


def close_history(ticker: str, days: int = 1825) -> pd.Series:
    """Date-indexed daily closes for backtesting. Empty Series on failure."""
    try:
        hist = _retry(lambda: yf.Ticker(ticker).history(period=f"{max(days, 60)}d"))
    except Exception:
        return pd.Series(dtype=float)
    if "Close" not in hist:
        return pd.Series(dtype=float)
    return hist["Close"].dropna()


def daily_log_returns(ticker: str, lookback_days: int = 504) -> np.ndarray:
    """Trailing daily log returns, for fitting vol / GARCH / bootstrap models.

    Returns an empty array (not None) on failure so the math layer can decide on
    a fallback. ``lookback_days`` is in trading days of history to request.
    """
    try:
        hist = _retry(lambda: yf.Ticker(ticker).history(period=f"{max(lookback_days, 60)}d"))
    except Exception:
        return np.array([])
    closes = hist["Close"].dropna() if "Close" in hist else pd.Series(dtype=float)
    if len(closes) < 2:
        return np.array([])
    log_ret = np.log(closes / closes.shift(1)).dropna()
    return log_ret.to_numpy()


def list_expirations(ticker: str) -> list[str]:
    """Available option expiry dates ('YYYY-MM-DD'); empty if none/unsupported."""
    try:
        return list(_retry(lambda: yf.Ticker(ticker).options) or [])
    except Exception:
        return []


def get_option_chain(ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(calls, puts) for one expiry, in a single fetch. Empty DataFrames on failure."""
    try:
        chain = _retry(lambda: yf.Ticker(ticker).option_chain(expiry))
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    calls = getattr(chain, "calls", None)
    puts = getattr(chain, "puts", None)
    return (calls.copy() if calls is not None else pd.DataFrame(),
            puts.copy() if puts is not None else pd.DataFrame())


def get_call_chain(ticker: str, expiry: str) -> pd.DataFrame:
    """Call side of the chain for one expiry. Empty DataFrame on failure."""
    calls, _ = get_option_chain(ticker, expiry)
    return calls


def days_to_expiry(expiry: str, today: dt.date | None = None) -> int:
    """Calendar days from today to an 'YYYY-MM-DD' expiry."""
    today = today or dt.date.today()
    exp = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
    return (exp - today).days


def _third_friday(year: int, month: int) -> dt.date:
    """The 3rd Friday of a month — the standard monthly option expiry date."""
    first = dt.date(year, month, 1)
    first_friday = 1 + (4 - first.weekday()) % 7   # weekday: Mon=0 .. Fri=4
    return dt.date(year, month, first_friday + 14)


def classify_expiry(expiry: str) -> str:
    """'monthly' if the expiry is its month's 3rd Friday, else 'weekly'."""
    d = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
    return "monthly" if d == _third_friday(d.year, d.month) else "weekly"
