"""Yahoo Finance access layer (via yfinance).

Everything that talks to the network lives here so the math modules stay pure and
testable. Yahoo data is delayed and sometimes incomplete, so each accessor is
defensive and returns ``None`` / empty rather than raising on missing fields.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class TickerSnapshot:
    """A point-in-time view of a ticker used by the screener."""

    ticker: str
    price: float
    quote_type: str          # "ETF", "EQUITY", ...
    size_usd: float          # market cap for equities, total assets (AUM) for ETFs
    dividend_yield: float    # annual, as a fraction (0.015 = 1.5%); 0 if unknown


def get_snapshot(ticker: str) -> TickerSnapshot | None:
    """Fetch price, size (market cap or AUM), quote type and dividend yield."""
    tk = yf.Ticker(ticker)

    price = _last_price(tk)
    if price is None or price <= 0:
        return None

    info: dict = {}
    try:
        info = tk.get_info() or {}
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
    )


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
        hist = yf.Ticker(ticker).history(period=f"{max(window * 2, 60)}d")
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


def list_expirations(ticker: str) -> list[str]:
    """Available option expiry dates ('YYYY-MM-DD'); empty if none/unsupported."""
    try:
        return list(yf.Ticker(ticker).options or [])
    except Exception:
        return []


def get_call_chain(ticker: str, expiry: str) -> pd.DataFrame:
    """Call side of the chain for one expiry. Empty DataFrame on failure."""
    try:
        chain = yf.Ticker(ticker).option_chain(expiry)
    except Exception:
        return pd.DataFrame()
    calls = getattr(chain, "calls", None)
    return calls.copy() if calls is not None else pd.DataFrame()


def days_to_expiry(expiry: str, today: dt.date | None = None) -> int:
    """Calendar days from today to an 'YYYY-MM-DD' expiry."""
    today = today or dt.date.today()
    exp = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
    return (exp - today).days
