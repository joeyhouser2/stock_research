"""SEC EDGAR point-in-time fundamentals — the data foundation for a value model.

yfinance only exposes *current* fundamentals, so you can't train/backtest a value
model on it without look-ahead. SEC EDGAR's XBRL ``companyfacts`` API gives every
reported financial value tagged with the date it was **filed**, which is exactly
the point-in-time information you need: at any historical date you can know only
what had been filed by then.

This module fetches and caches that data and parses it into tidy per-concept
series. Building the as-of feature panel (merging with prices, computing ratios
known at each rebalance) and training the model are the next steps — this is the
ingestion + parsing layer they sit on.

SEC asks for a descriptive User-Agent with a contact and <= 10 requests/sec. Set
the ``SEC_USER_AGENT`` env var to ``"Your Name your@email"``; responses are cached
to ``data/edgar/`` so repeat runs hit the network rarely.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .config import REPO_ROOT

EDGAR_DIR = REPO_ROOT / "data" / "edgar"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"

# Default fundamentals to pull. Each key maps to a priority list of us-gaap tags
# (companies tag the same idea differently) and the XBRL unit to read.
DEFAULT_CONCEPTS = [
    {"key": "revenue", "unit": "USD", "tags": [
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"]},
    {"key": "net_income", "unit": "USD", "tags": ["NetIncomeLoss"]},
    {"key": "operating_income", "unit": "USD", "tags": ["OperatingIncomeLoss"]},
    {"key": "assets", "unit": "USD", "tags": ["Assets"]},
    {"key": "equity", "unit": "USD", "tags": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]},
    {"key": "cash_from_ops", "unit": "USD", "tags": [
        "NetCashProvidedByUsedInOperatingActivities"]},
    {"key": "eps_diluted", "unit": "USD/shares", "tags": ["EarningsPerShareDiluted"]},
    {"key": "shares_diluted", "unit": "shares", "tags": [
        "WeightedAverageNumberOfDilutedSharesOutstanding"]},
]
_FACT_COLS = ["start", "end", "val", "filed", "form", "fy", "fp", "frame"]

# Balance-sheet (instant) concepts: take the latest reported value, never summed.
# Everything else is a flow (income/cash-flow) -> trailing-twelve-month (TTM).
STOCK_CONCEPTS = {"assets", "equity", "shares_diluted"}


def _user_agent() -> str:
    return os.environ.get("SEC_USER_AGENT", "stock_research/0.1 (contact@example.com)")


def _get_json(url: str, *, attempts: int = 4, base_delay: float = 1.0) -> dict:
    """Fetch JSON from SEC with the required User-Agent and polite backoff."""
    delay = base_delay
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _user_agent(),
                                                       "Accept-Encoding": "gzip, deflate"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    import gzip
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8", "replace"))
        except Exception as exc:  # noqa: BLE001 - transient; retried then re-raised
            last = exc
            if attempt == attempts - 1:
                break
            time.sleep(delay)
            delay *= 2
    raise last  # type: ignore[misc]


def cik10(cik) -> str:
    """Zero-pad a CIK to the 10-digit form EDGAR URLs expect."""
    return str(int(cik)).zfill(10)


# --------------------------------------------------------------------------- #
# Ticker -> CIK map.
# --------------------------------------------------------------------------- #
def parse_cik_map(payload: dict) -> dict[str, int]:
    """Parse SEC's company_tickers.json into {TICKER: cik_int}."""
    out: dict[str, int] = {}
    for row in payload.values():
        ticker = str(row.get("ticker", "")).upper().strip()
        cik = row.get("cik_str")
        if ticker and cik is not None:
            out[ticker] = int(cik)
    return out


def cik_map(*, refresh: bool = False, cache_dir: Path | None = None) -> dict[str, int]:
    cache_dir = cache_dir or EDGAR_DIR
    path = cache_dir / "cik_map.json"
    if path.exists() and not refresh:
        try:
            return {k: int(v) for k, v in json.loads(path.read_text()).items()}
        except (ValueError, OSError):
            pass
    mapping = parse_cik_map(_get_json(TICKERS_URL))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping))
    return mapping


# --------------------------------------------------------------------------- #
# Company facts.
# --------------------------------------------------------------------------- #
def company_facts(ticker: str, *, refresh: bool = False,
                  cache_dir: Path | None = None) -> dict:
    """Fetch (and cache) the full XBRL companyfacts JSON for ``ticker``."""
    cache_dir = cache_dir or EDGAR_DIR
    cik = cik_map(cache_dir=cache_dir).get(ticker.upper())
    if cik is None:
        raise ValueError(f"no CIK for ticker {ticker!r} in SEC map")
    path = cache_dir / f"facts_CIK{cik10(cik)}.json"
    if path.exists() and not refresh:
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError):
            pass
    facts = _get_json(FACTS_URL.format(cik10=cik10(cik)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(facts))
    return facts


# --------------------------------------------------------------------------- #
# Parsing (pure — testable without network).
# --------------------------------------------------------------------------- #
def concept_series(facts: dict, tag: str, unit: str = "USD") -> pd.DataFrame:
    """Tidy point-in-time series for one us-gaap ``tag``: columns from _FACT_COLS.

    Each row is a reported value with its ``filed`` (availability) date, sorted by
    filing date then period end. Empty frame if the tag/unit isn't present.
    """
    try:
        rows = facts["facts"]["us-gaap"][tag]["units"][unit]
    except (KeyError, TypeError):
        return pd.DataFrame(columns=_FACT_COLS)
    df = pd.DataFrame(rows).reindex(columns=_FACT_COLS)
    return df.sort_values(["filed", "end"], na_position="last").reset_index(drop=True)


def point_in_time_fundamentals(facts: dict, concepts=DEFAULT_CONCEPTS) -> pd.DataFrame:
    """Long table of the requested concepts: [concept, tag, end, val, filed, form, fy, fp].

    For each concept the first tag (in priority order) that has data is used.
    """
    frames = []
    for spec in concepts:
        for tag in spec["tags"]:
            s = concept_series(facts, tag, spec["unit"])
            if not s.empty:
                s = s.copy()
                s.insert(0, "concept", spec["key"])
                s.insert(1, "tag", tag)
                frames.append(s)
                break
    if not frames:
        return pd.DataFrame(columns=["concept", "tag", *_FACT_COLS])
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Point-in-time as-of extraction (pure — the no-look-ahead core).
# --------------------------------------------------------------------------- #
def _ttm(g: pd.DataFrame) -> float:
    """Trailing-twelve-month value for a flow concept from its reported rows.

    Sums the 4 most recent distinct quarterly figures (period ~90 days); falls
    back to the latest annual (~365 days), then to an annualized partial.
    """
    g = g.copy()
    g["days"] = (g["end"] - g["start"]).dt.days
    q = g[(g["days"] >= 80) & (g["days"] <= 100)].drop_duplicates("end", keep="last")
    q = q.sort_values("end")
    if len(q) >= 4:
        return float(q["val"].iloc[-4:].sum())
    ann = g[(g["days"] >= 350) & (g["days"] <= 380)].sort_values(["end", "filed"])
    if not ann.empty:
        return float(ann["val"].iloc[-1])
    if len(q):
        return float(q["val"].sum() * 4.0 / len(q))     # annualize a partial year
    return float("nan")


def as_of_snapshot(long_df: pd.DataFrame, as_of) -> dict[str, float]:
    """Fundamentals known **as of** ``as_of`` — only rows already filed by then.

    Flow concepts are returned TTM; balance-sheet concepts as their latest value.
    This is the point-in-time guarantee: nothing filed after ``as_of`` is used.
    """
    as_of = pd.Timestamp(as_of)
    df = long_df.copy()
    if df.empty:
        return {}
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df = df[df["filed"] <= as_of]
    if df.empty:
        return {}
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], errors="coerce")

    out: dict[str, float] = {}
    for concept, g in df.groupby("concept"):
        if concept in STOCK_CONCEPTS:
            row = g.sort_values(["end", "filed"]).iloc[-1]
            out[concept] = float(row["val"])
        else:
            out[concept] = _ttm(g)
    return out


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def fetch(tickers: list[str], *, refresh: bool = False, save: bool = False,
          throttle: float = 0.2, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Download + cache companyfacts for each ticker; return tidy fundamentals frames."""
    out: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(tickers):
        ticker = ticker.upper()
        try:
            facts = company_facts(ticker, refresh=refresh)
        except Exception as exc:
            _log(verbose, f"  {ticker}: {exc!r} - skipped")
            continue
        df = point_in_time_fundamentals(facts)
        out[ticker] = df
        entity = facts.get("entityName", ticker)
        last_filed = df["filed"].max() if not df.empty else "n/a"
        _log(verbose, f"  {ticker}: {entity} — {df['concept'].nunique() if not df.empty else 0} "
                      f"concepts, {len(df)} facts, latest filing {last_filed}")
        if save and not df.empty:
            EDGAR_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(EDGAR_DIR / f"fundamentals_{ticker}.csv", index=False)
        if throttle and i < len(tickers) - 1:
            time.sleep(throttle)
    return out


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)
