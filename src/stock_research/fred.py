"""FRED macro series — interest-rate context for the feature panel.

Uses FRED's public ``fredgraph.csv`` download endpoint, which needs no API key.
Series are cached to ``data/fred/`` and looked up **as-of** a date (last value on
or before it) so the panel stays point-in-time.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

import pandas as pd

from .config import REPO_ROOT

FRED_DIR = REPO_ROOT / "data" / "fred"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# Daily series we use: 10-year and 3-month Treasury yields (percent).
DEFAULT_SERIES = {"rate_10y": "DGS10", "rate_3m": "DGS3MO"}


def parse_fred_csv(text: str, name: str) -> pd.Series:
    """Parse a fredgraph CSV (DATE,<ID>) into a date-indexed float Series.

    FRED marks missing observations with ``.`` — those become NaN and are dropped.
    """
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return pd.Series(dtype=float, name=name)
    out = {}
    for r in rows[1:]:
        if len(r) < 2 or not r[0].strip():
            continue
        try:
            out[pd.Timestamp(r[0])] = float(r[1])
        except (ValueError, TypeError):
            continue                       # "." or unparseable -> skip
    return pd.Series(out, name=name).sort_index()


def fetch_series(series_id: str, *, timeout: int = 30) -> pd.Series:
    req = urllib.request.Request(FRED_CSV.format(series=series_id),
                                 headers={"User-Agent": "stock_research/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return parse_fred_csv(resp.read().decode("utf-8", "replace"), series_id)


def macro_frame(series: dict[str, str] = DEFAULT_SERIES, *, refresh: bool = False,
                cache_dir: Path | None = None) -> pd.DataFrame:
    """A date-indexed DataFrame of the requested macro series (forward-filled)."""
    cache_dir = cache_dir or FRED_DIR
    path = cache_dir / "macro.csv"
    if path.exists() and not refresh:
        try:
            return pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception:
            pass
    cols = {}
    for name, sid in series.items():
        try:
            cols[name] = fetch_series(sid)
        except Exception:
            cols[name] = pd.Series(dtype=float, name=name)
    df = pd.DataFrame(cols).sort_index().ffill()
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def as_of_macro(macro: pd.DataFrame, as_of) -> dict[str, float]:
    """Macro features known as of ``as_of`` (last observation on/before it)."""
    as_of = pd.Timestamp(as_of)
    past = macro[macro.index <= as_of]
    if past.empty:
        return {}
    row = past.iloc[-1]
    out = {c: (float(row[c]) if pd.notna(row[c]) else None) for c in macro.columns}
    if out.get("rate_10y") is not None and out.get("rate_3m") is not None:
        out["term_spread"] = out["rate_10y"] - out["rate_3m"]     # yield-curve slope
    # 3-month change in the 10y, if available.
    prior = macro[macro.index <= as_of - pd.Timedelta(days=90)]
    if "rate_10y" in macro.columns and not prior.empty and out.get("rate_10y") is not None:
        p = prior["rate_10y"].iloc[-1]
        if pd.notna(p):
            out["rate_10y_chg_3m"] = out["rate_10y"] - float(p)
    return out
