"""The Cboe 'Available Weeklys' list — the universe of symbols with weekly options.

Cboe publishes, daily, the set of symbols that have weekly option expirations:
https://www.cboe.com/available_weeklys/get_csv_download/

Using this list as the screening universe guarantees every name actually has
public weekly chains, so we only need the market-cap filter on top of it. The
parsed list is cached to ``config/weeklys.csv`` (a small, human-readable file that
doubles as the offline fallback if the download ever fails).

The raw Cboe CSV starts with ~20 rows of index-product expiration schedules
(SPXW, XSP, VIX, NANOS, ...), then two labelled sections:

    Available Weeklys - Exchange Traded Products (ETFs and ETNs)
    ...ETF/ETN symbol rows...
    Available Weeklys - Equity
    ...equity symbol rows...

Each symbol row is ``[TICKER, "Company Name"]``. We skip everything before the
first section header and keep rows whose first cell looks like a ticker.
"""

from __future__ import annotations

import csv
import io
import re
import urllib.request
from pathlib import Path

from .config import CONFIG_DIR

WEEKLYS_URL = "https://www.cboe.com/available_weeklys/get_csv_download/"
WEEKLYS_CSV = CONFIG_DIR / "weeklys.csv"

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,6}$")


def parse_weeklys(text: str) -> list[dict]:
    """Parse Cboe's raw CSV text into ``[{symbol, type, name}, ...]``."""
    records: list[dict] = []
    section: str | None = None
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        c0 = row[0].strip()
        if "Exchange Traded Products" in c0:
            section = "ETF"
            continue
        if "Available Weeklys - Equity" in c0 or (c0.startswith("Available Weeklys") and "Equity" in c0):
            section = "EQUITY"
            continue
        if section is None:
            continue                      # still in the index-schedule preamble
        if c0 in ("", "Symbol") or not _TICKER_RE.match(c0):
            continue
        name = row[1].strip() if len(row) > 1 else ""
        records.append({"symbol": c0, "type": section, "name": name})
    return records


def fetch_weeklys(url: str = WEEKLYS_URL, timeout: int = 30) -> list[dict]:
    """Download and parse the live Cboe weeklys list."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", "replace")
    records = parse_weeklys(text)
    if not records:
        raise ValueError("Parsed 0 symbols from Cboe CSV — format may have changed.")
    return records


def save_weeklys(records: list[dict], path: Path = WEEKLYS_CSV) -> None:
    """Write the parsed list to our compact ``symbol,type,name`` CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "type", "name"])
        for rec in records:
            writer.writerow([rec["symbol"], rec["type"], rec["name"]])


def load_weeklys(path: Path = WEEKLYS_CSV) -> list[dict]:
    """Load the cached ``config/weeklys.csv``."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def refresh(path: Path = WEEKLYS_CSV) -> list[dict]:
    """Fetch the live list and overwrite the cached CSV. Returns the records."""
    records = fetch_weeklys()
    save_weeklys(records, path)
    return records


def get_symbols(*, refresh_first: bool = False, path: Path = WEEKLYS_CSV) -> list[str]:
    """De-duplicated symbol list for the screener.

    Refreshes from Cboe when asked or when no cache exists; if the download fails
    but a cached file is present, falls back to it rather than erroring out.
    """
    records: list[dict]
    if refresh_first or not path.exists():
        try:
            records = refresh(path)
        except Exception:
            if not path.exists():
                raise
            records = load_weeklys(path)
    else:
        records = load_weeklys(path)

    seen: set[str] = set()
    symbols: list[str] = []
    for rec in records:
        sym = (rec.get("symbol") or "").strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            symbols.append(sym)
    return symbols
