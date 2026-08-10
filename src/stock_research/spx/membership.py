"""S&P 500 constituents and change events — the labels for everything else.

You can't train or score an add/drop model without knowing who was in the index
on each historical date, and S&P doesn't publish that history for free. The
practical free source is Wikipedia's component page, which carries both the
current list and a table of changes. Historical membership is then *reconstructed*
by walking the change table backwards from today's list:

    members(t) = today's list, with every change after t undone

Two honest caveats, surfaced in code rather than buried here:

* Wikipedia's change table is explicitly a *selected* list. Missed rows make the
  reconstruction drift, and the drift grows the further back you go.
  :func:`members_asof` reports its own uncertainty via
  :class:`Reconstruction.warnings`.
* Ticker symbols change (renames, reorganizations). A rename recorded as an
  add/drop pair reconstructs fine; one recorded as neither leaves a stale symbol
  in the historical set.

For a paper you'd pay for CRSP or S&P's own index history. For deciding whether
the signal is worth pursuing, this is enough — just don't quote precision it
can't support.

Parsing is pure and fixture-testable: :func:`parse_constituents` and
:func:`parse_changes` take HTML text, not URLs.
"""

from __future__ import annotations

import datetime as dt
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..config import REPO_ROOT

SPX_DIR = REPO_ROOT / "data" / "spx"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
SP600_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
CACHE_TTL_DAYS = 7

CHANGE_COLUMNS = ["date", "ticker", "action", "counterparty", "reason"]


# --------------------------------------------------------------------------- #
# Fetch + cache.
# --------------------------------------------------------------------------- #
def _user_agent() -> str:
    # Wikipedia rejects the stdlib default; identify the tool per their policy.
    return "stock_research/0.1 (https://github.com/; index membership research)"


def fetch_html(url: str, *, cache_name: str, refresh: bool = False,
               cache_dir: Path | None = None, ttl_days: float = CACHE_TTL_DAYS) -> str:
    """GET ``url``, caching to ``data/spx/<cache_name>`` with a TTL.

    Uses ``requests`` (and therefore certifi's CA bundle) rather than urllib,
    which validates against the OS trust store and fails on machines carrying an
    expired root.
    """
    cache_dir = cache_dir or SPX_DIR
    path = cache_dir / cache_name
    if path.exists() and not refresh:
        age = time.time() - path.stat().st_mtime
        if age <= ttl_days * 86400:
            return path.read_text(encoding="utf-8")
    resp = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=30)
    resp.raise_for_status()
    html = resp.text
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return html


# --------------------------------------------------------------------------- #
# HTML table parsing (pure).
# --------------------------------------------------------------------------- #
def _cell_text(cell) -> str:
    # Drop Wikipedia's footnote markers before collapsing whitespace.
    for sup in cell.find_all("sup"):
        sup.decompose()
    return " ".join(cell.get_text(" ", strip=True).split())


def table_matrix(table) -> list[list[str]]:
    """Dense text grid for an HTML table, expanding rowspan/colspan.

    The changes table puts one date across several rows via ``rowspan``; without
    expansion every multi-change day silently misdates its rows.
    """
    grid: list[list[str | None]] = []
    for r, row in enumerate(table.find_all("tr")):
        while len(grid) <= r:
            grid.append([])
        col = 0
        for cell in row.find_all(["td", "th"]):
            while col < len(grid[r]) and grid[r][col] is not None:
                col += 1
            text = _cell_text(cell)
            try:
                rowspan = max(1, int(cell.get("rowspan", 1)))
                colspan = max(1, int(cell.get("colspan", 1)))
            except (TypeError, ValueError):
                rowspan = colspan = 1
            for dr in range(rowspan):
                rr = r + dr
                while len(grid) <= rr:
                    grid.append([])
                for dc in range(colspan):
                    cc = col + dc
                    while len(grid[rr]) <= cc:
                        grid[rr].append(None)
                    grid[rr][cc] = text
            col += colspan
    return [[c if c is not None else "" for c in row] for row in grid]


def _find_table(soup: BeautifulSoup, table_id: str, header_hints: tuple[str, ...]):
    """Table by id, else the first table whose header row contains all hints."""
    table = soup.find("table", id=table_id)
    if table is not None:
        return table
    for candidate in soup.find_all("table"):
        head = " ".join(_cell_text(c) for c in candidate.find_all("th")[:12]).upper()
        if all(hint.upper() in head for hint in header_hints):
            return candidate
    return None


def parse_constituents(html: str) -> pd.DataFrame:
    """Current constituents: columns [ticker, security, sector]."""
    soup = BeautifulSoup(html, "html.parser")
    table = _find_table(soup, "constituents", ("Symbol", "Security"))
    if table is None:
        return pd.DataFrame(columns=["ticker", "security", "sector"])
    grid = table_matrix(table)
    if not grid:
        return pd.DataFrame(columns=["ticker", "security", "sector"])
    header = [h.upper() for h in grid[0]]
    i_sym = _index_of(header, "SYMBOL", "TICKER")
    i_sec = _index_of(header, "SECURITY", "COMPANY")
    i_gics = _index_of(header, "GICS SECTOR")
    rows = []
    for row in grid[1:]:
        if i_sym is None or i_sym >= len(row):
            continue
        ticker = normalize_ticker(row[i_sym])
        if not ticker:
            continue
        rows.append({
            "ticker": ticker,
            "security": row[i_sec] if i_sec is not None and i_sec < len(row) else "",
            "sector": row[i_gics] if i_gics is not None and i_gics < len(row) else "",
        })
    return pd.DataFrame(rows, columns=["ticker", "security", "sector"]).drop_duplicates("ticker")


def parse_changes(html: str) -> pd.DataFrame:
    """Change events, one row per (date, ticker, action).

    Columns: date, ticker, action ("add"/"drop"), counterparty, reason. A single
    swap produces two rows that reference each other through ``counterparty``.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = _find_table(soup, "changes", ("Added", "Removed"))
    if table is None:
        return pd.DataFrame(columns=CHANGE_COLUMNS)
    grid = table_matrix(table)

    # Header spans two rows: Date | Added(Ticker,Security) | Removed(...) | Reason.
    body_start = 0
    for i, row in enumerate(grid[:3]):
        joined = " ".join(row).upper()
        if "DATE" in joined or "TICKER" in joined or "SECURITY" in joined:
            body_start = i + 1
    rows = []
    for row in grid[body_start:]:
        if len(row) < 5:
            continue
        date = parse_date(row[0])
        if date is None:
            continue
        added, removed = normalize_ticker(row[1]), normalize_ticker(row[3])
        reason = row[5] if len(row) > 5 else ""
        if added:
            rows.append({"date": date, "ticker": added, "action": "add",
                         "counterparty": removed, "reason": reason})
        if removed:
            rows.append({"date": date, "ticker": removed, "action": "drop",
                         "counterparty": added, "reason": reason})
    df = pd.DataFrame(rows, columns=CHANGE_COLUMNS)
    if df.empty:
        return df
    return df.sort_values(["date", "ticker"]).drop_duplicates().reset_index(drop=True)


def normalize_ticker(text) -> str:
    """Wikipedia ticker cell -> a bare symbol, or "" when the cell is empty."""
    s = str(text or "").strip().upper()
    if not s or s in {"—", "-", "–", "N/A", "NAN", "NONE"}:
        return ""
    s = s.split()[0]
    s = re.sub(r"[^A-Z0-9.\-]", "", s)
    return s


def parse_date(text) -> dt.date | None:
    """Parse Wikipedia's date cells ("June 24, 2024", "2024-06-24")."""
    s = str(text or "").strip()
    if not s:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d %B %Y", "%B %d %Y"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _index_of(header: list[str], *names: str) -> int | None:
    for name in names:
        for i, h in enumerate(header):
            if name in h:
                return i
    return None


# --------------------------------------------------------------------------- #
# Point-in-time reconstruction.
# --------------------------------------------------------------------------- #
@dataclass
class Reconstruction:
    """Membership as of a date, with the bookkeeping that went wrong along the way."""

    as_of: dt.date
    members: frozenset[str]
    warnings: list[str] = field(default_factory=list)

    def __contains__(self, ticker: str) -> bool:
        return ticker.upper() in self.members

    def __len__(self) -> int:
        return len(self.members)


def members_asof(current: set[str] | list[str], changes: pd.DataFrame, as_of) -> Reconstruction:
    """Rewind ``current`` membership to ``as_of`` by undoing later changes.

    Every anomaly is recorded rather than silently absorbed: undoing an addition
    for a name that isn't currently a member, or undoing a deletion for one that
    already is, means the change table and the constituent list disagree — the
    signature of a missed or duplicated Wikipedia row.
    """
    as_of = _as_date(as_of)
    members = {str(t).upper() for t in current}
    warnings: list[str] = []
    if changes.empty:
        return Reconstruction(as_of, frozenset(members), warnings)

    later = changes[changes["date"].map(_as_date) > as_of]
    later = later.sort_values("date", ascending=False, kind="stable")
    for _, row in later.iterrows():
        ticker, action = str(row["ticker"]).upper(), str(row["action"])
        if action == "add":
            if ticker in members:
                members.discard(ticker)
            else:
                warnings.append(f"{row['date']}: undoing add of {ticker}, which was not a member")
        elif action == "drop":
            if ticker in members:
                warnings.append(f"{row['date']}: undoing drop of {ticker}, which was already a member")
            members.add(ticker)
    return Reconstruction(as_of, frozenset(members), warnings)


def membership_series(current, changes: pd.DataFrame, dates) -> dict[dt.date, frozenset[str]]:
    """``{date: members}`` for a series of dates (each independently rewound)."""
    return {_as_date(d): members_asof(current, changes, d).members for d in dates}


# --------------------------------------------------------------------------- #
# Labels.
# --------------------------------------------------------------------------- #
def label_events(changes: pd.DataFrame, as_of, horizon_months: int = 12,
                 action: str = "add") -> dict[str, dt.date]:
    """``{ticker: event_date}`` for changes in ``(as_of, as_of + horizon]``.

    The window is half-open on the left so a change *on* the as-of date isn't
    treated as forecastable — you'd be predicting something already announced.
    """
    as_of = _as_date(as_of)
    end = as_of + pd.DateOffset(months=horizon_months)
    end = _as_date(end)
    if changes.empty:
        return {}
    sel = changes[changes["action"] == action].copy()
    sel["date"] = sel["date"].map(_as_date)
    sel = sel[(sel["date"] > as_of) & (sel["date"] <= end)]
    return {str(r.ticker).upper(): r.date for r in sel.itertuples()}


def load(*, refresh: bool = False, cache_dir: Path | None = None
         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(constituents, changes)`` for the S&P 500, fetched or from cache."""
    html = fetch_html(SP500_URL, cache_name="sp500_wikipedia.html",
                      refresh=refresh, cache_dir=cache_dir)
    return parse_constituents(html), parse_changes(html)


def load_midcap400(*, refresh: bool = False, cache_dir: Path | None = None) -> pd.DataFrame:
    """S&P MidCap 400 constituents — the pool most S&P 500 additions come from."""
    html = fetch_html(SP400_URL, cache_name="sp400_wikipedia.html",
                      refresh=refresh, cache_dir=cache_dir)
    return parse_constituents(html)


def load_smallcap600(*, refresh: bool = False, cache_dir: Path | None = None) -> pd.DataFrame:
    """S&P SmallCap 600 constituents.

    Needed as *negatives*: a candidate panel built only from the 400 has an
    unrealistically strong non-member population, because names that fell from the
    400 down to the 600 are missing. Including the 600 makes "didn't get added"
    look like it actually does.
    """
    html = fetch_html(SP600_URL, cache_name="sp600_wikipedia.html",
                      refresh=refresh, cache_dir=cache_dir)
    return parse_constituents(html)


def _as_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    return dt.date.fromisoformat(str(value)[:10])
