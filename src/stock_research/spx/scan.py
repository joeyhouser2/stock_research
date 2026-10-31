"""Today's candidates: who is closest to joining, and who looks closest to leaving.

Two rankings, both built on :mod:`~stock_research.spx.rules`:

* **additions** — non-members ranked by readiness (how close the binding gate is).
  The default candidate pool is the S&P MidCap 400, because most S&P 500
  additions are migrations up from the 400.
* **deletions** — current members ranked by a market-cap-and-viability pressure
  heuristic. Read this as "visible pressure", not a forecast: real deletions are
  dominated by mergers and bankruptcies, which no fundamental screen anticipates.

Unlike the historical panel this uses live float and profile data, so the float
and IWF gates actually resolve.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from .. import edgar
from . import membership, panel, rules

OUT_DIR = panel.PANEL_DIR


def live_fields(ticker: str, *, history_days: int = 1825) -> dict | None:
    """Everything the gates need for ``ticker`` right now, or None if unavailable."""
    from .. import data

    ohlcv = data.ohlcv_history(ticker, history_days)
    if ohlcv.empty:
        return None
    as_of = ohlcv.index[-1]

    try:
        fundamentals = edgar.point_in_time_fundamentals(edgar.company_facts(ticker))
    except Exception:
        fundamentals = None

    fields = panel.candidate_fields(fundamentals, ohlcv, as_of)
    if fields is None:
        return None

    snap = None
    try:
        snap = data.get_snapshot(ticker)
    except Exception:
        pass
    info = (snap.info if snap else {}) or {}

    # Live profile beats the EDGAR estimate: real market cap, real float.
    if info.get("marketCap"):
        fields["market_cap"] = float(info["marketCap"])
    if info.get("sharesOutstanding"):
        fields["shares_outstanding"] = float(info["sharesOutstanding"])
    if info.get("floatShares"):
        fields["float_shares"] = float(info["floatShares"])
    fields["exchange"] = panel._exchange_name(
        info.get("exchange") or info.get("fullExchangeName"))
    fields["country"] = info.get("country")
    first = info.get("firstTradeDateEpochUtc")
    if first:
        try:
            fields["first_trade_date"] = dt.datetime.utcfromtimestamp(int(first)).date()
        except (TypeError, ValueError, OSError):
            pass
    fields["as_of"] = as_of.date()
    return fields


def scan_additions(tickers: list[str], *, criteria: rules.Criteria | None = None,
                   history_days: int = 1825, verbose: bool = True) -> pd.DataFrame:
    """Rank non-member candidates by readiness for inclusion."""
    criteria = criteria or rules.load_criteria()
    rows = []
    for ticker in tickers:
        ticker = ticker.upper()
        fields = live_fields(ticker, history_days=history_days)
        if fields is None:
            _log(verbose, f"  {ticker}: no data — skipped")
            continue
        ev = rules.evaluate(ticker, fields, fields["as_of"], criteria)
        row = ev.to_row()
        row["side"] = "add"
        rows.append(row)
        _log(verbose, f"  {ticker}: readiness {ev.readiness:.2f} "
                      f"binding={ev.binding.name if ev.binding else 'n/a'}"
                      + (f" blocks={','.join(ev.hard_blocks)}" if ev.hard_blocks else ""))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("readiness", ascending=False).reset_index(drop=True)


def scan_deletions(tickers: list[str], *, criteria: rules.Criteria | None = None,
                   history_days: int = 1825, verbose: bool = True) -> pd.DataFrame:
    """Rank current members by deletion pressure (see :func:`rules.deletion_pressure`)."""
    criteria = criteria or rules.load_criteria()
    rows = []
    for ticker in tickers:
        ticker = ticker.upper()
        fields = live_fields(ticker, history_days=history_days)
        if fields is None:
            _log(verbose, f"  {ticker}: no data — skipped")
            continue
        ev, risk = rules.deletion_pressure(ticker, fields, fields["as_of"], criteria)
        row = ev.to_row()
        row["deletion_risk"] = risk
        row["side"] = "drop"
        rows.append(row)
        _log(verbose, f"  {ticker}: deletion risk {risk:.2f} "
                      f"(cap {_ratio_txt(ev.gate('market_cap'))} of add threshold)")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("deletion_risk", ascending=False).reset_index(drop=True)


def run(*, side: str = "add", tickers: list[str] | None = None, limit: int | None = None,
        refresh: bool = False, history_days: int = 1825,
        out_dir: Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """Fetch membership, resolve the candidate universe, scan, and write a CSV.

    ``side`` is "add", "drop", or "both". With no explicit ``tickers``, additions
    scan the S&P MidCap 400 and deletions scan the current S&P 500.
    """
    constituents, _changes = membership.load(refresh=refresh)
    members = set(constituents["ticker"]) if not constituents.empty else set()
    if not members:
        _log(verbose, "Could not parse current constituents — aborting.")
        return pd.DataFrame()
    _log(verbose, f"{len(members)} current S&P 500 members.")

    criteria = rules.load_criteria()
    _, threshold_warnings = criteria.min_market_cap(dt.date.today())
    for w in threshold_warnings:
        _log(verbose, f"WARNING: {w}")

    frames = []
    if side in ("add", "both"):
        pool = tickers or _midcap_pool(members, refresh=refresh, verbose=verbose)
        pool = [t for t in pool if t.upper() not in members]
        if limit:
            pool = pool[:limit]
        _log(verbose, f"\nScanning {len(pool)} addition candidates:")
        frames.append(scan_additions(pool, criteria=criteria,
                                     history_days=history_days, verbose=verbose))
    if side in ("drop", "both"):
        pool = tickers or sorted(members)
        if limit:
            pool = pool[:limit]
        _log(verbose, f"\nScanning {len(pool)} members for deletion pressure:")
        frames.append(scan_deletions(pool, criteria=criteria,
                                     history_days=history_days, verbose=verbose))

    frames = [f for f in frames if not f.empty]
    if not frames:
        _log(verbose, "Nothing scanned.")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"spx_scan_{side}_{stamp}.csv"
    df.to_csv(out_path, index=False)
    _log(verbose, f"\n{len(df)} rows -> {out_path}")
    return df


def _midcap_pool(members: set[str], *, refresh: bool, verbose: bool) -> list[str]:
    try:
        mid = membership.load_midcap400(refresh=refresh)
    except Exception as exc:
        _log(verbose, f"MidCap 400 fetch failed ({exc!r}); pass --tickers explicitly.")
        return []
    return [t for t in mid["ticker"].tolist() if t not in members]


def _ratio_txt(gate) -> str:
    return "n/a" if gate is None or gate.ratio is None else f"{gate.ratio:.2f}x"


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)
