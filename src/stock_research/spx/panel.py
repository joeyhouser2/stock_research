"""As-of candidate panel: eligibility features joined to add/drop labels.

Same contract as :mod:`stock_research.panel` — one row per (ticker, date) where
every feature uses only what was knowable on that date — but the features are the
S&P 500 eligibility gates and the labels are index membership changes inside a
forward window.

What's genuinely point-in-time here: market cap (as-of price x as-of reported
shares), the GAAP viability screen (quarters filed by the date), and both
liquidity screens (traded volume up to the date). What isn't, and is flagged as
such on every row:

* **exchange / domicile** come from today's profile and are applied backwards. In
  practice these almost never change, but a re-domiciled company will be scored
  wrong in the past.
* **float / IWF** has no free point-in-time source, so the panel leaves it
  unknown rather than backfilling today's value. The float gate therefore shows
  up in ``unknown`` for historical rows, which correctly marks the readiness
  score as an upper bound. The live scan does fill it in.
* **shares outstanding** is EDGAR's weighted-average diluted share count, which
  lags true shares outstanding and ignores buybacks mid-quarter. Market cap here
  is an estimate, typically within a percent or two for large caps.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from .. import edgar
from ..config import REPO_ROOT
from . import membership, rules

PANEL_DIR = REPO_ROOT / "data" / "spx"

LABEL_COLUMNS = ["was_member", "label_add", "label_drop", "event_date", "event_reason"]


# --------------------------------------------------------------------------- #
# Feature extraction (pure — no network).
# --------------------------------------------------------------------------- #
def _asof_slice(ohlcv: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame(columns=["Close", "Volume"])
    idx = ohlcv.index
    if getattr(idx, "tz", None) is not None:
        ohlcv = ohlcv.copy()
        ohlcv.index = idx.tz_localize(None)
    return ohlcv[ohlcv.index <= as_of]


def monthly_volumes(ohlcv: pd.DataFrame, as_of, n: int = 6) -> list[float]:
    """Shares traded in each of the ``n`` complete calendar months before ``as_of``.

    The month containing ``as_of`` is excluded — it's partial, and a partial month
    would understate volume and fail the screen spuriously.
    """
    as_of = pd.Timestamp(as_of)
    hist = _asof_slice(ohlcv, as_of)
    if hist.empty:
        return []
    by_month = hist.groupby([hist.index.year, hist.index.month])["Volume"].sum()
    by_month = by_month[by_month.index < (as_of.year, as_of.month)]
    return [float(v) for v in by_month.iloc[-n:]]


def annual_dollar_volume(ohlcv: pd.DataFrame, as_of, trading_days: int = 252) -> float | None:
    """Dollar value traded over the trailing year ending at ``as_of``."""
    hist = _asof_slice(ohlcv, pd.Timestamp(as_of))
    if hist.empty:
        return None
    window = hist.iloc[-trading_days:]
    if len(window) < trading_days // 2:      # too little history to annualize honestly
        return None
    traded = float((window["Close"] * window["Volume"]).sum())
    # Scale a short window up to a full year so the ratio stays comparable.
    return traded * trading_days / len(window)


def candidate_fields(fundamentals: pd.DataFrame, ohlcv: pd.DataFrame, as_of,
                     *, static: dict | None = None) -> dict | None:
    """The ``fields`` dict :func:`stock_research.spx.rules.evaluate` expects.

    Returns None when there's no price on or before ``as_of`` — the name wasn't
    trading yet, so there's nothing to evaluate.
    """
    as_of = pd.Timestamp(as_of)
    hist = _asof_slice(ohlcv, as_of)
    if hist.empty:
        return None
    price = float(hist["Close"].iloc[-1])
    if not (price > 0):
        return None

    snap = edgar.as_of_snapshot(fundamentals, as_of) if fundamentals is not None else {}
    shares = snap.get("shares_diluted")
    market_cap = price * shares if shares and shares > 0 else None

    fields: dict = {
        "market_cap": market_cap,
        "shares_outstanding": shares,
        "annual_dollar_volume": annual_dollar_volume(ohlcv, as_of),
        "monthly_volumes": monthly_volumes(ohlcv, as_of),
        "quarterly_earnings": (
            edgar.as_of_quarters(fundamentals, "net_income", as_of, n=4)
            if fundamentals is not None else []
        ),
        # Lower bound: the earliest bar we pulled, which is the IPO date only when
        # the requested history window reaches back past it.
        "first_trade_date": hist.index[0].date(),
        "price": price,
    }
    # The strict liquidity gate divides by float-adjusted cap, which is unknown
    # historically — so it comes back unresolved and the panel would carry no
    # liquidity information at all. Turnover (dollar volume / unadjusted cap) is
    # always computable and is the same quantity up to the IWF, so it stands in as
    # a model feature. It is NOT the published ratio; the gate stays unknown.
    fields["turnover"] = _safe_ratio(fields["annual_dollar_volume"], market_cap)
    fields.update(static or {})
    return fields


def _safe_ratio(numerator, denominator) -> float | None:
    if numerator is None or not denominator or float(denominator) <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


# --------------------------------------------------------------------------- #
# Panel assembly.
# --------------------------------------------------------------------------- #
def build_panel(data_by_ticker: dict, changes: pd.DataFrame, current_members,
                dates, *, horizon_months: int = 12,
                criteria: rules.Criteria | None = None) -> pd.DataFrame:
    """Rows of (ticker, date) eligibility features + forward add/drop labels.

    ``data_by_ticker`` maps ticker -> ``{"fundamentals": df, "ohlcv": df,
    "static": dict}``. ``changes`` is the membership change table from
    :mod:`~stock_research.spx.membership`.
    """
    criteria = criteria or rules.load_criteria()
    dates = [pd.Timestamp(d) for d in dates]

    # Membership and forward events are per-date, not per-ticker — compute once.
    per_date = {}
    for as_of in dates:
        recon = membership.members_asof(current_members, changes, as_of)
        per_date[as_of] = {
            "members": recon.members,
            "adds": membership.label_events(changes, as_of, horizon_months, "add"),
            "drops": membership.label_events(changes, as_of, horizon_months, "drop"),
            "reasons": _reason_map(changes, as_of, horizon_months),
        }

    rows = []
    for ticker, d in data_by_ticker.items():
        ticker = ticker.upper()
        fundamentals, ohlcv = d.get("fundamentals"), d.get("ohlcv")
        if ohlcv is None or ohlcv.empty:
            continue
        for as_of in dates:
            fields = candidate_fields(fundamentals, ohlcv, as_of, static=d.get("static"))
            if fields is None:
                continue
            ev = rules.evaluate(ticker, fields, as_of, criteria)
            row = ev.to_row()
            row["turnover"] = fields.get("turnover")
            ctx = per_date[as_of]
            add_date, drop_date = ctx["adds"].get(ticker), ctx["drops"].get(ticker)
            row.update({
                "was_member": ticker in ctx["members"],
                "label_add": int(add_date is not None),
                "label_drop": int(drop_date is not None),
                "event_date": (add_date or drop_date).isoformat() if (add_date or drop_date) else None,
                "event_reason": ctx["reasons"].get(ticker, ""),
                "horizon_months": horizon_months,
                "static_fields_are_current": True,
            })
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["as_of", "readiness"], ascending=[True, False]).reset_index(drop=True)


def _reason_map(changes: pd.DataFrame, as_of, horizon_months: int) -> dict[str, str]:
    """``{ticker: reason}`` for changes inside the label window."""
    if changes.empty:
        return {}
    as_of_d = pd.Timestamp(as_of).date()
    end = (pd.Timestamp(as_of) + pd.DateOffset(months=horizon_months)).date()
    sel = changes.copy()
    sel["date"] = sel["date"].map(lambda d: pd.Timestamp(d).date())
    sel = sel[(sel["date"] > as_of_d) & (sel["date"] <= end)]
    return {str(r.ticker).upper(): str(r.reason) for r in sel.itertuples()}


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def run(tickers: list[str], *, start=None, end=None, freq_months: int = 3,
        horizon_months: int = 12, history_days: int = 3650,
        out_dir: Path | None = None, refresh: bool = False,
        verbose: bool = True) -> pd.DataFrame:
    """Fetch EDGAR + prices + membership history and write the labelled panel."""
    from .. import data

    end = pd.Timestamp(end) if end else pd.Timestamp(dt.date.today())
    start = pd.Timestamp(start) if start else end - pd.DateOffset(years=5)
    dates = pd.date_range(start, end, freq=f"{freq_months}MS")

    constituents, changes = membership.load(refresh=refresh)
    current = set(constituents["ticker"]) if not constituents.empty else set()
    _log(verbose, f"Membership: {len(current)} current members, {len(changes)} change rows.")
    if not current:
        _log(verbose, "No constituents parsed — labels would be meaningless. Aborting.")
        return pd.DataFrame()

    data_by: dict[str, dict] = {}
    for ticker in tickers:
        ticker = ticker.upper()
        try:
            fundamentals = edgar.point_in_time_fundamentals(edgar.company_facts(ticker))
        except Exception as exc:
            _log(verbose, f"  {ticker}: EDGAR {exc!r} — no viability screen")
            fundamentals = None
        ohlcv = data.ohlcv_history(ticker, history_days)
        if ohlcv.empty:
            _log(verbose, f"  {ticker}: no price history — skipped")
            continue
        data_by[ticker] = {"fundamentals": fundamentals, "ohlcv": ohlcv,
                           "static": _static_fields(ticker)}
        _log(verbose, f"  {ticker}: {len(ohlcv)} bars, "
                      f"{0 if fundamentals is None else len(fundamentals)} facts")

    df = build_panel(data_by, changes, current, dates, horizon_months=horizon_months)
    if df.empty:
        _log(verbose, "Empty panel.")
        return df

    out_dir = out_dir or PANEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"spx_panel_{stamp}.csv"
    df.to_csv(out_path, index=False)
    _log(verbose, f"\n{len(df)} rows ({df['ticker'].nunique()} names x {len(dates)} dates), "
                  f"{int(df['label_add'].sum())} adds / {int(df['label_drop'].sum())} drops "
                  f"in the {horizon_months}-month window -> {out_path}")
    return df


def _static_fields(ticker: str) -> dict:
    """Exchange/country/float from today's profile. Current-value, not as-of."""
    from .. import data

    try:
        snap = data.get_snapshot(ticker)
    except Exception:
        return {}
    if snap is None:
        return {}
    info = snap.info or {}
    return {
        "exchange": _exchange_name(info.get("exchange") or info.get("fullExchangeName")),
        "country": info.get("country"),
    }


_EXCHANGE_CODES = {
    "NYQ": "NYSE", "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ", "NAS": "NASDAQ",
    "ASE": "NYSE American", "PCX": "NYSE Arca", "BTS": "Cboe BZX", "IEX": "IEX",
}


def _exchange_name(code) -> str | None:
    """Map Yahoo's exchange codes onto the methodology's exchange names."""
    if not code:
        return None
    return _EXCHANGE_CODES.get(str(code).strip().upper(), str(code))


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)
