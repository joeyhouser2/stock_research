"""Universe scan: rank tradable OTM call sales across the watchlist into a CSV."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pandas as pd

from . import cboe, data, fundamentals, metrics
from .cache import MarketCapCache
from .config import REPO_ROOT, Settings

# Stable column order for the output CSV.
COLUMNS = [
    "ticker", "quote_type", "size_b", "expiry", "exp_type", "dte", "strike", "spot",
    "pct_otm", "mid", "annual_yield", "score", "if_called_yield", "prob_otm", "delta",
    "downside_cushion", "breakeven", "iv", "hv", "iv_hv",
    "open_interest", "volume", "spread_pct", "contract",
]

# Columns the screener can rank by (descending).
SORT_KEYS = ("annual_yield", "score", "if_called_yield", "prob_otm", "downside_cushion")


# Map each value-filter setting to the fundamentals column it caps.
_PE_FILTERS = {
    "max_pe": "trailing_pe",
    "max_forward_pe": "forward_pe",
    "max_peg": "peg",
}


def _value_active(settings: Settings, with_value: bool) -> bool:
    """True if value metrics must be computed — explicitly asked, or a cap is set."""
    return with_value or any(getattr(settings, s) is not None for s in _PE_FILTERS)


def _passes_value_filters(value_cols: dict | None, settings: Settings) -> bool:
    """True if the underlying clears every active P/E-style cap.

    A cap with no figure available (e.g. an ETF has no P/E) fails — if you ask for
    'good P/E' we won't pass through names whose P/E we can't see.
    """
    for setting, col in _PE_FILTERS.items():
        cap = getattr(settings, setting)
        if cap is None:
            continue
        val = value_cols.get(col) if value_cols else None
        if val is None or val > cap:
            return False
    return True


def output_columns(with_value: bool) -> list[str]:
    """Full column list, with the value metrics appended when requested."""
    return COLUMNS + (fundamentals.VALUE_COLUMNS if with_value else [])


def analyze_ticker(
    ticker: str,
    settings: Settings,
    *,
    with_value: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """Return stat rows for every OTM call on ``ticker`` passing the filters."""
    snap = data.get_snapshot(ticker)
    if snap is None:
        _log(verbose, f"  {ticker}: no price/size data - skipped")
        return []
    if snap.size_usd < settings.min_market_cap:
        _log(verbose, f"  {ticker}: size ${snap.size_usd/1e9:.2f}B < floor - skipped")
        return []

    need_value = _value_active(settings, with_value)
    value_cols = fundamentals.compute(snap.info, snap.price) if need_value else None
    if not _passes_value_filters(value_cols, settings):
        pe = value_cols.get("trailing_pe") if value_cols else None
        _log(verbose, f"  {ticker}: P/E {pe} fails value cap - skipped")
        return []

    hv = data.historical_volatility(ticker, settings.hv_window)
    today = dt.date.today()
    rows: list[dict] = []

    for expiry in data.list_expirations(ticker):
        dte = data.days_to_expiry(expiry, today)
        if dte < settings.min_dte or dte > settings.max_dte:
            continue
        exp_type = data.classify_expiry(expiry)
        if settings.expiry_type != "any" and exp_type != settings.expiry_type:
            continue
        chain = data.get_call_chain(ticker, expiry)
        if chain.empty:
            continue
        for _, contract in chain.iterrows():
            stat = metrics.compute(
                row=contract,
                spot=snap.price,
                dte=dte,
                risk_free_rate=settings.risk_free_rate,
                hv=hv,
                dividend_yield=snap.dividend_yield,
            )
            if stat is None:
                continue
            if not (settings.min_otm <= stat["pct_otm"] <= settings.max_otm):
                continue
            if settings.min_prob_otm is not None and (
                stat["prob_otm"] is None or stat["prob_otm"] < settings.min_prob_otm
            ):
                continue
            if not metrics.passes_liquidity(
                stat,
                min_oi=settings.min_open_interest,
                min_volume=settings.min_volume,
                max_spread=settings.max_spread_pct,
            ):
                continue
            stat.update(
                ticker=ticker,
                quote_type=snap.quote_type,
                size_b=round(snap.size_usd / 1e9, 2),
                expiry=expiry,
                exp_type=exp_type,
            )
            if value_cols is not None:
                stat.update(value_cols)
            rows.append(stat)

    _log(verbose, f"  {ticker}: {len(rows)} qualifying OTM calls")
    return rows


def run(
    tickers: list[str],
    settings: Settings,
    *,
    sort_by: str = "annual_yield",
    with_value: bool = False,
    throttle: float = 0.0,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Scan every ticker, rank by ``sort_by`` (descending), and write a CSV.

    ``throttle`` pauses between tickers to ease Yahoo rate limits — important when
    scanning hundreds of names, since each does several option-chain requests.
    """
    if sort_by not in SORT_KEYS:
        raise ValueError(f"sort_by must be one of {SORT_KEYS}, got {sort_by!r}")
    # If a P/E-style cap is set we compute (and therefore can show) value metrics.
    show_value = _value_active(settings, with_value)
    all_rows: list[dict] = []
    for i, ticker in enumerate(tickers):
        try:
            all_rows.extend(
                analyze_ticker(ticker, settings, with_value=show_value, verbose=verbose)
            )
        except Exception as exc:  # one bad ticker shouldn't kill the run
            _log(verbose, f"  {ticker}: error {exc!r} - skipped")
        if throttle and i < len(tickers) - 1:
            time.sleep(throttle)

    columns = output_columns(show_value)
    if not all_rows:
        _log(verbose, "No qualifying contracts found.")
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(all_rows)
    df = df.reindex(columns=columns)
    df = df.sort_values(sort_by, ascending=False, na_position="last").reset_index(drop=True)
    df = df.head(settings.top)

    out_dir = out_dir or (REPO_ROOT / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"screen_{stamp}.csv"
    df.to_csv(out_path, index=False)
    _log(verbose, f"\nWrote {len(df)} rows -> {out_path}")
    return df


def qualifying_universe(
    settings: Settings,
    *,
    refresh_weeklys: bool = False,
    cache_ttl_days: float = 7,
    throttle: float = 0.0,
    max_tickers: int | None = None,
    verbose: bool = True,
) -> list[str]:
    """Symbols from the Cboe weeklys list whose size clears ``min_market_cap``.

    This is the slow pass — one size lookup per symbol — but it's cached, so only
    cache-misses hit the network. When ``max_tickers`` is set, the survivors are
    ranked by size (largest first) before truncation.
    """
    symbols = cboe.get_symbols(refresh_first=refresh_weeklys)
    _log(verbose, f"Weeklys universe: {len(symbols)} symbols. "
                  f"Filtering to market cap / AUM >= ${settings.min_market_cap/1e9:.1f}B...")

    cache = MarketCapCache(ttl_days=cache_ttl_days)
    survivors: list[tuple[str, float]] = []
    misses = 0
    for i, sym in enumerate(symbols, 1):
        rec = cache.get(sym)
        if rec is None:
            size, quote_type = data.get_size(sym)
            cache.put(sym, size, quote_type)
            misses += 1
            if throttle:
                time.sleep(throttle)
        else:
            size = rec.get("size_usd")
        if size is not None and size >= settings.min_market_cap:
            survivors.append((sym, size))
        if verbose and i % 50 == 0:
            _log(verbose, f"  ...{i}/{len(symbols)} checked, {len(survivors)} qualify")
    cache.save()

    survivors.sort(key=lambda pair: pair[1], reverse=True)
    qualified = [sym for sym, _ in survivors]
    _log(verbose, f"{len(qualified)} symbols qualify "
                  f"({misses} fetched, {len(symbols)-misses} from cache).")
    if max_tickers is not None and len(qualified) > max_tickers:
        _log(verbose, f"Capping to the {max_tickers} largest by size.")
        qualified = qualified[:max_tickers]
    return qualified


def run_weeklys(
    settings: Settings,
    *,
    sort_by: str = "annual_yield",
    with_value: bool = False,
    refresh_weeklys: bool = False,
    cache_ttl_days: float = 7,
    throttle: float = 0.0,
    max_tickers: int | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Screen the full $1B+ weeklys universe: gate by size, then scan survivors."""
    tickers = qualifying_universe(
        settings,
        refresh_weeklys=refresh_weeklys,
        cache_ttl_days=cache_ttl_days,
        throttle=throttle,
        max_tickers=max_tickers,
        verbose=verbose,
    )
    if not tickers:
        _log(verbose, "No symbols cleared the market-cap floor.")
        return pd.DataFrame(columns=output_columns(_value_active(settings, with_value)))
    _log(verbose, f"\nScanning option chains for {len(tickers)} symbols...")
    return run(tickers, settings, sort_by=sort_by, with_value=with_value,
               throttle=throttle, out_dir=out_dir, verbose=verbose)


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)
