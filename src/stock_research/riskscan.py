"""Universe risk scan: rank names by simulated risk/return into a CSV.

Where ``screener`` ranks individual option contracts, this ranks the *underlyings*
by their simulated forward risk/return over a horizon — drift, volatility, a
Sharpe-like ratio, the odds of touching +/-X%, and tail risk (VaR / CVaR / mean
max drawdown). It's the "which names should I be looking at" view that feeds the
option screen. Each name needs only a price snapshot and a return history (no
option chains), so it's lighter than a full screen.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pandas as pd

from . import data, screener, simulate
from .config import REPO_ROOT, Settings

COLUMNS = [
    "ticker", "quote_type", "size_b", "spot", "drift", "sigma", "sharpe",
    "exp_return", "prob_up", "prob_down", "prob_term_up", "var", "cvar", "mdd",
    "horizon_days", "model",
]

# Columns the scan can rank by. Risk metrics rank ascending (lower is better);
# everything else descending.
SORT_KEYS = ("sharpe", "drift", "exp_return", "prob_up", "prob_down", "var", "cvar", "mdd")
_ASCENDING = {"var", "cvar", "mdd", "prob_down"}

SIM_LOOKBACK = 504


def analyze_ticker(
    ticker: str,
    settings: Settings,
    *,
    horizon_days: int,
    target_pct: float,
    model: str,
    n_paths: int,
    seed: int = 12345,
    verbose: bool = False,
) -> dict | None:
    """Simulate one name and return its risk-summary row (or None if skipped)."""
    snap = data.get_snapshot(ticker)
    if snap is None:
        _log(verbose, f"  {ticker}: no price/size data - skipped")
        return None
    if snap.size_usd < settings.min_market_cap:
        _log(verbose, f"  {ticker}: size ${snap.size_usd/1e9:.2f}B < floor - skipped")
        return None

    returns = data.daily_log_returns(ticker, SIM_LOOKBACK)
    drift, garch = simulate.prepare_drift_and_garch(
        info=snap.info, price=snap.price, dividend_yield=snap.dividend_yield,
        returns=returns, settings=settings, sim_model=model)
    sim = simulate.simulate(
        spot=snap.price, returns=returns, horizon_days=horizon_days, model=model,
        mu=drift.annual_drift, garch=garch, n_paths=n_paths, seed=seed)

    row = simulate.summarize_name(sim, drift, target_pct=target_pct,
                                  rf=settings.risk_free_rate)
    row.update(ticker=ticker, quote_type=snap.quote_type,
               size_b=round(snap.size_usd / 1e9, 2))
    _log(verbose, f"  {ticker}: drift {row['drift']:+.1%}  sigma {row['sigma']:.0%}  "
                  f"P(touch +{target_pct:.0%}) {row['prob_up']:.0%}  "
                  f"P(touch -{target_pct:.0%}) {row['prob_down']:.0%}")
    return row


def run(
    tickers: list[str],
    settings: Settings,
    *,
    horizon_days: int | None = None,
    target_pct: float = 0.05,
    model: str = "garch",
    n_paths: int = 30_000,
    sort_by: str = "sharpe",
    throttle: float = 0.0,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Simulate every name, rank by ``sort_by``, and write a CSV."""
    if sort_by not in SORT_KEYS:
        raise ValueError(f"sort_by must be one of {SORT_KEYS}, got {sort_by!r}")
    horizon_days = horizon_days or settings.max_dte

    rows: list[dict] = []
    for i, ticker in enumerate(tickers):
        try:
            row = analyze_ticker(ticker, settings, horizon_days=horizon_days,
                                 target_pct=target_pct, model=model, n_paths=n_paths,
                                 verbose=verbose)
            if row is not None:
                rows.append(row)
        except Exception as exc:  # one bad ticker shouldn't kill the run
            _log(verbose, f"  {ticker}: error {exc!r} - skipped")
        if throttle and i < len(tickers) - 1:
            time.sleep(throttle)

    if not rows:
        _log(verbose, "No names simulated.")
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    df = df.sort_values(sort_by, ascending=(sort_by in _ASCENDING),
                        na_position="last").reset_index(drop=True)

    out_dir = out_dir or (REPO_ROOT / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"risk_{stamp}.csv"
    df.to_csv(out_path, index=False)
    _log(verbose, f"\nWrote {len(df)} rows -> {out_path}")
    return df


def run_weeklys(
    settings: Settings,
    *,
    horizon_days: int | None = None,
    target_pct: float = 0.05,
    model: str = "garch",
    n_paths: int = 30_000,
    sort_by: str = "sharpe",
    refresh_weeklys: bool = False,
    cache_ttl_days: float = 7,
    throttle: float = 0.0,
    max_tickers: int | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Risk-scan the full $1B+ weeklys universe (size gate, then simulate)."""
    tickers = screener.qualifying_universe(
        settings, refresh_weeklys=refresh_weeklys, cache_ttl_days=cache_ttl_days,
        throttle=throttle, max_tickers=max_tickers, verbose=verbose)
    if not tickers:
        _log(verbose, "No symbols cleared the market-cap floor.")
        return pd.DataFrame(columns=COLUMNS)
    _log(verbose, f"\nSimulating {len(tickers)} symbols...")
    return run(tickers, settings, horizon_days=horizon_days, target_pct=target_pct,
               model=model, n_paths=n_paths, sort_by=sort_by, throttle=throttle,
               out_dir=out_dir, verbose=verbose)


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)
