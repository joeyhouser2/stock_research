"""Single-ticker deep dive: the full OTM-call stat grid across strikes/expiries."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from . import data, fundamentals, metrics
from .config import REPO_ROOT, Settings

GRID_COLUMNS = [
    "expiry", "exp_type", "dte", "strike", "pct_otm", "mid", "annual_yield", "score",
    "if_called_yield", "prob_otm", "delta", "downside_cushion", "breakeven",
    "iv", "hv", "iv_hv", "open_interest", "volume", "spread_pct",
]


def build(
    ticker: str,
    settings: Settings,
    *,
    apply_liquidity: bool = False,
    with_value: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Return (grid DataFrame, header info) for one ticker.

    By default the deep dive shows *all* OTM strikes in the DTE window (liquidity
    is reported, not filtered) so you can see the whole surface. Pass
    ``apply_liquidity=True`` to drop illiquid strikes.
    """
    snap = data.get_snapshot(ticker)
    if snap is None:
        raise ValueError(f"No price/size data for {ticker!r}.")

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
            if apply_liquidity and not metrics.passes_liquidity(
                stat,
                min_oi=settings.min_open_interest,
                min_volume=settings.min_volume,
                max_spread=settings.max_spread_pct,
            ):
                continue
            stat["expiry"] = expiry
            stat["exp_type"] = exp_type
            rows.append(stat)

    grid = pd.DataFrame(rows).reindex(columns=GRID_COLUMNS) if rows else pd.DataFrame(columns=GRID_COLUMNS)
    if not grid.empty:
        grid = grid.sort_values(["expiry", "strike"]).reset_index(drop=True)

    header = {
        "ticker": ticker,
        "quote_type": snap.quote_type,
        "price": snap.price,
        "size_b": round(snap.size_usd / 1e9, 2),
        "dividend_yield": snap.dividend_yield,
        "hv": hv,
        "value": fundamentals.compute(snap.info, snap.price) if with_value else None,
    }
    return grid, header


def render_text(grid: pd.DataFrame, header: dict) -> str:
    """Format the grid + header as a console-friendly string."""
    h = header
    size_label = "AUM" if h["quote_type"] == "ETF" else "Mkt cap"
    hv_str = f"HV {h['hv']*100:.1f}%" if h["hv"] else "HV n/a"
    lines = [
        f"\n{h['ticker']}  ({h['quote_type']})   spot ${h['price']:.2f}   "
        f"{size_label} ${h['size_b']:.1f}B   div {h['dividend_yield']*100:.2f}%   {hv_str}"
    ]
    if h.get("value"):
        lines.append("  value:  " + _format_value(h["value"]))
    if grid.empty:
        lines.append("  No OTM calls in the configured DTE / OTM window.")
        return "\n".join(lines)
    lines.append(grid.to_string(index=False))
    return "\n".join(lines)


def _format_value(value: dict) -> str:
    """One-line summary of the underlying's valuation metrics."""
    def pct(key):
        v = value.get(key)
        return f"{v*100:+.1f}%" if v is not None else "n/a"

    def num(key):
        v = value.get(key)
        return f"{v:.2f}" if v is not None else "n/a"

    return (
        f"P/E {num('trailing_pe')} (fwd {num('forward_pe')})   "
        f"PEG {num('peg')}   P/B {num('price_to_book')}   "
        f"margin {pct('profit_margin')}   ROE {pct('roe')}   "
        f"off 52w-high {pct('pct_off_52w_high')}   analyst {pct('analyst_upside')}"
    )


def save_charts(grid: pd.DataFrame, header: dict, out_dir: Path | None = None) -> list[Path]:
    """Save annual-yield-vs-%OTM and prob-OTM-vs-%OTM PNGs. Returns written paths."""
    if grid.empty:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or (REPO_ROOT / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ticker = header["ticker"]
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 5))
    for expiry, grp in grid.groupby("expiry"):
        ax.plot(grp["pct_otm"] * 100, grp["annual_yield"] * 100, marker="o", label=str(expiry))
    ax.set_xlabel("% out of the money")
    ax.set_ylabel("Annualized premium yield (%)")
    ax.set_title(f"{ticker} — OTM call annualized yield")
    ax.legend(title="expiry", fontsize=8)
    ax.grid(True, alpha=0.3)
    p1 = out_dir / f"{ticker}_yield.png"
    fig.tight_layout(); fig.savefig(p1, dpi=120); plt.close(fig)
    written.append(p1)

    if grid["prob_otm"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        for expiry, grp in grid.groupby("expiry"):
            ax.plot(grp["pct_otm"] * 100, grp["prob_otm"] * 100, marker="o", label=str(expiry))
        ax.set_xlabel("% out of the money")
        ax.set_ylabel("P(expires OTM) (%)")
        ax.set_title(f"{ticker} — probability call expires worthless")
        ax.legend(title="expiry", fontsize=8)
        ax.grid(True, alpha=0.3)
        p2 = out_dir / f"{ticker}_probotm.png"
        fig.tight_layout(); fig.savefig(p2, dpi=120); plt.close(fig)
        written.append(p2)

    return written
