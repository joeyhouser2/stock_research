"""Loading of the YAML config files and the resolved run settings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

# Repo root is two levels up from this file: src/stock_research/config.py -> repo/
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


@dataclass
class Settings:
    risk_free_rate: float = 0.04
    min_market_cap: float = 1_000_000_000
    max_market_cap: float | None = None     # value picker: cap the size (find smaller names)
    min_dte: int = 20
    max_dte: int = 50
    expiry_type: str = "any"   # "any" | "weekly" | "monthly"

    # Underlying value filters (None = off). A contract is dropped when its
    # underlying's metric is above the cap or missing the figure entirely.
    max_pe: float | None = None
    max_forward_pe: float | None = None
    max_peg: float | None = None
    min_prob_otm: float | None = None
    min_iv_hv: float | None = None     # keep only contracts with IV/HV >= this (rich vol)
    min_otm: float = 0.02
    max_otm: float = 0.15
    min_open_interest: int = 50
    min_volume: int = 1
    max_spread_pct: float = 0.15
    hv_window: int = 30
    top: int = 40

    # Expected-return (simulation drift) model. See expected_return.py.
    drift_model: str = "fundamental"   # "fixed" | "fundamental" | "analyst" | "blend"
    equity_risk_premium: float = 0.045
    pe_anchor: float | None = None     # None -> PEG=1 / market-default anchor
    pe_reversion_years: float = 5.0
    pe_reversion_shrink: float = 1.0


def load_settings(path: Path | None = None) -> Settings:
    path = path or (CONFIG_DIR / "settings.yaml")
    data = {}
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
    known = {f for f in Settings().__dataclass_fields__}  # type: ignore[attr-defined]
    return Settings(**{k: v for k, v in data.items() if k in known})


def override(settings: Settings, **kwargs) -> Settings:
    """Return a copy of settings with non-None overrides applied."""
    clean = {k: v for k, v in kwargs.items() if v is not None}
    return replace(settings, **clean)


def load_universe(path: Path | None = None) -> list[str]:
    """Flatten the etfs + stocks lists into a de-duplicated ticker list."""
    path = path or (CONFIG_DIR / "universe.yaml")
    data = yaml.safe_load(path.read_text()) or {}
    tickers: list[str] = []
    for group in ("etfs", "stocks"):
        for t in data.get(group, []) or []:
            t = str(t).strip().upper()
            if t and t not in tickers:
                tickers.append(t)
    return tickers
