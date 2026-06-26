"""On-disk cache of per-ticker size (market cap / AUM) for the universe gate.

Screening the whole weeklys universe means a market-cap lookup for every symbol
— the slow, rate-limited part. Market caps barely move day to day, so we cache
them and only re-fetch once the entry goes stale (default 7 days). After the first
full pass, reruns skip the network for every name whose size we already know, and
only pull fresh option chains for the few hundred that clear the $1B floor.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import REPO_ROOT

CACHE_DIR = REPO_ROOT / "cache"
MARKETCAP_CACHE = CACHE_DIR / "marketcaps.json"
DEFAULT_TTL_DAYS = 7


class MarketCapCache:
    """JSON map of ``ticker -> {size_usd, quote_type, ts}`` with TTL expiry."""

    def __init__(self, path: Path = MARKETCAP_CACHE, ttl_days: float = DEFAULT_TTL_DAYS,
                 now: float | None = None):
        self.path = Path(path)
        self.ttl = ttl_days * 86400
        self._now = now            # fixed clock for tests; else wall clock
        self.data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except (ValueError, OSError):
                self.data = {}

    def _time(self) -> float:
        return self._now if self._now is not None else time.time()

    def get(self, ticker: str) -> dict | None:
        """Return the cached record for ``ticker`` if present and still fresh."""
        rec = self.data.get(ticker.upper())
        if not rec:
            return None
        if self._time() - rec.get("ts", 0) > self.ttl:
            return None
        return rec

    def put(self, ticker: str, size_usd: float | None, quote_type: str) -> None:
        self.data[ticker.upper()] = {
            "size_usd": size_usd,
            "quote_type": quote_type,
            "ts": self._time(),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=0))
