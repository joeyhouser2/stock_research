"""S&P 500 index-membership forecasting: who joins, who drops out.

Index additions and deletions are the episodic form of the passive-flow trade —
when a name enters the index, every fund tracking it is *required* to buy. Unlike
most equity signals the rules are published, so this is a constrained
classification problem rather than a guess:

* :mod:`~stock_research.spx.rules` — the published eligibility criteria as pure
  gate functions (market cap, float, GAAP viability, liquidity, seasoning).
* :mod:`~stock_research.spx.membership` — historical constituents and change
  events, reconstructed point-in-time so labels don't look ahead.
* :mod:`~stock_research.spx.panel` — as-of candidate features joined to add/drop
  labels, in the same no-look-ahead style as :mod:`stock_research.panel`.
* :mod:`~stock_research.spx.scan` — rank today's candidates for addition and
  today's members for deletion risk.
* :mod:`~stock_research.spx.model` — a purged walk-forward classifier over the
  panel, scored against the readiness rule it has to beat.

A word on expectations before you trade any of this: Greenwood & Sammon (*The
Disappearing Index Effect*, Journal of Finance 2025) find the addition abnormal
return fell from ~7.4% in the 1990s to ~0.3% in the 2010s, and deletions from
large negative to ~0.1%. Two causes both apply here — most adds are now S&P 400
migrations (index funds already hold the name), and the changes became
predictable enough to be front-run. Predicting membership correctly is the price
of entry, not an edge by itself.
"""

from . import membership, model, panel, rules, scan  # noqa: F401

__all__ = ["membership", "model", "panel", "rules", "scan"]
