"""S&P 500 eligibility criteria as pure, testable gate functions.

Everything here is arithmetic over a dict of already-fetched fields — no network,
no pandas — so the rules can be unit-tested against the published methodology and
reused unchanged by both the historical panel and the live scan.

The output deliberately keeps two things apart:

* **hard blocks** — binary criteria a name either meets or doesn't (positive GAAP
  earnings, seasoning, domicile, exchange). One failure makes it ineligible today
  regardless of how good everything else looks.
* **readiness** — how close the *continuous* criteria are to their thresholds,
  reported as the worst (binding) ratio. 0.80 means "the tightest gate is at 80%
  of what it needs". This is the sortable number for "who's next".

Missing data is never silently treated as a pass. A gate with no value comes back
``passed=None`` and lands in :attr:`Evaluation.unknown`, which flags the readiness
score as an upper bound.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from ..config import CONFIG_DIR

THRESHOLDS_PATH = CONFIG_DIR / "spx_thresholds.yaml"

# Members aren't deleted at a bright line — the methodology says names are removed
# when they "substantially" violate the criteria, at committee discretion. This
# fraction of the current *addition* threshold is our heuristic review trigger for
# ranking deletion risk. It is OUR number, not S&P's.
MEMBER_REVIEW_FRACTION = 0.50


# --------------------------------------------------------------------------- #
# Threshold table.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Criteria:
    """Resolved eligibility parameters, loaded from ``config/spx_thresholds.yaml``."""

    cap_schedule: tuple[dict, ...]
    float_cap_fraction: float
    min_iwf: float
    min_liquidity_ratio: float
    min_monthly_volume: float
    seasoning_months: int
    viability_quarters: int
    eligible_exchanges: frozenset[str]
    eligible_countries: frozenset[str]
    staleness_warn_days: int

    def min_market_cap(self, as_of) -> tuple[float, list[str]]:
        """Minimum unadjusted market cap in force on ``as_of``, plus any caveats.

        Returns ``(usd, warnings)``. Warnings call out the two ways this can lie:
        a date before the table starts, and a table that hasn't been topped up
        since the last quarterly review.
        """
        as_of = _as_date(as_of)
        warnings: list[str] = []
        rows = self.cap_schedule
        applicable = [r for r in rows if r["effective"] <= as_of]
        if not applicable:
            row = rows[0]
            warnings.append(
                f"{as_of} precedes the threshold table (starts {row['effective']}); "
                f"using the earliest known ${row['min_cap_b']}B."
            )
        else:
            row = applicable[-1]
            if not row.get("verified", False):
                warnings.append(
                    f"threshold ${row['min_cap_b']}B effective {row['effective']} is "
                    f"unverified — see config/spx_thresholds.yaml."
                )
            if row is rows[-1]:
                age = (as_of - row["effective"]).days
                if age > self.staleness_warn_days:
                    warnings.append(
                        f"newest threshold row is {age} days old; S&P reviews the range "
                        f"quarterly, so ${row['min_cap_b']}B may be superseded."
                    )
        return float(row["min_cap_b"]) * 1e9, warnings


def load_criteria(path: Path | None = None) -> Criteria:
    raw = yaml.safe_load((path or THRESHOLDS_PATH).read_text()) or {}
    schedule = []
    for row in raw.get("sp500_min_market_cap", []) or []:
        schedule.append({
            "effective": _as_date(row["effective"]),
            "min_cap_b": float(row["min_cap_b"]),
            "verified": bool(row.get("verified", False)),
        })
    schedule.sort(key=lambda r: r["effective"])
    if not schedule:
        raise ValueError(f"no sp500_min_market_cap rows in {path or THRESHOLDS_PATH}")

    c = raw.get("constants", {}) or {}
    return Criteria(
        cap_schedule=tuple(schedule),
        float_cap_fraction=float(c.get("float_cap_fraction", 0.50)),
        min_iwf=float(c.get("min_iwf", 0.10)),
        min_liquidity_ratio=float(c.get("min_liquidity_ratio", 0.75)),
        min_monthly_volume=float(c.get("min_monthly_volume", 250_000)),
        seasoning_months=int(c.get("seasoning_months", 12)),
        viability_quarters=int(c.get("viability_quarters", 4)),
        eligible_exchanges=frozenset(_norm(x) for x in c.get("eligible_exchanges", [])),
        eligible_countries=frozenset(_norm(x) for x in c.get("eligible_countries", [])),
        staleness_warn_days=int(raw.get("staleness_warn_days", 200)),
    )


# --------------------------------------------------------------------------- #
# Gate results.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Gate:
    """One criterion's verdict.

    ``ratio`` is value/threshold for continuous gates (>= 1.0 passes) and is None
    for binary ones. ``passed=None`` means the input data was missing.
    """

    name: str
    kind: str                    # "continuous" | "binary"
    passed: bool | None
    value: float | None = None
    threshold: float | None = None
    ratio: float | None = None
    detail: str = ""


@dataclass(frozen=True)
class Evaluation:
    ticker: str
    as_of: dt.date
    gates: tuple[Gate, ...]
    warnings: tuple[str, ...] = ()

    def gate(self, name: str) -> Gate | None:
        return next((g for g in self.gates if g.name == name), None)

    @property
    def hard_blocks(self) -> list[str]:
        """Binary criteria the name currently fails outright."""
        return [g.name for g in self.gates if g.kind == "binary" and g.passed is False]

    @property
    def failed(self) -> list[str]:
        return [g.name for g in self.gates if g.passed is False]

    @property
    def unknown(self) -> list[str]:
        """Criteria we couldn't evaluate — the reason readiness is an upper bound."""
        return [g.name for g in self.gates if g.passed is None]

    @property
    def eligible(self) -> bool | None:
        """True/False if every gate resolved; None while anything is unknown."""
        if any(g.passed is False for g in self.gates):
            return False
        if self.unknown:
            return None
        return True

    @property
    def binding(self) -> Gate | None:
        """The continuous gate furthest from being met — the thing to watch."""
        scored = [g for g in self.gates if g.kind == "continuous" and g.ratio is not None]
        return min(scored, key=lambda g: g.ratio) if scored else None

    @property
    def readiness(self) -> float:
        """0..1: how close the binding continuous gate is, 0 if hard-blocked.

        Clipped at 1 per gate so a name with an enormous market cap doesn't
        paper over a liquidity gate it barely clears.
        """
        if self.hard_blocks:
            return 0.0
        binding = self.binding
        if binding is None:
            return 0.0
        return round(min(float(binding.ratio), 1.0), 4)

    @property
    def readiness_is_upper_bound(self) -> bool:
        return bool(self.unknown)

    def to_row(self) -> dict[str, Any]:
        """Flat dict for a DataFrame: one column per gate ratio/pass, plus summary."""
        row: dict[str, Any] = {"ticker": self.ticker, "as_of": self.as_of.isoformat()}
        for g in self.gates:
            row[f"pass_{g.name}"] = g.passed
            if g.kind == "continuous":
                row[f"ratio_{g.name}"] = None if g.ratio is None else round(g.ratio, 4)
                row[f"value_{g.name}"] = g.value
        row["readiness"] = self.readiness
        row["binding_gate"] = self.binding.name if self.binding else None
        row["hard_blocks"] = ";".join(self.hard_blocks)
        row["unknown"] = ";".join(self.unknown)
        row["eligible"] = self.eligible
        return row


# --------------------------------------------------------------------------- #
# The gates.
# --------------------------------------------------------------------------- #
def _continuous(name: str, value, threshold) -> Gate:
    """Gate that passes when value >= threshold, carrying the distance as a ratio."""
    if value is None or threshold is None or not (threshold > 0):
        return Gate(name, "continuous", None, _f(value), _f(threshold),
                    detail="missing input")
    value, threshold = float(value), float(threshold)
    return Gate(name, "continuous", value >= threshold, value, threshold,
                ratio=value / threshold)


def _binary(name: str, passed: bool | None, detail: str = "") -> Gate:
    return Gate(name, "binary", passed, detail=detail)


def gate_market_cap(fields: dict, min_cap: float) -> Gate:
    return _continuous("market_cap", fields.get("market_cap"), min_cap)


def gate_float_market_cap(fields: dict, min_cap: float, criteria: Criteria) -> Gate:
    """Float-adjusted cap vs. half the unadjusted minimum."""
    return _continuous("float_market_cap", _float_cap(fields),
                       min_cap * criteria.float_cap_fraction)


def gate_iwf(fields: dict, criteria: Criteria) -> Gate:
    return _continuous("iwf", _iwf(fields), criteria.min_iwf)


def gate_liquidity_ratio(fields: dict, criteria: Criteria) -> Gate:
    """Annual dollar value traded / float-adjusted market cap."""
    traded, float_cap = fields.get("annual_dollar_volume"), _float_cap(fields)
    ratio_value = None
    if traded is not None and float_cap and float_cap > 0:
        ratio_value = float(traded) / float(float_cap)
    return _continuous("liquidity_ratio", ratio_value, criteria.min_liquidity_ratio)


def gate_monthly_volume(fields: dict, criteria: Criteria) -> Gate:
    """Every one of the last six months must clear the share-volume floor.

    Scored on the *weakest* month, since that's the one that binds.
    """
    vols = _clean_numbers(fields.get("monthly_volumes"))
    if len(vols) < 6:
        return Gate("monthly_volume", "continuous", None,
                    detail=f"need 6 monthly volumes, have {len(vols)}")
    return _continuous("monthly_volume", min(vols[-6:]), criteria.min_monthly_volume)


def gate_viability(fields: dict, criteria: Criteria) -> Gate:
    """Positive GAAP earnings in the most recent quarter AND summed over four.

    ``quarterly_earnings`` is oldest-first; only quarters filed as of the
    evaluation date should be passed in.
    """
    q = _clean_numbers(fields.get("quarterly_earnings"))
    n = criteria.viability_quarters
    if len(q) < n:
        return _binary("viability", None, f"need {n} quarters of GAAP earnings, have {len(q)}")
    recent = q[-n:]
    latest_ok, sum_ok = recent[-1] > 0, sum(recent) > 0
    detail = f"latest {recent[-1]:,.0f}, {n}q sum {sum(recent):,.0f}"
    return _binary("viability", latest_ok and sum_ok, detail)


def gate_seasoning(fields: dict, as_of, criteria: Criteria) -> Gate:
    """At least 12 months of trading on an eligible exchange."""
    first = fields.get("first_trade_date")
    if first is None:
        return _binary("seasoning", None, "no first trade date")
    months = _months_between(_as_date(first), _as_date(as_of))
    return _binary("seasoning", months >= criteria.seasoning_months,
                   f"{months} months listed")


def gate_exchange(fields: dict, criteria: Criteria) -> Gate:
    ex = fields.get("exchange")
    if not ex:
        return _binary("exchange", None, "unknown exchange")
    return _binary("exchange", _norm(ex) in criteria.eligible_exchanges, str(ex))


def gate_domicile(fields: dict, criteria: Criteria) -> Gate:
    country = fields.get("country")
    if not country:
        return _binary("domicile", None, "unknown country")
    return _binary("domicile", _norm(country) in criteria.eligible_countries, str(country))


ALL_GATES = ("market_cap", "float_market_cap", "iwf", "liquidity_ratio",
             "monthly_volume", "viability", "seasoning", "exchange", "domicile")


def evaluate(ticker: str, fields: dict, as_of, criteria: Criteria | None = None) -> Evaluation:
    """Run every criterion for one name at one date.

    ``fields`` keys (all optional; missing ones become unknown gates):
        market_cap, shares_outstanding, float_shares, iwf, annual_dollar_volume,
        monthly_volumes (oldest-first), quarterly_earnings (oldest-first, GAAP
        net income), first_trade_date, exchange, country.
    """
    criteria = criteria or load_criteria()
    as_of = _as_date(as_of)
    min_cap, warnings = criteria.min_market_cap(as_of)

    gates = (
        gate_market_cap(fields, min_cap),
        gate_float_market_cap(fields, min_cap, criteria),
        gate_iwf(fields, criteria),
        gate_liquidity_ratio(fields, criteria),
        gate_monthly_volume(fields, criteria),
        gate_viability(fields, criteria),
        gate_seasoning(fields, as_of, criteria),
        gate_exchange(fields, criteria),
        gate_domicile(fields, criteria),
    )
    return Evaluation(ticker.upper(), as_of, gates, tuple(warnings))


def deletion_pressure(ticker: str, fields: dict, as_of,
                      criteria: Criteria | None = None) -> tuple[Evaluation, float]:
    """Evaluation plus a 0..1 deletion-risk heuristic for a current member.

    S&P removes members that "substantially" violate the criteria, at committee
    discretion — there is no published deletion threshold, so this is a ranking
    aid, not a rule. Risk rises as the name falls below
    :data:`MEMBER_REVIEW_FRACTION` of the current *addition* market-cap threshold,
    and is forced high when GAAP viability breaks.

    Most real deletions are M&A and bankruptcy, which no fundamental screen sees
    coming — treat a low score as "no visible pressure", not "safe".
    """
    ev = evaluate(ticker, fields, as_of, criteria)
    cap_gate = ev.gate("market_cap")
    if cap_gate is None or cap_gate.ratio is None:
        risk = 0.0 if not ev.hard_blocks else 0.75
    else:
        # ratio 1.0 (still add-eligible) -> 0 risk; at/below the review fraction -> 1.
        span = 1.0 - MEMBER_REVIEW_FRACTION
        risk = (1.0 - cap_gate.ratio) / span if span > 0 else 0.0
        risk = min(max(risk, 0.0), 1.0)
    if "viability" in ev.hard_blocks:
        risk = max(risk, 0.75)
    return ev, round(risk, 4)


# --------------------------------------------------------------------------- #
# Field helpers.
# --------------------------------------------------------------------------- #
def _iwf(fields: dict) -> float | None:
    """Investable weight factor: given directly, or float shares / shares out."""
    iwf = fields.get("iwf")
    if iwf is not None:
        return float(iwf)
    fs, so = fields.get("float_shares"), fields.get("shares_outstanding")
    if fs is None or not so or float(so) <= 0:
        return None
    return float(fs) / float(so)


def _float_cap(fields: dict) -> float | None:
    """Float-adjusted market cap = market cap x IWF."""
    cap, iwf = fields.get("market_cap"), _iwf(fields)
    if cap is None or iwf is None:
        return None
    return float(cap) * float(iwf)


def _clean_numbers(seq: Iterable | None) -> list[float]:
    if seq is None:
        return []
    out = []
    for x in seq:
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if v == v:                       # drop NaN
            out.append(v)
    return out


def _months_between(start: dt.date, end: dt.date) -> int:
    months = (end.year - start.year) * 12 + (end.month - start.month)
    return months - 1 if end.day < start.day else months


def _as_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _norm(value) -> str:
    return " ".join(str(value).strip().upper().split())


def _f(value) -> float | None:
    return None if value is None else float(value)


def render_text(evals: Sequence[Evaluation]) -> str:
    """Human-readable summary, most-ready first."""
    lines = []
    for ev in sorted(evals, key=lambda e: -e.readiness):
        mark = "?" if ev.eligible is None else ("PASS" if ev.eligible else "fail")
        bind = ev.binding
        bind_txt = f"{bind.name} {bind.ratio:.2f}x" if bind else "n/a"
        lines.append(f"{ev.ticker:<6} readiness {ev.readiness:5.2f}  {mark:<4} "
                     f"binding={bind_txt}"
                     + (f"  blocks={','.join(ev.hard_blocks)}" if ev.hard_blocks else "")
                     + (f"  unknown={','.join(ev.unknown)}" if ev.unknown else ""))
    return "\n".join(lines)
