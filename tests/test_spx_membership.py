"""Tests for membership parsing, point-in-time reconstruction and labels — no network."""

import datetime as dt

import pandas as pd
import pytest

from stock_research.spx import membership, panel

CONSTITUENTS_HTML = """
<table id="constituents">
  <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
  <tr><td>AAA</td><td>Alpha Corp</td><td>Industrials</td></tr>
  <tr><td>BBB<sup>[1]</sup></td><td>Beta Inc</td><td>Health Care</td></tr>
  <tr><td>CCC</td><td>Gamma Ltd</td><td>Financials</td></tr>
</table>
"""

# Two changes share one date via rowspan — the case that silently misdates rows
# if rowspan isn't expanded.
CHANGES_HTML = """
<table id="changes">
  <tr><th rowspan="2">Date</th><th colspan="2">Added</th>
      <th colspan="2">Removed</th><th rowspan="2">Reason</th></tr>
  <tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
  <tr><td rowspan="2">June 24, 2024</td>
      <td>AAA</td><td>Alpha Corp</td><td>XXX</td><td>Old Co</td><td>Market cap change</td></tr>
  <tr><td>BBB</td><td>Beta Inc</td><td>YYY</td><td>Older Co</td><td>Acquired</td></tr>
  <tr><td>March 1, 2023</td>
      <td>CCC</td><td>Gamma Ltd</td><td>ZZZ</td><td>Ancient Co</td><td>Bankruptcy</td></tr>
</table>
"""


# --- parsing ---------------------------------------------------------------- #

def test_parse_constituents_strips_footnotes():
    df = membership.parse_constituents(CONSTITUENTS_HTML)
    assert list(df["ticker"]) == ["AAA", "BBB", "CCC"]
    assert df.loc[df["ticker"] == "BBB", "security"].iloc[0] == "Beta Inc"


def test_parse_changes_expands_rowspan_dates():
    """Both June 24 rows must carry that date, not inherit blanks."""
    df = membership.parse_changes(CHANGES_HTML)
    june = df[df["date"] == dt.date(2024, 6, 24)]
    assert set(june["ticker"]) == {"AAA", "BBB", "XXX", "YYY"}


def test_parse_changes_splits_swaps_into_add_and_drop():
    df = membership.parse_changes(CHANGES_HTML)
    row = df[(df["ticker"] == "AAA") & (df["action"] == "add")].iloc[0]
    assert row["counterparty"] == "XXX"
    assert row["reason"] == "Market cap change"
    assert (df[df["ticker"] == "XXX"]["action"] == "drop").all()


def test_parse_changes_records_six_events():
    df = membership.parse_changes(CHANGES_HTML)
    assert len(df) == 6                       # 3 swaps x (1 add + 1 drop)
    assert set(df["action"]) == {"add", "drop"}


def test_normalize_ticker_handles_empty_cells():
    assert membership.normalize_ticker("—") == ""
    assert membership.normalize_ticker("") == ""
    assert membership.normalize_ticker("BRK.B ") == "BRK.B"


def test_parse_date_formats():
    assert membership.parse_date("June 24, 2024") == dt.date(2024, 6, 24)
    assert membership.parse_date("2024-06-24") == dt.date(2024, 6, 24)
    assert membership.parse_date("not a date") is None


def test_missing_table_returns_empty_frames():
    assert membership.parse_constituents("<p>nothing</p>").empty
    assert membership.parse_changes("<p>nothing</p>").empty


# --- point-in-time reconstruction ------------------------------------------- #

@pytest.fixture
def changes():
    return membership.parse_changes(CHANGES_HTML)


def test_membership_today_is_unchanged(changes):
    recon = membership.members_asof({"AAA", "BBB", "CCC"}, changes, dt.date(2026, 1, 1))
    assert recon.members == {"AAA", "BBB", "CCC"}
    assert recon.warnings == []


def test_rewinding_past_a_change_undoes_it(changes):
    """Before June 2024, AAA/BBB weren't in yet and XXX/YYY still were."""
    recon = membership.members_asof({"AAA", "BBB", "CCC"}, changes, dt.date(2024, 1, 1))
    assert recon.members == {"CCC", "XXX", "YYY"}


def test_rewinding_past_every_change(changes):
    recon = membership.members_asof({"AAA", "BBB", "CCC"}, changes, dt.date(2022, 1, 1))
    assert recon.members == {"XXX", "YYY", "ZZZ"}


def test_change_on_the_asof_date_is_not_undone(changes):
    """A change dated exactly as-of has already happened."""
    recon = membership.members_asof({"AAA", "BBB", "CCC"}, changes, dt.date(2024, 6, 24))
    assert "AAA" in recon.members
    assert "XXX" not in recon.members


def test_inconsistent_change_table_is_reported_not_swallowed():
    """Undoing an add for a non-member means Wikipedia and the list disagree."""
    changes = pd.DataFrame(
        [{"date": dt.date(2025, 1, 1), "ticker": "GHOST", "action": "add",
          "counterparty": "", "reason": ""}],
        columns=membership.CHANGE_COLUMNS)
    recon = membership.members_asof({"AAA"}, changes, dt.date(2024, 1, 1))
    assert recon.members == {"AAA"}
    assert any("GHOST" in w for w in recon.warnings)


# --- labels ----------------------------------------------------------------- #

def test_label_events_finds_adds_inside_the_horizon(changes):
    labels = membership.label_events(changes, dt.date(2024, 1, 1), 12, "add")
    assert labels == {"AAA": dt.date(2024, 6, 24), "BBB": dt.date(2024, 6, 24)}


def test_label_events_excludes_events_outside_the_horizon(changes):
    assert membership.label_events(changes, dt.date(2024, 1, 1), 3, "add") == {}


def test_label_events_excludes_the_asof_date_itself(changes):
    """You can't forecast a change already announced on the evaluation date."""
    assert "AAA" not in membership.label_events(changes, dt.date(2024, 6, 24), 12, "add")


def test_label_events_finds_drops(changes):
    labels = membership.label_events(changes, dt.date(2022, 6, 1), 12, "drop")
    assert labels == {"ZZZ": dt.date(2023, 3, 1)}


# --- panel feature helpers -------------------------------------------------- #

def _ohlcv(days: int, volume: float = 1e6, close: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=days)
    return pd.DataFrame({"Close": [close] * days, "Volume": [volume] * days}, index=idx)


def test_monthly_volumes_excludes_the_partial_current_month():
    hist = _ohlcv(200)
    as_of = hist.index[-1]
    vols = panel.monthly_volumes(hist, as_of, n=6)
    assert len(vols) == 6
    # Every returned month is complete, so none can be the as-of month.
    by_month = hist.groupby([hist.index.year, hist.index.month])["Volume"].sum()
    assert vols[-1] != by_month.iloc[-1] or by_month.index[-1] != (as_of.year, as_of.month)


def test_monthly_volumes_short_history_returns_what_exists():
    assert len(panel.monthly_volumes(_ohlcv(40), pd.Timestamp("2024-02-26"), n=6)) <= 6


def test_annual_dollar_volume_scales_a_short_window():
    """A half-year of history is annualized rather than reported as a half-year."""
    hist = _ohlcv(140, volume=1e6, close=100.0)
    got = panel.annual_dollar_volume(hist, hist.index[-1])
    assert got == pytest.approx(100.0 * 1e6 * 252, rel=1e-9)


def test_annual_dollar_volume_refuses_a_stub_history():
    assert panel.annual_dollar_volume(_ohlcv(30), pd.Timestamp("2024-02-09")) is None


def test_turnover_stands_in_when_float_is_unknown():
    """Historical rows have no float, so the liquidity gate can't resolve — but
    turnover (dollar volume / unadjusted cap) must still be there for the model."""
    hist = _ohlcv(300, volume=1e6, close=100.0)
    fundamentals = pd.DataFrame(
        [{"concept": "shares_diluted", "tag": "WeightedAverageNumberOfDilutedSharesOutstanding",
          "start": None, "end": "2024-03-31", "val": 1e9, "filed": "2024-04-20",
          "form": "10-Q", "fy": 2024, "fp": "Q1", "frame": None}])
    fields = panel.candidate_fields(fundamentals, hist, hist.index[-1])
    assert fields["market_cap"] == pytest.approx(100.0 * 1e9)
    assert fields["turnover"] == pytest.approx(100.0 * 1e6 * 252 / (100.0 * 1e9))


def test_turnover_is_none_without_a_market_cap():
    fields = panel.candidate_fields(None, _ohlcv(300), pd.Timestamp("2025-01-02"))
    assert fields["market_cap"] is None
    assert fields["turnover"] is None


def test_candidate_fields_returns_none_before_listing():
    hist = _ohlcv(60)
    assert panel.candidate_fields(None, hist, pd.Timestamp("2020-01-01")) is None


def test_candidate_fields_uses_only_prices_up_to_asof():
    hist = _ohlcv(300, close=100.0)
    hist.iloc[-50:, hist.columns.get_loc("Close")] = 999.0     # future spike
    as_of = hist.index[-51]
    fields = panel.candidate_fields(None, hist, as_of)
    assert fields["price"] == 100.0
