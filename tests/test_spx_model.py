"""Tests for the index add/drop classifier — purging, features, metrics. No network."""

import numpy as np
import pandas as pd
import pytest

from stock_research.spx import model


def _panel(n_dates: int = 40, n_tickers: int = 30, seed: int = 0) -> pd.DataFrame:
    """Synthetic monthly panel where big, fast-growing names get added."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="MS")
    rows = []
    for t in range(n_tickers):
        drift = rng.normal(0.01, 0.01)
        level = rng.uniform(0.3, 1.2)
        for i, d in enumerate(dates):
            level = max(0.05, level + drift + rng.normal(0, 0.02))
            rows.append({
                "ticker": f"T{t:02d}", "as_of": d.date().isoformat(),
                "was_member": False,
                "ratio_market_cap": level,
                "ratio_monthly_volume": 2.0 + rng.normal(0, 0.1),
                "turnover": 0.8 + rng.normal(0, 0.05),
                "readiness": min(level, 1.0),
                "pass_viability": True,
                "label_add": int(level > 1.05 and drift > 0.012),
                "label_drop": 0,
            })
    return pd.DataFrame(rows)


# --- trajectory features ---------------------------------------------------- #

def test_trajectory_change_is_exact_over_the_lag():
    df = pd.DataFrame({
        "ticker": ["A"] * 5,
        "as_of": pd.date_range("2024-01-01", periods=5, freq="MS"),
        "ratio_market_cap": [1.0, 1.1, 1.2, 1.3, 1.4],
        "ratio_monthly_volume": [2.0] * 5,
        "turnover": [0.5] * 5,
    })
    out = model.add_trajectory_features(df, lags=(3,), window=3)
    got = out.sort_values("as_of")["ratio_market_cap_chg_3m"].tolist()
    assert np.isnan(got[0]) and np.isnan(got[2])
    assert got[3] == pytest.approx(0.3)
    assert got[4] == pytest.approx(0.3)


def test_trajectory_does_not_compare_across_a_per_ticker_gap():
    """One name missing a month must give NaN, not a silent 2-month comparison."""
    df = pd.DataFrame({
        "ticker": ["A", "A", "A", "B", "B", "B", "B"],
        "as_of": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-04-01",
                                 "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
        "ratio_market_cap": [1.0, 1.1, 1.4, 2.0, 2.0, 2.0, 2.0],
        "ratio_monthly_volume": [2.0] * 7, "turnover": [0.5] * 7,
    })
    out = model.add_trajectory_features(df, lags=(1,), window=3)
    row = out[(out["ticker"] == "A") & (out["as_of"] == pd.Timestamp("2024-04-01"))].iloc[0]
    assert np.isnan(row["ratio_market_cap_chg_1m"])       # March is missing for A
    row_b = out[(out["ticker"] == "B") & (out["as_of"] == pd.Timestamp("2024-04-01"))].iloc[0]
    assert row_b["ratio_market_cap_chg_1m"] == pytest.approx(0.0)


def test_trajectory_does_not_hop_a_date_missing_for_every_ticker():
    """March absent from the whole panel must still break the 1-month lag."""
    dates = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-04-01"])
    df = pd.DataFrame({
        "ticker": ["A"] * 3, "as_of": dates,
        "ratio_market_cap": [1.0, 1.1, 1.4],
        "ratio_monthly_volume": [2.0] * 3, "turnover": [0.5] * 3,
    })
    out = model.add_trajectory_features(df, lags=(1,), window=3)
    row = out[out["as_of"] == pd.Timestamp("2024-04-01")].iloc[0]
    assert np.isnan(row["ratio_market_cap_chg_1m"])


def test_trajectory_lags_are_calendar_months_on_a_quarterly_panel():
    """A 12-month lag on a quarterly panel must span 4 rows, not 12."""
    df = pd.DataFrame({
        "ticker": ["A"] * 6,
        "as_of": pd.date_range("2023-01-01", periods=6, freq="QS"),
        "ratio_market_cap": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        "ratio_monthly_volume": [2.0] * 6, "turnover": [0.5] * 6,
    })
    out = model.add_trajectory_features(df, lags=(12,), window=12).sort_values("as_of")
    assert out["ratio_market_cap_chg_12m"].iloc[4] == pytest.approx(0.4)


def test_rolling_min_uses_only_past_values():
    df = pd.DataFrame({
        "ticker": ["A"] * 5,
        "as_of": pd.date_range("2024-01-01", periods=5, freq="MS"),
        "ratio_market_cap": [1.0, 0.5, 1.2, 1.3, 1.4],
        "ratio_monthly_volume": [2.0] * 5, "turnover": [0.5] * 5,
    })
    out = model.add_trajectory_features(df, lags=(1,), window=3).sort_values("as_of")
    # At the third date the trailing min is 0.5; the later rise must not lower it
    # retroactively, and the dip must not leak backwards into date one.
    assert out["ratio_market_cap_min_3m"].iloc[2] == pytest.approx(0.5)
    assert np.isnan(out["ratio_market_cap_min_3m"].iloc[0])


# --- prepare ---------------------------------------------------------------- #

def test_prepare_keeps_non_members_for_the_add_side():
    df = _panel(n_dates=6, n_tickers=4)
    df.loc[df["ticker"] == "T00", "was_member"] = True
    out = model.prepare(df, side="add")
    assert "T00" not in set(out["ticker"])


def test_prepare_keeps_members_for_the_drop_side():
    df = _panel(n_dates=6, n_tickers=4)
    df.loc[df["ticker"] == "T00", "was_member"] = True
    out = model.prepare(df, side="drop")
    assert set(out["ticker"]) == {"T00"}


def test_prepare_maps_pass_columns_to_numbers_keeping_missing_missing():
    """A gate we couldn't evaluate must stay NaN, not collapse into False."""
    df = _panel(n_dates=6, n_tickers=3)
    df["pass_viability"] = df["pass_viability"].astype(object)
    df.loc[0, "pass_viability"] = None
    out = model.prepare(df, side="add")
    assert set(out["pass_viability"].dropna().unique()) <= {0.0, 1.0}
    assert out["pass_viability"].isna().any()


def test_prepare_rejects_a_bad_side():
    with pytest.raises(ValueError):
        model.prepare(_panel(n_dates=4, n_tickers=3), side="sideways")


# --- purging ---------------------------------------------------------------- #

def test_walk_forward_never_trains_on_unresolved_labels(monkeypatch):
    """Training rows must have their whole label window closed before the test date."""
    seen = []
    real_fit = model._fit

    def spy(train, features, params):
        seen.append((train["as_of"].max(), len(train)))
        return real_fit(train, features, params)

    monkeypatch.setattr(model, "_fit", spy)
    df = _panel(n_dates=40, n_tickers=30)
    preds = model.walk_forward(df, side="add", horizon_months=12,
                               min_train_rows=50, min_train_positives=1, min_names=5)
    assert not preds.empty
    assert seen
    for (train_max, _), test_date in zip(seen, sorted(preds["as_of"].unique())):
        assert train_max <= pd.Timestamp(test_date) - pd.DateOffset(months=12)


def test_walk_forward_skips_dates_without_enough_history():
    df = _panel(n_dates=40, n_tickers=30)
    preds = model.walk_forward(df, side="add", horizon_months=12,
                               min_train_rows=50, min_train_positives=1, min_names=5)
    first = pd.Timestamp(min(preds["as_of"]))
    assert first >= pd.Timestamp("2021-01-01")      # needs 12 months of closed labels


def test_walk_forward_returns_one_row_per_name_per_scored_date():
    df = _panel(n_dates=40, n_tickers=30)
    preds = model.walk_forward(df, side="add", horizon_months=12,
                               min_train_rows=50, min_train_positives=1, min_names=5)
    assert not preds.duplicated(["as_of", "ticker"]).any()
    assert preds["pred"].between(0, 1).all()


# --- metrics ---------------------------------------------------------------- #

def test_auc_is_one_for_a_perfect_ranking():
    assert model._auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_auc_is_half_for_a_tied_score():
    assert model._auc([0.5] * 4, [1, 1, 0, 0]) == pytest.approx(0.5)


def test_auc_is_nan_with_one_class():
    assert np.isnan(model._auc([0.9, 0.1], [1, 1]))


def test_precision_at_k_counts_hits_in_the_top_k():
    score = [0.9, 0.8, 0.7, 0.1, 0.05]
    label = [1, 0, 1, 0, 0]
    assert model._precision_at_k(score, label, 2) == pytest.approx(0.5)
    assert model._precision_at_k(score, label, 3) == pytest.approx(2 / 3)


def test_precision_at_k_caps_k_at_the_population():
    assert model._precision_at_k([0.9, 0.1], [1, 0], 10) == pytest.approx(0.5)


def test_recall_at_k_is_the_share_of_positives_captured():
    score = [0.9, 0.8, 0.7, 0.1]
    label = [1, 0, 1, 1]
    assert model._recall_at_k(score, label, 2) == pytest.approx(1 / 3)
    assert model._recall_at_k(score, label, 3) == pytest.approx(2 / 3)


def test_recall_at_k_is_nan_without_positives():
    assert np.isnan(model._recall_at_k([0.9, 0.1], [0, 0], 2))


def test_baseline_is_size_over_threshold_and_flips_sign_for_drops():
    """A member drifting AWAY from the threshold is the one at deletion risk."""
    df = pd.DataFrame({"ratio_market_cap": [0.2, 0.9], "readiness": [0.9, 0.2]})
    assert list(model.baseline_score(df, "add")) == [0.2, 0.9]
    assert list(model.baseline_score(df, "drop")) == [-0.2, -0.9]


def test_baseline_is_not_readiness():
    """Readiness clips at 1.0 and zeroes on hard blocks, so it ties at both ends
    and is a weak ranker — using it as the baseline would flatter the model."""
    df = pd.DataFrame({"ratio_market_cap": [1.5, 2.5], "readiness": [1.0, 1.0]})
    got = model.baseline_score(df, "add")
    assert got[0] != got[1]              # the rule must separate what readiness ties


def test_baseline_is_nan_without_the_feature():
    got = model.baseline_score(pd.DataFrame({"x": [1, 2]}), "add")
    assert len(got) == 2 and np.isnan(got).all()


def test_drop_side_baseline_ranks_the_smallest_member_first():
    df = _panel(n_dates=40, n_tickers=30)
    df["was_member"] = True
    df["label_drop"] = (df["ratio_market_cap"] < 0.5).astype(int)
    preds = model.walk_forward(df, side="drop", horizon_months=12,
                               min_train_rows=50, min_train_positives=1, min_names=5)
    if not preds.empty:
        score = model.evaluate(preds, side="drop", k=5)
        assert score.baseline_auc > 0.5


def test_evaluate_reports_model_and_baseline_side_by_side():
    preds = pd.DataFrame({
        "as_of": ["2024-01-01"] * 6,
        "ticker": list("ABCDEF"),
        "pred": [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
        "label": [1, 1, 0, 0, 0, 0],
        "baseline": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],      # exactly inverted
    })
    score = model.evaluate(preds, side="add", k=2)
    assert score.precision_at_k == pytest.approx(1.0)
    assert score.baseline_precision_at_k == pytest.approx(0.0)
    assert score.auc == pytest.approx(1.0)
    assert score.baseline_auc == pytest.approx(0.0)
    assert score.base_rate == pytest.approx(2 / 6)


def test_evaluate_skips_dates_with_a_single_class():
    preds = pd.DataFrame({
        "as_of": ["2024-01-01"] * 3 + ["2024-02-01"] * 3,
        "ticker": list("ABCABC"),
        "pred": [0.9, 0.5, 0.1] * 2,
        "label": [0, 0, 0, 1, 0, 0],
        "baseline": [0.5] * 6,
    })
    assert model.evaluate(preds, side="add", k=1).n_dates == 1


def test_render_text_names_the_winner():
    score = model.ClassifierScore(
        side="add", n_dates=10, n_obs=1000, base_rate=0.03, auc=0.80,
        precision_at_k=0.30, recall_at_k=0.5, k=10, baseline_auc=0.60,
        baseline_precision_at_k=0.10, baseline_recall_at_k=0.2)
    text = model.render_text(score)
    assert "beats" in text and "precision@10" in text


def test_render_text_handles_no_validation_dates():
    score = model.ClassifierScore("add", 0, 0, float("nan"), *([float("nan")] * 3),
                                  10, *([float("nan")] * 3))
    assert "not enough" in model.render_text(score)
