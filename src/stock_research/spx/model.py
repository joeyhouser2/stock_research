"""Add/drop classifier over the eligibility panel — and whether it beats the rule.

The gates in :mod:`~stock_research.spx.rules` tell you who is *eligible*. They
don't tell you who the committee will actually pick, and in any given month far
more names are eligible than are added. This module asks whether a model over the
gate distances predicts the pick better than simply ranking by ``readiness``.

Three disciplines carried over from :mod:`stock_research.valuemodel`:

* **Purged walk-forward.** A label at date D spans (D, D+H months]. To predict D
  the model trains only on rows whose label window closed on or before D
  (``as_of <= D - H``), so an overlapping window can never leak backwards.
* **Raw, not rank-transformed, features.** The gate ratios are already normalized
  by the threshold in force on their own date, so they're comparable across time
  *and* they carry the absolute "am I over the line yet" information that
  cross-sectional ranking would destroy. That absolute distance is the mechanism.
* **A baseline that must be beaten.** Ranking by ``readiness`` is free. Every
  metric is reported against it, because a model that ties the rule is not worth
  running.

The metric that matters operationally is **precision@k**: if you take the top k
names each month, what fraction are actually added inside the horizon? AUC on a
3%-positive problem flatters everything.

Trajectory features (how fast a name is closing on the threshold, and whether it
has held above it) are derived here rather than in the panel, since they're pure
functions of the panel's own history — no new look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Gate distances that exist for every row and carry real information.
BASE_FEATURES = [
    "ratio_market_cap", "ratio_float_market_cap", "ratio_iwf",
    "ratio_liquidity_ratio", "ratio_monthly_volume", "turnover", "readiness",
]
# Features whose trajectory matters: closing on the threshold beats sitting still.
TRAJECTORY_FEATURES = ["ratio_market_cap", "ratio_monthly_volume", "turnover"]
TRAJECTORY_LAGS = (3, 12)
ROLLING_WINDOW = 12

LABELS = {"add": "label_add", "drop": "label_drop"}

# The free rule the model has to beat: size relative to the index threshold in
# force on that date. Bigger = more likely to be added, smaller = more likely to
# be dropped, hence the sign flip.
#
# Note this is NOT `readiness`, which is a much weaker ranker and would flatter
# the model. Readiness answers "is this name eligible yet" — it clips at 1.0, so
# every candidate comfortably over the line ties at the top, and it collapses to
# 0.0 on a hard block, so ~14% of members tie at the bottom. Both ties destroy
# exactly the ordering information a ranking metric needs. It is the right
# eligibility score and the wrong baseline.
BASELINE_FEATURE = "ratio_market_cap"
BASELINE_SIGN = {"add": 1.0, "drop": -1.0}
BASELINE_LABEL = {"add": "cap/threshold", "drop": "-cap/threshold"}

_LGBM_PARAMS = dict(n_estimators=250, learning_rate=0.03, num_leaves=15,
                    min_child_samples=30, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, reg_lambda=1.0, random_state=0,
                    n_jobs=-1, verbosity=-1)


# --------------------------------------------------------------------------- #
# Feature construction.
# --------------------------------------------------------------------------- #
def _date_grid(dates) -> tuple[pd.DatetimeIndex, int]:
    """A regular date index spanning the panel, plus its step in months.

    Reindexing onto this grid is what makes the lags calendar-exact. A per-ticker
    gap already shows up as a NaN cell in the pivot, but a date where *no* ticker
    has data would vanish from the index entirely and a positional shift would
    silently hop over it.
    """
    uniq = pd.DatetimeIndex(sorted(set(pd.to_datetime(dates))))
    if len(uniq) < 2:
        return uniq, 1
    months = uniq.year * 12 + uniq.month
    step = int(np.median(np.diff(months))) or 1
    grid = pd.date_range(uniq.min(), uniq.max(), freq=pd.DateOffset(months=step))
    return grid.union(uniq), step


def _wide(df: pd.DataFrame, col: str, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """ticker x date matrix for one column on the regular grid, gaps left as NaN."""
    wide = df.pivot_table(index="as_of", columns="ticker", values=col, aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    return wide.reindex(grid)


def _melt(wide: pd.DataFrame, name: str) -> pd.DataFrame:
    """Long form of a ticker x date matrix, keeping the NaN cells.

    ``melt`` rather than ``stack``: stack's NA handling changed across pandas
    versions, and dropping the NaN cells here would quietly turn "no data" into
    "no row" and misalign the merge.
    """
    flat = wide.reset_index()
    return flat.melt(id_vars=[flat.columns[0]], var_name="ticker", value_name=name) \
               .rename(columns={flat.columns[0]: "as_of"})


def add_trajectory_features(df: pd.DataFrame, *, lags=TRAJECTORY_LAGS,
                            window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Attach change-over-N-months and rolling min/max columns per ticker.

    All backward-looking: at date D every added column uses only rows at or before
    D for that ticker.
    """
    out = df.copy()
    out["as_of"] = pd.to_datetime(out["as_of"])
    grid, step = _date_grid(out["as_of"])
    for col in TRAJECTORY_FEATURES:
        if col not in out.columns:
            continue
        wide = _wide(out, col, grid)
        pieces = []
        for lag in lags:
            periods = round(lag / step)
            if periods < 1:
                continue                # lag finer than the panel's own frequency
            pieces.append(_melt(wide - wide.shift(periods), f"{col}_chg_{lag}m"))
        # Held-above and worst-case over the window: S&P favours sustained size,
        # not a name that touched the threshold once on a spike.
        periods = max(round(window / step), 2)
        min_periods = max(round(periods / 4), 2)
        pieces.append(_melt(wide.rolling(periods, min_periods=min_periods).min(),
                            f"{col}_min_{window}m"))
        pieces.append(_melt(wide.rolling(periods, min_periods=min_periods).mean(),
                            f"{col}_mean_{window}m"))
        for piece in pieces:
            out = out.merge(piece, on=["as_of", "ticker"], how="left")
    return out


def feature_cols(df: pd.DataFrame) -> list[str]:
    base = [c for c in BASE_FEATURES if c in df.columns]
    derived = [c for c in df.columns
               if any(c.startswith(f"{f}_chg_") or c.startswith(f"{f}_min_")
                      or c.startswith(f"{f}_mean_") for f in TRAJECTORY_FEATURES)]
    passes = [c for c in df.columns if c.startswith("pass_")]
    return base + sorted(derived) + sorted(passes)


def prepare(panel_df: pd.DataFrame, *, side: str = "add") -> pd.DataFrame:
    """Filter to the relevant population, coerce types, and derive trajectories.

    ``side="add"`` keeps non-members (a member can't be added); ``side="drop"``
    keeps members. Boolean gate columns become 0/1/NaN so a missing gate stays
    missing rather than collapsing to False.
    """
    if side not in LABELS:
        raise ValueError(f"side must be one of {sorted(LABELS)}, got {side!r}")
    df = panel_df.copy()
    df["as_of"] = pd.to_datetime(df["as_of"])
    was_member = df["was_member"].astype(str).str.lower().isin(("true", "1"))
    df = df[~was_member] if side == "add" else df[was_member]
    if df.empty:
        return df

    for col in [c for c in df.columns if c.startswith("pass_")]:
        df[col] = df[col].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    df = add_trajectory_features(df)
    df["label"] = df[LABELS[side]].astype(int)
    return df.sort_values(["as_of", "ticker"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Walk-forward.
# --------------------------------------------------------------------------- #
def baseline_score(df: pd.DataFrame, side: str) -> np.ndarray:
    """The free rule's score, oriented so higher always means "more likely".

    Market cap over the threshold for additions; negated for deletions, since the
    member drifting furthest below the line is the one at risk.
    """
    if BASELINE_FEATURE not in df:
        return np.full(len(df), np.nan)
    return BASELINE_SIGN[side] * df[BASELINE_FEATURE].to_numpy(dtype=float)


def _fit(train: pd.DataFrame, features: list[str], params: dict | None):
    """LightGBM if present, else a standardized logistic regression."""
    pos = int(train["label"].sum())
    neg = len(train) - pos
    try:
        from lightgbm import LGBMClassifier
        p = dict(_LGBM_PARAMS)
        p.update(params or {})
        p["scale_pos_weight"] = (neg / pos) if pos else 1.0
        model = LGBMClassifier(**p)
        model.fit(train[features], train["label"])
        return model
    except ImportError:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"))
        model.fit(train[features], train["label"])
        return model


def walk_forward(panel_df: pd.DataFrame, *, side: str = "add",
                 horizon_months: int = 12, min_train_rows: int = 500,
                 min_train_positives: int = 15, min_names: int = 20,
                 params: dict | None = None, verbose: bool = False) -> pd.DataFrame:
    """Purged walk-forward predictions: ``[as_of, ticker, pred, label, baseline]``."""
    df = prepare(panel_df, side=side)
    if df.empty:
        return pd.DataFrame(columns=["as_of", "ticker", "pred", "label", "baseline"])
    features = feature_cols(df)
    offset = pd.DateOffset(months=horizon_months)

    out = []
    for d in sorted(df["as_of"].unique()):
        d = pd.Timestamp(d)
        train = df[df["as_of"] <= d - offset]        # labels fully resolved before d
        test = df[df["as_of"] == d]
        if (len(train) < min_train_rows or int(train["label"].sum()) < min_train_positives
                or len(test) < min_names):
            continue
        model = _fit(train, features, params)
        pred = model.predict_proba(test[features])[:, 1]
        out.append(pd.DataFrame({
            "as_of": d, "ticker": test["ticker"].to_numpy(), "pred": pred,
            "label": test["label"].to_numpy(),
            "baseline": baseline_score(test, side),
        }))
        if verbose:
            print(f"  {d.date()}: train {len(train)} rows / "
                  f"{int(train['label'].sum())} pos -> predict {len(test)}")
    if not out:
        return pd.DataFrame(columns=["as_of", "ticker", "pred", "label", "baseline"])
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# Scoring.
# --------------------------------------------------------------------------- #
def _auc(score, label) -> float:
    """ROC AUC via the rank identity; NaN when a date has one class only."""
    label = np.asarray(label, dtype=float)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    label, score = label[ok], score[ok]
    pos, neg = label.sum(), (1 - label).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    return float((ranks[label == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _precision_at_k(score, label, k: int) -> float:
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=float)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    if score.size == 0:
        return float("nan")
    k = min(k, score.size)
    top = np.argsort(-score, kind="stable")[:k]
    return float(label[top].sum() / k)


def _recall_at_k(score, label, k: int) -> float:
    label_arr = np.asarray(label, dtype=float)
    if label_arr.sum() == 0:
        return float("nan")
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    score, label_arr = score[ok], label_arr[ok]
    if score.size == 0 or label_arr.sum() == 0:
        return float("nan")
    k = min(k, score.size)
    top = np.argsort(-score, kind="stable")[:k]
    return float(label_arr[top].sum() / label_arr.sum())


@dataclass
class ClassifierScore:
    side: str
    n_dates: int
    n_obs: int
    base_rate: float
    auc: float
    precision_at_k: float
    recall_at_k: float
    k: int
    baseline_auc: float
    baseline_precision_at_k: float
    baseline_recall_at_k: float


def evaluate(preds: pd.DataFrame, *, side: str = "add", k: int = 10) -> ClassifierScore:
    """Per-date AUC and precision/recall@k, for the model and the readiness rule."""
    aucs, precs, recs = [], [], []
    b_aucs, b_precs, b_recs = [], [], []
    n_dates = 0
    for _, g in preds.groupby("as_of"):
        if g["label"].nunique() < 2:
            continue
        n_dates += 1
        aucs.append(_auc(g["pred"], g["label"]))
        precs.append(_precision_at_k(g["pred"], g["label"], k))
        recs.append(_recall_at_k(g["pred"], g["label"], k))
        b_aucs.append(_auc(g["baseline"], g["label"]))
        b_precs.append(_precision_at_k(g["baseline"], g["label"], k))
        b_recs.append(_recall_at_k(g["baseline"], g["label"], k))

    mean = lambda xs: float(np.nanmean(xs)) if len(xs) else float("nan")  # noqa: E731
    return ClassifierScore(
        side=side, n_dates=n_dates, n_obs=len(preds),
        base_rate=float(preds["label"].mean()) if len(preds) else float("nan"),
        auc=mean(aucs), precision_at_k=mean(precs), recall_at_k=mean(recs), k=k,
        baseline_auc=mean(b_aucs), baseline_precision_at_k=mean(b_precs),
        baseline_recall_at_k=mean(b_recs),
    )


def feature_importance(panel_df: pd.DataFrame, *, side: str = "add",
                       params: dict | None = None) -> pd.DataFrame:
    """Gain importance from a model fit on everything — description, not validation."""
    df = prepare(panel_df, side=side)
    if df.empty:
        return pd.DataFrame(columns=["feature", "importance"])
    features = feature_cols(df)
    model = _fit(df, features, params)
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    imp = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def render_text(score: ClassifierScore) -> str:
    if not score.n_dates:
        return "\nIndex model: not enough labelled dates to validate."
    lift = score.precision_at_k - score.baseline_precision_at_k
    verdict = ("beats" if lift > 0.02 else "ties" if lift > -0.02 else "loses to")
    naive = score.base_rate
    rule = BASELINE_LABEL[score.side]
    return "\n".join([
        f"\nIndex {score.side} model — purged walk-forward "
        f"({score.n_dates} dates, {score.n_obs} obs, {score.base_rate:.2%} base rate)",
        f"  {'':<22}{'model':>10}{rule:>15}{'random':>10}",
        f"  {'AUC':<22}{score.auc:>10.3f}{score.baseline_auc:>15.3f}{0.5:>10.3f}",
        f"  {f'precision@{score.k}':<22}{score.precision_at_k:>10.3f}"
        f"{score.baseline_precision_at_k:>15.3f}{naive:>10.3f}",
        f"  {f'recall@{score.k}':<22}{score.recall_at_k:>10.3f}"
        f"{score.baseline_recall_at_k:>15.3f}",
        f"  -> model {verdict} the one-feature rule by {lift:+.3f} precision@{score.k}",
        "  precision@k is the operational number: of the k names you'd flag each "
        "month, that fraction actually changed membership inside the horizon.",
        f"  The baseline is size vs. the index threshold alone. It is deliberately "
        f"the strongest simple rule, not the most flattering one.",
    ])
