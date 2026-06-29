"""Cross-sectional value model — a LightGBM forward-return ranker, validated.

Trains gradient-boosted trees on the as-of feature panel (:mod:`panel`) to predict
forward returns *cross-sectionally* (which names beat their peers), and — the part
that matters — validates it **walk-forward with purging** so the score is honest.

Key disciplines:

* **Cross-sectional rank features.** Name-specific features are percentile-ranked
  within each date, so the model learns *relative* cheapness/quality and is robust
  to level drift across regimes. Macro features are left raw (they're the same for
  every name on a date — a conditioning context, not a cross-sectional signal).
* **Purged walk-forward.** To predict date D (label spans D..D+H), the model trains
  only on rows whose label window ended on or before D (``as_of <= D - H``), so an
  overlapping forward-return window can never leak.
* **Honest metrics.** Rank IC (Spearman of prediction vs realized return, per date),
  its t-stat, top-minus-bottom quantile spread, and a single-factor (earnings-yield)
  **baseline** — because the model has to beat just buying cheap, not just beat zero.

LightGBM is a CPU/tree job; it barely uses a GPU (the right tool here regardless).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import panel

NON_FEATURES = {"ticker", "as_of", "price", "market_cap_b", "fwd_return"}
MACRO_FEATURES = ["rate_10y", "rate_3m", "term_spread", "rate_10y_chg_3m"]
BASELINE_FEATURE = "earnings_yield"

_LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, random_state=0, n_jobs=-1, verbosity=-1)


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in panel.FEATURE_COLUMNS if c not in NON_FEATURES and c in df.columns]


def cross_sectional_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in feature_cols(df) if c not in MACRO_FEATURES]


def prepare(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Drop unlabeled rows and rank-transform the cross-sectional features per date."""
    df = panel_df.copy()
    df = df.dropna(subset=["fwd_return"])
    df["as_of"] = df["as_of"].astype(str)
    for col in cross_sectional_cols(df):
        df[col] = df.groupby("as_of")[col].rank(pct=True)
    return df.reset_index(drop=True)


def _fit(train: pd.DataFrame, features: list[str], params: dict | None):
    from lightgbm import LGBMRegressor  # lazy: keep module importable without lightgbm
    model = LGBMRegressor(**(params or _LGBM_PARAMS))
    model.fit(train[features], train["fwd_return"])
    return model


def walk_forward(
    panel_df: pd.DataFrame,
    *,
    horizon_days: int = 126,
    min_train_rows: int = 80,
    min_names: int = 5,
    params: dict | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Purged walk-forward: for each date, train on non-overlapping history, predict.

    Returns a tidy frame of out-of-sample predictions:
    ``[as_of, ticker, pred, fwd_return, baseline]``.
    """
    df = prepare(panel_df)
    features = feature_cols(df)
    h = pd.Timedelta(days=horizon_days)
    as_of_ts = pd.to_datetime(df["as_of"])
    dates = sorted(df["as_of"].unique())

    out = []
    for d in dates:
        dt = pd.Timestamp(d)
        train = df[as_of_ts <= dt - h]          # purge: labels fully realized before d
        test = df[df["as_of"] == d]
        if len(train) < min_train_rows or len(test) < min_names:
            continue
        model = _fit(train, features, params)
        pred = model.predict(test[features])
        out.append(pd.DataFrame({
            "as_of": d, "ticker": test["ticker"].to_numpy(), "pred": pred,
            "fwd_return": test["fwd_return"].to_numpy(),
            "baseline": test[BASELINE_FEATURE].to_numpy() if BASELINE_FEATURE in test else np.nan,
        }))
        if verbose:
            print(f"  {d}: trained on {len(train)} rows, predicted {len(test)} names")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["as_of", "ticker", "pred", "fwd_return", "baseline"])


def _ic(a, b) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


@dataclass
class ValueScore:
    n_dates: int
    n_obs: int
    mean_ic: float
    ic_std: float
    ic_tstat: float
    hit_rate: float           # fraction of dates with IC > 0
    mean_spread: float        # top-half minus bottom-half realized return
    baseline_mean_ic: float   # IC of just ranking by earnings yield


def evaluate(preds: pd.DataFrame, *, min_names: int = 5) -> ValueScore:
    """Rank-IC, t-stat, quantile spread, and the single-factor baseline."""
    ics, spreads, base_ics = [], [], []
    for _, g in preds.groupby("as_of"):
        if len(g) < min_names:
            continue
        ics.append(_ic(g["pred"], g["fwd_return"]))
        base_ics.append(_ic(g["baseline"], g["fwd_return"]))
        med = g["pred"].median()
        top = g[g["pred"] >= med]["fwd_return"].mean()
        bot = g[g["pred"] < med]["fwd_return"].mean()
        spreads.append(top - bot)

    ics = np.array([x for x in ics if np.isfinite(x)])
    base = np.array([x for x in base_ics if np.isfinite(x)])
    spreads = np.array([x for x in spreads if np.isfinite(x)])
    if ics.size == 0:
        return ValueScore(0, len(preds), *( [float("nan")] * 5 ), float("nan"))
    mean_ic, ic_std = float(ics.mean()), float(ics.std(ddof=1) if ics.size > 1 else float("nan"))
    tstat = (mean_ic / ic_std * np.sqrt(ics.size)) if ic_std and np.isfinite(ic_std) else float("nan")
    return ValueScore(
        n_dates=int(ics.size), n_obs=int(len(preds)), mean_ic=mean_ic, ic_std=ic_std,
        ic_tstat=float(tstat), hit_rate=float(np.mean(ics > 0)),
        mean_spread=float(spreads.mean()) if spreads.size else float("nan"),
        baseline_mean_ic=float(base.mean()) if base.size else float("nan"),
    )


def train_full(panel_df: pd.DataFrame, *, params: dict | None = None):
    """Fit a final model on all labeled rows (for live ranking). Returns (model, features)."""
    df = prepare(panel_df)
    features = feature_cols(df)
    return _fit(df, features, params), features


def render_text(score: ValueScore, *, horizon_days: int) -> str:
    if not score.n_dates:
        return "\nValue model: not enough data to validate."
    edge = score.mean_ic - score.baseline_mean_ic
    verdict = ("beats" if edge > 0.01 else "ties" if edge > -0.01 else "loses to")
    return "\n".join([
        f"\nValue model — purged walk-forward ({score.n_dates} dates, {score.n_obs} obs, "
        f"{horizon_days}d forward return)",
        f"  Rank IC:    mean {score.mean_ic:+.4f}   std {score.ic_std:.4f}   "
        f"t-stat {score.ic_tstat:+.2f}   hit-rate {score.hit_rate:.0%}",
        f"  Quantile spread (top-half minus bottom-half): {score.mean_spread:+.2%} per {horizon_days}d",
        f"  Baseline (earnings-yield only) IC: {score.baseline_mean_ic:+.4f}   "
        f"-> model {verdict} the baseline by {edge:+.4f} IC",
        "  Note: IC ~0.03-0.05 with t>2 is a respectable cross-sectional signal; "
        "near-zero or t<2 means no reliable edge.",
    ])
