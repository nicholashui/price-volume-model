"""
Direction models: +1 UP, -1 DOWN, 0 abstain.

Scoring: correct bet +1, wrong bet -1, abstain 0.
Objective: maximize cumulative score (hits − misses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    name: str
    preds: pd.Series  # values in {-1, 0, +1}
    params: dict = field(default_factory=dict)
    score: float = 0.0
    n_bets: int = 0
    n_correct: int = 0
    n_wrong: int = 0
    hit_rate: float = 0.0  # among bets only


def score_predictions(preds: pd.Series, actual: pd.Series) -> dict:
    a = actual.reindex(preds.index)
    p = preds.astype(float)
    mask = p != 0
    if mask.sum() == 0:
        return {
            "score": 0.0,
            "n_bets": 0,
            "n_correct": 0,
            "n_wrong": 0,
            "hit_rate": 0.0,
            "coverage": 0.0,
            "cum": p * 0.0,
        }
    hits = (p[mask] == a[mask]).astype(int)
    # +1 correct, -1 wrong
    step = np.where(p == 0, 0, np.where(p == a, 1, -1)).astype(float)
    step = pd.Series(step, index=preds.index)
    n_bets = int(mask.sum())
    n_correct = int((p[mask] == a[mask]).sum())
    n_wrong = n_bets - n_correct
    return {
        "score": float(step.sum()),
        "n_bets": n_bets,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "hit_rate": n_correct / n_bets if n_bets else 0.0,
        "coverage": n_bets / len(p),
        "cum": step.cumsum(),
    }


def decide_from_edge(edge: np.ndarray, thr: float, min_strength: float = 0.0) -> np.ndarray:
    """edge > thr → +1, edge < -thr → -1, else 0. Optional |edge|>=min_strength."""
    out = np.zeros(len(edge), dtype=int)
    e = np.asarray(edge, dtype=float)
    up = (e >= thr) & (np.abs(e) >= min_strength)
    dn = (e <= -thr) & (np.abs(e) >= min_strength)
    out[up] = 1
    out[dn] = -1
    return out


# ---------------------------------------------------------------------------
# Model 1: Intensity edge (model.md §12)
# ---------------------------------------------------------------------------
def model_intensity(
    intensity: pd.DataFrame,
    thr: float = 0.02,
    min_psum: float = 0.0,
    conf_quantile: float | None = None,
) -> pd.Series:
    edge = intensity["edge"].values.astype(float)
    psum = intensity["p_sum"].values.astype(float)
    # normalize edge by rolling mad for scale-free threshold
    edge_s = pd.Series(edge, index=intensity.index)
    scale = edge_s.rolling(288, min_periods=48).std().replace(0, np.nan)
    z = (edge_s / (scale + 1e-12)).fillna(0.0).values
    preds = decide_from_edge(z, thr)
    if min_psum > 0:
        preds = np.where(psum >= min_psum, preds, 0)
    if conf_quantile is not None:
        abs_z = np.abs(z)
        # causal rolling quantile
        q = (
            pd.Series(abs_z, index=intensity.index)
            .rolling(576, min_periods=100)
            .quantile(conf_quantile)
            .values
        )
        preds = np.where(abs_z >= q, preds, 0)
    return pd.Series(preds, index=intensity.index, name="intensity")


# ---------------------------------------------------------------------------
# Model 2: Mean reversion
# ---------------------------------------------------------------------------
def model_mean_reversion(
    feats: pd.DataFrame,
    z_col: str = "zprice_24",
    thr: float = 1.5,
    vol_filter: float | None = 1.0,
) -> pd.Series:
    z = feats[z_col].values.astype(float)
    # fade extremes
    preds = np.zeros(len(z), dtype=int)
    preds[z >= thr] = -1
    preds[z <= -thr] = 1
    if vol_filter is not None and "vol_z_12" in feats.columns:
        vz = feats["vol_z_12"].values
        # only bet when volume not exploding (noise)
        preds = np.where(vz <= vol_filter, preds, 0)
    return pd.Series(preds, index=feats.index, name="mean_reversion")


# ---------------------------------------------------------------------------
# Model 3: Momentum / trend
# ---------------------------------------------------------------------------
def model_momentum(
    feats: pd.DataFrame,
    ret_col: str = "ret_12",
    thr: float = 0.001,
    confirm_col: str = "tfi_12",
    confirm_thr: float = 0.0,
) -> pd.Series:
    r = feats[ret_col].values.astype(float)
    preds = decide_from_edge(r, thr)
    if confirm_col in feats.columns:
        cf = feats[confirm_col].values.astype(float)
        # require order-flow agreement
        agree = ((preds == 1) & (cf > confirm_thr)) | ((preds == -1) & (cf < -confirm_thr)) | (
            preds == 0
        )
        preds = np.where(agree, preds, 0)
    return pd.Series(preds, index=feats.index, name="momentum")


# ---------------------------------------------------------------------------
# Model 4: Order-flow imbalance (TFI)
# ---------------------------------------------------------------------------
def model_orderflow(
    feats: pd.DataFrame,
    col: str = "tfi_12",
    thr: float = 0.15,
) -> pd.Series:
    x = feats[col].values.astype(float)
    return pd.Series(decide_from_edge(x, thr), index=feats.index, name="orderflow")


# ---------------------------------------------------------------------------
# Model 5: RSI extremes
# ---------------------------------------------------------------------------
def model_rsi(
    feats: pd.DataFrame,
    col: str = "rsi_14",
    lo: float = 30.0,
    hi: float = 70.0,
) -> pd.Series:
    r = feats[col].values.astype(float)
    preds = np.zeros(len(r), dtype=int)
    preds[r <= lo] = 1
    preds[r >= hi] = -1
    return pd.Series(preds, index=feats.index, name="rsi")


# ---------------------------------------------------------------------------
# Model 6: Bollinger %B mean reversion
# ---------------------------------------------------------------------------
def model_bollinger(
    feats: pd.DataFrame,
    col: str = "bb_pctb_20",
    lo: float = 0.05,
    hi: float = 0.95,
) -> pd.Series:
    b = feats[col].values.astype(float)
    preds = np.zeros(len(b), dtype=int)
    preds[b <= lo] = 1
    preds[b >= hi] = -1
    return pd.Series(preds, index=feats.index, name="bollinger")


# ---------------------------------------------------------------------------
# Model 7: Volatility breakout
# ---------------------------------------------------------------------------
def model_breakout(
    feats: pd.DataFrame,
    ret_col: str = "ret_6",
    vol_col: str = "vol_z_12",
    ret_thr: float = 0.0008,
    vol_thr: float = 0.5,
) -> pd.Series:
    r = feats[ret_col].values.astype(float)
    vz = feats[vol_col].values.astype(float)
    preds = decide_from_edge(r, ret_thr)
    preds = np.where(vz >= vol_thr, preds, 0)
    return pd.Series(preds, index=feats.index, name="breakout")


# ---------------------------------------------------------------------------
# Model 8: Intensity skew
# ---------------------------------------------------------------------------
def model_intensity_skew(
    intensity: pd.DataFrame,
    thr: float = 0.2,
) -> pd.Series:
    s = intensity["intensity_skew"].values.astype(float)
    return pd.Series(decide_from_edge(s, thr), index=intensity.index, name="intensity_skew")


# ---------------------------------------------------------------------------
# Model 9: Composite z-score ensemble of many indicators
# ---------------------------------------------------------------------------
def model_indicator_vote(
    feats: pd.DataFrame,
    cols: list[str] | None = None,
    vote_thr: float = 0.25,
) -> pd.Series:
    """Each indicator votes ±1 after z-scoring; average vote thresholded."""
    if cols is None:
        cols = [
            c
            for c in feats.columns
            if c.startswith(("ret_", "zprice_", "tfi_", "roc_", "mr_signal", "macd_hist"))
        ]
        cols = [c for c in cols if c in feats.columns][:40]
    X = feats[cols].astype(float)
    # rolling z
    z = (X - X.rolling(288, min_periods=50).mean()) / (
        X.rolling(288, min_periods=50).std() + 1e-12
    )
    # map: for mean-reversion cols invert
    votes = z.copy()
    for c in votes.columns:
        if c.startswith("zprice_") or c.startswith("mr_") or "rsi" in c:
            votes[c] = -votes[c]
    # sign of z as vote, strength weighted
    mean_vote = votes.clip(-3, 3).mean(axis=1) / 3.0
    preds = decide_from_edge(mean_vote.values, vote_thr)
    return pd.Series(preds, index=feats.index, name="indicator_vote")


# ---------------------------------------------------------------------------
# Walk-forward ML models
# ---------------------------------------------------------------------------
def _walk_forward_clf(
    X: pd.DataFrame,
    y: pd.Series,
    clf_factory,
    train_bars: int = 2000,
    retrain_every: int = 500,
    proba_thr: float = 0.55,
    min_train: int = 500,
) -> pd.Series:
    """
    Sliding walk-forward with batched prediction between retrains.
    Features at t predict y[t] = sign(r_{t→t+1}). Train only on indices < t.
    """
    idx = X.index
    n = len(idx)
    preds = np.zeros(n, dtype=int)
    feature_cols = list(X.columns)
    Xv = X[feature_cols].replace([np.inf, -np.inf], np.nan)
    yv = y.reindex(idx)
    Xmat = Xv.values.astype(float)
    yarr = yv.values.astype(float)
    valid_row = np.isfinite(Xmat).all(axis=1) & np.isin(yarr, [-1.0, 1.0])

    start_i = max(train_bars, min_train)
    t = start_i
    while t < n:
        lo = max(0, t - train_bars)
        # training mask: rows in [lo, t)
        tr_idx = np.where(valid_row[lo:t])[0] + lo
        if len(tr_idx) < min_train:
            t += retrain_every
            continue
        tr_y = yarr[tr_idx]
        if len(np.unique(tr_y)) < 2:
            t += retrain_every
            continue
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xmat[tr_idx])
        model = clf_factory()
        model.fit(Xs, tr_y)
        classes = model.classes_ if hasattr(model, "classes_") else np.array([-1, 1])

        # predict batch [t, t+retrain_every)
        t_end = min(n, t + retrain_every)
        batch_idx = np.arange(t, t_end)
        ok = valid_row[batch_idx] if t_end > t else np.array([], dtype=bool)
        # for prediction we only need finite features (label may be unknown at live time)
        finite = np.isfinite(Xmat[batch_idx]).all(axis=1) if t_end > t else np.array([], dtype=bool)
        use = batch_idx[finite]
        if len(use) == 0:
            t = t_end
            continue
        Xs_b = scaler.transform(Xmat[use])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xs_b)
            class_to_col = {int(c): j for j, c in enumerate(classes)}
            j_up = class_to_col.get(1)
            j_dn = class_to_col.get(-1)
            for k, irow in enumerate(use):
                p_up = proba[k, j_up] if j_up is not None else 0.0
                p_dn = proba[k, j_dn] if j_dn is not None else 0.0
                if p_up >= proba_thr and p_up > p_dn:
                    preds[irow] = 1
                elif p_dn >= proba_thr and p_dn > p_up:
                    preds[irow] = -1
        else:
            pred_b = model.predict(Xs_b).astype(int)
            preds[use] = pred_b
        t = t_end
    return pd.Series(preds, index=idx)


def model_logistic_wf(
    feats: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    proba_thr: float = 0.55,
    train_bars: int = 3000,
    retrain_every: int = 500,
) -> pd.Series:
    cols = [c for c in feature_cols if c in feats.columns]
    X = feats[cols].copy()

    def factory():
        return LogisticRegression(max_iter=200, C=0.5, solver="lbfgs")

    s = _walk_forward_clf(
        X, y, factory, train_bars=train_bars, retrain_every=retrain_every, proba_thr=proba_thr
    )
    s.name = "logistic_wf"
    return s


def model_gbm_wf(
    feats: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    proba_thr: float = 0.55,
    train_bars: int = 3000,
    retrain_every: int = 750,
) -> pd.Series:
    cols = [c for c in feature_cols if c in feats.columns][:60]
    X = feats[cols].copy()

    def factory():
        return GradientBoostingClassifier(
            n_estimators=40,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42,
        )

    s = _walk_forward_clf(
        X, y, factory, train_bars=train_bars, retrain_every=retrain_every, proba_thr=proba_thr
    )
    s.name = "gbm_wf"
    return s


def model_rf_wf(
    feats: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    proba_thr: float = 0.55,
    train_bars: int = 3000,
    retrain_every: int = 750,
) -> pd.Series:
    cols = [c for c in feature_cols if c in feats.columns][:60]
    X = feats[cols].copy()

    def factory():
        return RandomForestClassifier(
            n_estimators=60,
            max_depth=6,
            min_samples_leaf=40,
            random_state=42,
            n_jobs=-1,
        )

    s = _walk_forward_clf(
        X, y, factory, train_bars=train_bars, retrain_every=retrain_every, proba_thr=proba_thr
    )
    s.name = "rf_wf"
    return s


# ---------------------------------------------------------------------------
# Ensemble 2-of-3
# ---------------------------------------------------------------------------
def ensemble_2of3(p1: pd.Series, p2: pd.Series, p3: pd.Series) -> pd.Series:
    """Bet only when at least 2 models share the same non-zero decision."""
    df = pd.concat([p1, p2, p3], axis=1).fillna(0).astype(int)
    out = np.zeros(len(df), dtype=int)
    arr = df.values
    for i in range(len(df)):
        votes = arr[i]
        n_up = int(np.sum(votes == 1))
        n_dn = int(np.sum(votes == -1))
        if n_up >= 2:
            out[i] = 1
        elif n_dn >= 2:
            out[i] = -1
    return pd.Series(out, index=df.index, name="ensemble_2of3")


def tune_threshold_model(
    edge: pd.Series,
    actual: pd.Series,
    thr_grid: list[float],
    min_bets: int = 200,
) -> tuple[float, dict]:
    """Pick threshold maximizing score subject to min_bets."""
    best_thr, best = thr_grid[0], {"score": -1e18}
    for thr in thr_grid:
        preds = pd.Series(decide_from_edge(edge.values, thr), index=edge.index)
        sc = score_predictions(preds, actual)
        if sc["n_bets"] < min_bets:
            continue
        # primary: score; secondary: hit_rate
        key = (sc["score"], sc["hit_rate"])
        bkey = (best.get("score", -1e18), best.get("hit_rate", 0))
        if key > bkey:
            best = sc
            best_thr = thr
            best["thr"] = thr
    return best_thr, best
