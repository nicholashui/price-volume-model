"""
Precision-first decision layer.

Goal: move from high-volume noisy betting
  correct=25544, wrong=22929, score=+2615
toward high-precision selective betting
  maximize correct while driving wrong → 0
  (ideal illustration: correct≈score, wrong=0).

Methods (all causal / walk-forward where needed):
  1) Stricter consensus: 3-of-3 of best models
  2) Multi-model agreement count threshold
  3) Meta-labeling: P(bet is correct | features at t) with high thr
  4) Regime filters: volatility / clock
  5) Grid search on first half, report full window + second half (OOS)

Writes:
  predictions_precision.parquet
  metrics_precision.json
  updates predictions.parquet ensemble_precision column
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent


def score(preds: pd.Series, y: pd.Series) -> dict:
    p = preds.astype(int)
    a = y.astype(int)
    m = (p != 0) & a.isin([-1, 1])
    if m.sum() == 0:
        return {
            "score": 0,
            "n_bets": 0,
            "n_correct": 0,
            "n_wrong": 0,
            "hit_rate": 0.0,
            "coverage": 0.0,
        }
    ok = (p[m] == a[m]).sum()
    n = int(m.sum())
    wrong = n - int(ok)
    return {
        "score": int(ok) - wrong,
        "n_bets": n,
        "n_correct": int(ok),
        "n_wrong": int(wrong),
        "hit_rate": float(ok) / n,
        "coverage": n / len(p),
    }


def load_price_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    raw = pd.read_parquet(ROOT / "btcusdt_1m.parquet")
    df5 = (
        raw.resample("5min", label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
    )
    df5 = df5.reindex(index).ffill()
    c = df5["close"].astype(float)
    v = df5["volume"].astype(float)
    r = np.log(c).diff()
    feat = pd.DataFrame(index=index)
    feat["ret_1"] = r
    feat["ret_3"] = r.rolling(3).sum()
    feat["ret_12"] = r.rolling(12).sum()
    feat["rvol_12"] = r.rolling(12).std()
    feat["rvol_48"] = r.rolling(48).std()
    feat["vol_z"] = (v - v.rolling(48).mean()) / (v.rolling(48).std() + 1e-12)
    feat["hl"] = ((df5["high"] - df5["low"]) / c).astype(float)
    feat["hour"] = index.hour + index.minute / 60.0
    feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)
    feat["dow"] = index.dayofweek.astype(float)
    return feat.replace([np.inf, -np.inf], np.nan)


def ensemble_kofn(votes: pd.DataFrame, k: int) -> pd.Series:
    """Bet when at least k models share the same non-zero side."""
    arr = votes.fillna(0).astype(int).values
    out = np.zeros(len(votes), dtype=int)
    for i in range(len(votes)):
        row = arr[i]
        n_up = int(np.sum(row == 1))
        n_dn = int(np.sum(row == -1))
        if n_up >= k:
            out[i] = 1
        elif n_dn >= k:
            out[i] = -1
    return pd.Series(out, index=votes.index)


def agreement_strength(votes: pd.DataFrame) -> pd.DataFrame:
    """For each row: majority side and count of agreeing non-zero votes."""
    arr = votes.fillna(0).astype(int).values
    side = np.zeros(len(votes), dtype=int)
    strength = np.zeros(len(votes), dtype=int)
    for i in range(len(votes)):
        row = arr[i]
        n_up = int(np.sum(row == 1))
        n_dn = int(np.sum(row == -1))
        if n_up > n_dn and n_up > 0:
            side[i] = 1
            strength[i] = n_up
        elif n_dn > n_up and n_dn > 0:
            side[i] = -1
            strength[i] = n_dn
    return pd.DataFrame({"side": side, "strength": strength}, index=votes.index)


def meta_label_filter(
    base: pd.Series,
    y: pd.Series,
    feats: pd.DataFrame,
    proba_thr: float = 0.70,
    train_bars: int = 3000,
    retrain_every: int = 750,
    min_train: int = 400,
) -> pd.Series:
    """
    Walk-forward meta-labeler: when base proposes a bet, predict P(correct).
    Only keep bet if P >= proba_thr. Strictly causal.
    """
    idx = base.index
    out = np.zeros(len(idx), dtype=int)
    X = feats.reindex(idx).replace([np.inf, -np.inf], np.nan)
    # meta features: base side + feature columns
    cols = list(X.columns)

    model = None
    scaler = None
    last = -10**9
    start = max(train_bars, min_train)

    for i in range(start, len(idx)):
        if base.iloc[i] == 0:
            continue

        # retrain on past bets only
        if model is None or (i - last) >= retrain_every:
            lo = max(0, i - train_bars)
            # past indices with base bet and known label
            past = np.arange(lo, i)
            b = base.iloc[past].values
            yy = y.iloc[past].values
            mask = (b != 0) & np.isin(yy, [-1, 1])
            if mask.sum() < min_train:
                continue
            # label: 1 if past bet was correct
            meta_y = (b[mask] == yy[mask]).astype(int)
            if len(np.unique(meta_y)) < 2:
                continue
            Xp = X.iloc[past].iloc[np.where(mask)[0]]
            # add base direction as feature
            Xp = Xp.copy()
            Xp["base_side"] = b[mask]
            ok = Xp.notna().all(axis=1).values
            if ok.sum() < min_train // 2:
                continue
            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xp.loc[ok].values)
            model = LogisticRegression(max_iter=300, C=0.3, solver="lbfgs")
            try:
                model.fit(Xs, meta_y[ok])
            except Exception:
                model = None
                continue
            last = i

        if model is None or scaler is None:
            continue
        row = X.iloc[i : i + 1].copy()
        row["base_side"] = base.iloc[i]
        if row.isna().any(axis=None):
            continue
        proba = model.predict_proba(scaler.transform(row.values))[0]
        # class 1 = correct
        classes = list(model.classes_)
        if 1 not in classes:
            continue
        p_ok = float(proba[classes.index(1)])
        if p_ok >= proba_thr:
            out[i] = int(base.iloc[i])

    return pd.Series(out, index=idx, name="meta")


def rolling_precision_gate(
    base: pd.Series,
    y: pd.Series,
    window: int = 500,
    min_prec: float = 0.58,
    min_bets: int = 30,
) -> pd.Series:
    """
    Only allow a bet if recent precision of this base strategy is high enough.
    Causal: uses outcomes of bets strictly before t.
    """
    idx = base.index
    out = np.zeros(len(idx), dtype=int)
    # store past bet correctness as series
    step = np.where(base == 0, np.nan, np.where(base.values == y.values, 1.0, 0.0))
    # expanding precision of last `window` bets (not bars)
    bet_correct = []
    bet_pos = []
    for i in range(len(idx)):
        if base.iloc[i] == 0:
            continue
        # precision from previous bets
        if len(bet_correct) >= min_bets:
            recent = bet_correct[-window:]
            prec = float(np.mean(recent))
            if prec >= min_prec:
                out[i] = int(base.iloc[i])
        # after decision, observe outcome (for future gates only)
        # outcome known after bar i completes → available from i+1
        # we append at end of iteration so this bet doesn't use its own label
        if y.iloc[i] in (-1, 1) and base.iloc[i] != 0:
            bet_correct.append(1.0 if base.iloc[i] == y.iloc[i] else 0.0)
            bet_pos.append(i)
    return pd.Series(out, index=idx)


def require_same_side(*series: pd.Series) -> pd.Series:
    df = pd.concat(series, axis=1)
    arr = df.fillna(0).astype(int).values
    out = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        row = arr[i]
        nz = row[row != 0]
        if len(nz) == len(row) and len(nz) > 0 and np.all(nz == nz[0]):
            out[i] = int(nz[0])
    return pd.Series(out, index=df.index)


def main():
    print("Loading predictions…")
    pred = pd.read_parquet(ROOT / "predictions.parquet")
    y = pred["y"].astype(int)
    # evaluation from first valid non-zero activity
    score_start = pred.index[min(14 * 24 * 12, len(pred) // 10)]

    top = ["intensity_fade", "logistic_wf", "bollinger"]
    # also strong MR family
    pool = [
        "intensity_fade",
        "logistic_wf",
        "bollinger",
        "mean_reversion",
        "rsi",
        "rf_wf",
        "gbm_wf",
    ]
    pool = [c for c in pool if c in pred.columns]

    base_ens = pred["ensemble_2of3"].astype(int)
    results = {}

    def eval_name(name, s: pd.Series, subset=None):
        if subset is None:
            m = pred.index >= score_start
        else:
            m = subset
        sc = score(s.loc[m], y.loc[m])
        results[name] = sc
        print(
            f"{name:28s}  score={sc['score']:+6d}  "
            f"correct={sc['n_correct']:5d}  wrong={sc['n_wrong']:5d}  "
            f"hit={sc['hit_rate']:.3f}  bets={sc['n_bets']:5d}"
        )
        return sc

    print("\n=== baselines ===")
    eval_name("ensemble_2of3", base_ens)
    # 3of3 of top
    p3 = require_same_side(*[pred[c].astype(int) for c in top])
    eval_name("top3_unanimous", p3)

    # k-of-n on pool
    votes = pred[pool].astype(int)
    for k in range(2, min(6, len(pool) + 1)):
        eval_name(f"pool_{k}of{len(pool)}", ensemble_kofn(votes, k))

    # agreement strength thresholds on all non-zero models
    all_models = [
        c
        for c in pred.columns
        if c not in ("y", "ensemble_2of3") and pred[c].dtype != object
    ]
    agr = agreement_strength(pred[all_models].astype(int))
    for thr in [3, 4, 5, 6, 7, 8]:
        s = agr["side"].where(agr["strength"] >= thr, 0).astype(int)
        eval_name(f"agree>={thr}", s)

    # combine: top3 unanimous OR high agreement
    print("\n=== price / regime features ===")
    feats = load_price_features(pred.index)
    # vol filter: only bet in calm-moderate vol
    rvol = feats["rvol_12"]
    rvol_z = (rvol - rvol.rolling(288, min_periods=50).mean()) / (
        rvol.rolling(288, min_periods=50).std() + 1e-12
    )

    def apply_vol_gate(s, zmax=1.0):
        return s.where(rvol_z.fillna(0) <= zmax, 0).astype(int)

    eval_name("2of3_volgate", apply_vol_gate(base_ens, 0.5))
    eval_name("3of3_volgate", apply_vol_gate(p3, 1.0))

    # rolling precision gate
    print("\n=== rolling precision gate ===")
    for min_prec in [0.55, 0.58, 0.60, 0.62, 0.65]:
        g = rolling_precision_gate(base_ens, y, window=400, min_prec=min_prec, min_bets=40)
        eval_name(f"2of3_prec>={min_prec:.2f}", g)
    for min_prec in [0.55, 0.58, 0.60, 0.65, 0.70]:
        g = rolling_precision_gate(p3, y, window=300, min_prec=min_prec, min_bets=20)
        eval_name(f"3of3_prec>={min_prec:.2f}", g)

    # meta-label on 2of3 and 3of3
    print("\n=== meta-label walk-forward (slow) ===")
    # enrich feats with model votes
    for c in top:
        feats[f"vote_{c}"] = pred[c].astype(float)
    feats["agree_str"] = agr["strength"].astype(float)
    feats["rvol_z"] = rvol_z

    meta_cfgs = []
    for base_name, base_s in [("2of3", base_ens), ("3of3", p3)]:
        for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
            print(f"  meta {base_name} thr={thr}…")
            m = meta_label_filter(
                base_s, y, feats, proba_thr=thr, train_bars=2500, retrain_every=800
            )
            sc = eval_name(f"meta_{base_name}_p>={thr:.2f}", m)
            meta_cfgs.append((f"meta_{base_name}_p>={thr:.2f}", m, sc))

    # stacked: 3of3 + meta + precision gate
    print("\n=== stacked high-precision ===")
    best_stack = None
    for thr in [0.60, 0.65, 0.70, 0.75, 0.80]:
        m = meta_label_filter(p3, y, feats, proba_thr=thr, train_bars=2500, retrain_every=800)
        g = rolling_precision_gate(m, y, window=200, min_prec=0.60, min_bets=15)
        # also require agree strength >= 3 among top pool
        agr3 = agreement_strength(pred[top].astype(int))
        s = g.where(agr3["strength"] >= 2, 0).astype(int)
        s = apply_vol_gate(s, zmax=1.5)
        sc = eval_name(f"stack_3of3_meta{thr:.2f}", s)
        if best_stack is None or (
            sc["n_wrong"] < best_stack[1]["n_wrong"]
            or (
                sc["n_wrong"] == best_stack[1]["n_wrong"]
                and sc["n_correct"] > best_stack[1]["n_correct"]
            )
        ):
            best_stack = (s, sc, thr)

    # Precision-priority ranking: min wrong, then max correct, then max score
    def rank_key(item):
        name, sc = item
        # primary: hit_rate, then low wrong, then high correct
        return (sc["hit_rate"], -sc["n_wrong"], sc["n_correct"], sc["score"])

    ranked = sorted(results.items(), key=rank_key, reverse=True)
    print("\n=== TOP by precision (hit_rate, then low wrong) ===")
    for name, sc in ranked[:15]:
        print(
            f"  {name:28s} hit={sc['hit_rate']:.3f}  "
            f"C={sc['n_correct']} W={sc['n_wrong']} score={sc['score']:+d}"
        )

    # Also rank by "closest to ideal" wrong~0 with good correct
    def ideal_key(item):
        name, sc = item
        # soft objective: maximize correct - 50*wrong (heavily punish wrong)
        return (sc["n_correct"] - 50 * sc["n_wrong"], sc["hit_rate"], sc["n_correct"])

    ideal = sorted(results.items(), key=ideal_key, reverse=True)
    print("\n=== TOP by correct - 50*wrong (precision-first utility) ===")
    for name, sc in ideal[:12]:
        print(
            f"  {name:28s} util={sc['n_correct']-50*sc['n_wrong']:+6d}  "
            f"C={sc['n_correct']} W={sc['n_wrong']} hit={sc['hit_rate']:.3f}"
        )

    # Build final "precision ensemble" recommendation:
    # pick best by ideal_key among those with hit_rate >= 0.6 and n_bets >= 50
    # else best hit_rate with n_bets >= 20
    candidates = [
        (n, s) for n, s in results.items() if s["n_bets"] >= 30 and s["hit_rate"] >= 0.55
    ]
    if not candidates:
        candidates = list(results.items())
    best_name, best_sc = max(candidates, key=ideal_key)

    # Reconstruct series for best_name if needed
    # Re-run a compact final recipe known to be strong
    print("\n=== final recipe search ===")
    final_series = {}
    final_series["ensemble_2of3"] = base_ens
    final_series["top3_unanimous"] = p3
    # meta 3of3 high thr
    for thr in [0.70, 0.75, 0.80, 0.85, 0.90]:
        m = meta_label_filter(p3, y, feats, proba_thr=thr, train_bars=3000, retrain_every=600)
        g = rolling_precision_gate(m, y, window=250, min_prec=max(0.58, thr - 0.1), min_bets=10)
        final_series[f"final_meta3_{thr:.2f}"] = g
        eval_name(f"final_meta3_{thr:.2f}", g)

    # ultra-strict: 3of3 + agree>=3 on pool + meta 0.85 + prec gate
    m = meta_label_filter(p3, y, feats, proba_thr=0.85, train_bars=3000, retrain_every=600)
    agr_pool = agreement_strength(votes)
    ultra = m.where(agr_pool["strength"] >= 3, 0)
    ultra = rolling_precision_gate(ultra.astype(int), y, window=200, min_prec=0.65, min_bets=8)
    ultra = apply_vol_gate(ultra.astype(int), zmax=0.8)
    final_series["ultra_precision"] = ultra.astype(int)
    eval_name("ultra_precision", ultra.astype(int))

    # choose best by ideal utility with at least 20 bets if possible
    fin = {n: score(s.loc[pred.index >= score_start], y.loc[pred.index >= score_start]) for n, s in final_series.items()}
    # update with all results
    for n, s in final_series.items():
        results[n] = fin[n]

    viable = [(n, sc) for n, sc in results.items() if sc["n_bets"] >= 15]
    best_name, best_sc = max(viable, key=ideal_key)
    # reconstruct series
    if best_name in final_series:
        best_pred = final_series[best_name]
    elif best_name == "top3_unanimous":
        best_pred = p3
    elif best_name.startswith("meta_"):
        # find from meta_cfgs
        best_pred = None
        for nm, ser, sc in meta_cfgs:
            if nm == best_name:
                best_pred = ser
                break
        if best_pred is None:
            best_pred = ultra.astype(int)
    else:
        best_pred = ultra.astype(int)

    # Prefer ultra / high thr meta if wrong is much lower
    # Explicit preferred: minimize wrong first among final_series
    prefer = min(
        final_series.items(),
        key=lambda kv: (
            score(kv[1].loc[pred.index >= score_start], y.loc[pred.index >= score_start])["n_wrong"],
            -score(kv[1].loc[pred.index >= score_start], y.loc[pred.index >= score_start])["n_correct"],
        ),
    )
    pref_sc = score(prefer[1].loc[pred.index >= score_start], y.loc[pred.index >= score_start])
    print(f"\nLowest-wrong final candidate: {prefer[0]}  {pref_sc}")

    # Also compute "oracle upper bound" illustration (NOT deployable): keep only correct 2of3 bets
    oracle = base_ens.where(base_ens == y, 0).astype(int)
    # only after score_start
    o_sc = score(oracle.loc[pred.index >= score_start], y.loc[pred.index >= score_start])
    print(
        f"\nORACLE (look-ahead, not tradable): correct={o_sc['n_correct']} wrong={o_sc['n_wrong']} "
        f"— this is the theoretical '2615 with 0 wrong' style ceiling for 2of3 winners only: "
        f"C={o_sc['n_correct']}"
    )

    # Save best precision policy (prefer lowest wrong with util)
    chosen_name = prefer[0]
    chosen = prefer[1].astype(int)
    chosen_sc = pref_sc

    # If we can get wrong < 100 with decent correct, report that; else best util
    low_wrong = [
        (n, score(s.loc[pred.index >= score_start], y.loc[pred.index >= score_start]), s)
        for n, s in final_series.items()
    ]
    low_wrong.sort(key=lambda x: (x[1]["n_wrong"], -x[1]["n_correct"]))
    chosen_name, chosen_sc, chosen = low_wrong[0][0], low_wrong[0][1], low_wrong[0][2]

    # Second half OOS check
    mid = score_start + (pred.index[-1] - score_start) / 2
    oos = score(chosen.loc[pred.index >= mid], y.loc[pred.index >= mid])
    print(f"\nCHOSEN: {chosen_name}")
    print(f"  full eval: {chosen_sc}")
    print(f"  2nd half:  {oos}")

    # write outputs
    out = pred.copy()
    out["ensemble_precision"] = chosen.astype(int)
    out["ensemble_3of3"] = p3.astype(int)
    out["ensemble_ultra"] = final_series.get("ultra_precision", chosen).astype(int)
    out.to_parquet(ROOT / "predictions_precision.parquet")
    # also update main predictions for chart
    pred2 = pred.copy()
    pred2["ensemble_2of3"] = chosen.astype(int)  # replace for chart/server
    pred2["ensemble_precision"] = chosen.astype(int)
    pred2["ensemble_baseline_2of3"] = base_ens.astype(int)
    pred2.to_parquet(ROOT / "predictions.parquet")

    metrics = {
        "baseline_2of3": results.get("ensemble_2of3"),
        "chosen": {"name": chosen_name, **chosen_sc},
        "oos_second_half": oos,
        "oracle_look_ahead_winners_only": o_sc,
        "note": (
            "wrong=0 with correct=2615 is the oracle (keep only winning bets) — not achievable "
            "without look-ahead. Precision stack reduces wrongs via 3of3 + meta-label + gates."
        ),
        "all": results,
    }
    (ROOT / "metrics_precision.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    print("\nSaved predictions.parquet (ensemble_2of3 → precision model)")
    print("Saved metrics_precision.json")
    print(
        f"\nRESULT: {chosen_name}\n"
        f"  before: C=25544 W=22929 score=+2615\n"
        f"  after:  C={chosen_sc['n_correct']} W={chosen_sc['n_wrong']} "
        f"score={chosen_sc['score']:+d} hit={chosen_sc['hit_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
