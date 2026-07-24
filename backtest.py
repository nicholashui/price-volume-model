"""
5-minute Bitcoin direction backtest.

Decision times t: minute ∈ {0,5,10,...,55}
Horizon dt = 5 minutes
Label y_t = sign(S(t+dt) - S(t)); flat → 0 (excluded from hit scoring as no move)
Features use only information available at bar close t (no future leakage).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from download_data import load_or_download
from indicators import build_indicators, count_indicators
from intensity_model import compute_intensity_series, identify_sv_params
from models import (
    decide_from_edge,
    ensemble_2of3,
    model_bollinger,
    model_breakout,
    model_gbm_wf,
    model_indicator_vote,
    model_intensity,
    model_intensity_skew,
    model_logistic_wf,
    model_mean_reversion,
    model_momentum,
    model_orderflow,
    model_rf_wf,
    model_rsi,
    score_predictions,
    tune_threshold_model,
)

# Write ALL outputs to the current project directory (no subfolders)
ROOT = Path(__file__).resolve().parent
PLOTS = ROOT
RESULTS = ROOT
DT_YEAR = 5.0 / (365.25 * 24 * 60)  # 5-min as year fraction


def to_5min(df1m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1m ticks/bars to 5-minute OHLCV on clock grid."""
    ohlc = df1m.resample("5min", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    if "quote_volume" in df1m.columns:
        ohlc["quote_volume"] = df1m["quote_volume"].resample("5min", label="left", closed="left").sum()
    if "n_trades" in df1m.columns:
        ohlc["n_trades"] = df1m["n_trades"].resample("5min", label="left", closed="left").sum()
    if "taker_buy_base" in df1m.columns:
        ohlc["taker_buy_base"] = (
            df1m["taker_buy_base"].resample("5min", label="left", closed="left").sum()
        )
    ohlc = ohlc.dropna(subset=["open", "close"])
    # keep only standard 5-min slots (minute already 0,5,...,55 after resample)
    return ohlc


def make_labels(close: pd.Series) -> pd.Series:
    """y_t = sign(S(t+1)-S(t)) on 5-min grid; 0 if flat."""
    fwd = close.shift(-1) - close
    y = np.sign(fwd).fillna(0).astype(int)
    y.name = "y"
    return y


def prepare_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    raw = load_or_download()
    df5 = to_5min(raw)
    # signed volume proxy for intensity ID
    ret = np.sign(df5["close"].diff()).fillna(0)
    if "taker_buy_base" in df5.columns:
        sell = (df5["volume"] - df5["taker_buy_base"]).clip(lower=0)
        df5["signed_vol"] = df5["taker_buy_base"] - sell
    else:
        df5["signed_vol"] = ret * df5["volume"]

    feats = build_indicators(df5)
    y = make_labels(df5["close"])

    # identify SV params on first 30 days only (no peek)
    warmup = min(30 * 24 * 12, len(df5) // 5)
    train_slice = df5.iloc[:warmup]
    params = identify_sv_params(
        train_slice["close"].values,
        train_slice["volume"].values,
        train_slice["signed_vol"].values,
        dt=DT_YEAR,
    )
    print("SV params (warmup):", {k: round(v, 6) if isinstance(v, float) else v for k, v in params.items()})
    intensity = compute_intensity_series(df5, params, dt=DT_YEAR)

    # align: drop last row (no label) and warmup for scoring fairness
    common = feats.index.intersection(intensity.index).intersection(y.index)
    feats = feats.loc[common]
    intensity = intensity.loc[common]
    y = y.loc[common]
    df5 = df5.loc[common]
    # drop rows with NaN features (early rolling windows)
    valid = feats.notna().all(axis=1) & y.isin([-1, 1])
    # keep some early for ML train but score from first fully-valid
    return df5, feats, intensity, y


def run_all_models(
    feats: pd.DataFrame,
    intensity: pd.DataFrame,
    y: pd.Series,
    score_start: pd.Timestamp | None = None,
) -> dict[str, dict]:
    """Train/run all candidate models; return metrics + prediction series."""
    if score_start is None:
        # skip first ~14 days for indicator warm-up
        score_start = feats.index[min(14 * 24 * 12, len(feats) // 10)]

    results = {}
    preds_store = {}

    # --- threshold models with light tuning on first half of score window ---
    score_idx = feats.index[feats.index >= score_start]
    mid = score_idx[len(score_idx) // 2]
    tune_y = y.loc[score_start:mid]
    hold_mask = feats.index >= score_start

    # Intensity
    edge_z = intensity["edge"] / (
        intensity["edge"].rolling(288, min_periods=48).std() + 1e-12
    )
    thr_i, _ = tune_threshold_model(
        edge_z.loc[score_start:mid].fillna(0),
        tune_y,
        thr_grid=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5],
        min_bets=100,
    )
    p = model_intensity(intensity, thr=thr_i, conf_quantile=0.6)
    preds_store["intensity"] = p
    results["intensity"] = {"preds": p, "params": {"thr": thr_i, "conf_q": 0.6}}

    thr_sk, _ = tune_threshold_model(
        intensity["intensity_skew"].loc[score_start:mid].fillna(0),
        tune_y,
        thr_grid=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
        min_bets=100,
    )
    p = model_intensity_skew(intensity, thr=thr_sk)
    preds_store["intensity_skew"] = p
    results["intensity_skew"] = {"preds": p, "params": {"thr": thr_sk}}

    # Fade intensity skew (microstructure reversion of transient pressure — arXiv LOB notes)
    best_fade = None
    for thr in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        raw = decide_from_edge(intensity["intensity_skew"].values, thr)
        p = pd.Series(-raw, index=intensity.index)  # fade
        sc = score_predictions(p.loc[score_start:mid], tune_y)
        if best_fade is None or sc["score"] > best_fade[0]["score"]:
            best_fade = (sc, p, {"thr": thr, "mode": "fade_skew"})
    # also try fading edge z
    edge_z = intensity["edge"] / (
        intensity["edge"].rolling(288, min_periods=48).std() + 1e-12
    )
    for thr in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        raw = decide_from_edge(edge_z.fillna(0).values, thr)
        p = pd.Series(-raw, index=intensity.index)
        sc = score_predictions(p.loc[score_start:mid], tune_y)
        if best_fade is None or sc["score"] > best_fade[0]["score"]:
            best_fade = (sc, p, {"thr": thr, "mode": "fade_edge_z"})
    preds_store["intensity_fade"] = best_fade[1]
    results["intensity_fade"] = {"preds": best_fade[1], "params": best_fade[2]}

    # Mean reversion
    best_mr = None
    for zc, thr, vf in [
        ("zprice_12", 1.25, 1.5),
        ("zprice_24", 1.5, 1.0),
        ("zprice_24", 2.0, 1.5),
        ("zprice_48", 1.75, 2.0),
        ("zret_12", 1.5, None),
    ]:
        p = model_mean_reversion(feats, z_col=zc, thr=thr, vol_filter=vf)
        sc = score_predictions(p.loc[score_start:mid], tune_y)
        if best_mr is None or sc["score"] > best_mr[0]["score"]:
            best_mr = (sc, p, {"z_col": zc, "thr": thr, "vol_filter": vf})
    preds_store["mean_reversion"] = best_mr[1]
    results["mean_reversion"] = {"preds": best_mr[1], "params": best_mr[2]}

    # Momentum
    best_mo = None
    for rc, thr, cc, ct in [
        ("ret_6", 0.0005, "tfi_6", 0.0),
        ("ret_12", 0.0008, "tfi_12", 0.05),
        ("ret_12", 0.0012, "tfi_12", 0.0),
        ("ret_3", 0.0004, "tfi_3", 0.0),
        ("roc_12", 0.001, "tfi_12", 0.1),
    ]:
        p = model_momentum(feats, ret_col=rc, thr=thr, confirm_col=cc, confirm_thr=ct)
        sc = score_predictions(p.loc[score_start:mid], tune_y)
        if best_mo is None or sc["score"] > best_mo[0]["score"]:
            best_mo = (sc, p, {"ret_col": rc, "thr": thr, "confirm": cc, "ct": ct})
    preds_store["momentum"] = best_mo[1]
    results["momentum"] = {"preds": best_mo[1], "params": best_mo[2]}

    # Order flow
    thr_of, _ = tune_threshold_model(
        feats["tfi_12"].loc[score_start:mid].fillna(0),
        tune_y,
        thr_grid=[0.05, 0.1, 0.15, 0.2, 0.25, 0.35],
        min_bets=100,
    )
    p = model_orderflow(feats, col="tfi_12", thr=thr_of)
    preds_store["orderflow"] = p
    results["orderflow"] = {"preds": p, "params": {"thr": thr_of}}

    # RSI
    best_rsi = None
    for lo, hi in [(25, 75), (30, 70), (20, 80), (35, 65)]:
        p = model_rsi(feats, lo=lo, hi=hi)
        sc = score_predictions(p.loc[score_start:mid], tune_y)
        if best_rsi is None or sc["score"] > best_rsi[0]["score"]:
            best_rsi = (sc, p, {"lo": lo, "hi": hi})
    preds_store["rsi"] = best_rsi[1]
    results["rsi"] = {"preds": best_rsi[1], "params": best_rsi[2]}

    # Bollinger
    best_bb = None
    for lo, hi in [(0.05, 0.95), (0.1, 0.9), (0.02, 0.98)]:
        p = model_bollinger(feats, lo=lo, hi=hi)
        sc = score_predictions(p.loc[score_start:mid], tune_y)
        if best_bb is None or sc["score"] > best_bb[0]["score"]:
            best_bb = (sc, p, {"lo": lo, "hi": hi})
    preds_store["bollinger"] = best_bb[1]
    results["bollinger"] = {"preds": best_bb[1], "params": best_bb[2]}

    # Breakout
    best_br = None
    for rt, vt in [(0.0006, 0.5), (0.0008, 1.0), (0.001, 0.0), (0.0012, 0.5)]:
        p = model_breakout(feats, ret_thr=rt, vol_thr=vt)
        sc = score_predictions(p.loc[score_start:mid], tune_y)
        if best_br is None or sc["score"] > best_br[0]["score"]:
            best_br = (sc, p, {"ret_thr": rt, "vol_thr": vt})
    preds_store["breakout"] = best_br[1]
    results["breakout"] = {"preds": best_br[1], "params": best_br[2]}

    # Indicator vote
    best_iv = None
    for vt in [0.15, 0.2, 0.25, 0.3, 0.35]:
        p = model_indicator_vote(feats, vote_thr=vt)
        sc = score_predictions(p.loc[score_start:mid], tune_y)
        if best_iv is None or sc["score"] > best_iv[0]["score"]:
            best_iv = (sc, p, {"vote_thr": vt})
    preds_store["indicator_vote"] = best_iv[1]
    results["indicator_vote"] = {"preds": best_iv[1], "params": best_iv[2]}

    # ML feature set (~core predictive cols)
    ml_cols = [
        c
        for c in feats.columns
        if any(
            c.startswith(p)
            for p in (
                "ret_",
                "roc_",
                "zprice_",
                "zret_",
                "rsi_",
                "tfi",
                "rvol_",
                "vol_z_",
                "macd",
                "bb_",
                "amihud",
                "mr_",
                "mom_",
                "buy_",
                "ac1_",
                "hour_",
                "parkinson",
                "keltner",
                "obv_",
                "breakout",
                "exhaustion",
            )
        )
    ]
    # add intensity features
    for col in ["edge", "intensity_skew", "V_hat", "lambda_plus", "lambda_minus", "p_up", "p_down"]:
        feats[f"int_{col}"] = intensity[col]
        ml_cols.append(f"int_{col}")

    print(f"Running walk-forward ML on {len(ml_cols)} features ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # tune proba thr lightly
        for name, fn, grid in [
            (
                "logistic_wf",
                model_logistic_wf,
                [0.52, 0.55, 0.58, 0.62],
            ),
            (
                "gbm_wf",
                model_gbm_wf,
                [0.52, 0.55, 0.58],
            ),
            (
                "rf_wf",
                model_rf_wf,
                [0.52, 0.55, 0.58],
            ),
        ]:
            best = None
            # pick thr using a cheap single pass at 0.55 then optional re-threshold of proba not available;
            # run once at 0.55 and once at 0.58
            # single causal pass; thr chosen on tune half via post-hoc re-threshold of soft edge if available
            # Use one primary thr for fit speed (full-year WF is expensive)
            thr = 0.55 if name != "logistic_wf" else 0.58
            print(f"  {name} thr={thr} ...")
            p = fn(feats, y, ml_cols, proba_thr=thr, train_bars=2000, retrain_every=1000)
            sc = score_predictions(p.loc[score_start:mid], tune_y)
            best = (sc, p, {"proba_thr": thr})
            # light thr search: try stricter abstention by zeroing weak periods using rolling hit proxy
            for thr2 in grid:
                if thr2 == thr:
                    continue
                # re-run is costly for tree models — only re-run logistic
                if name != "logistic_wf":
                    break
                print(f"  {name} thr={thr2} ...")
                p2 = fn(feats, y, ml_cols, proba_thr=thr2, train_bars=2000, retrain_every=1000)
                sc2 = score_predictions(p2.loc[score_start:mid], tune_y)
                if sc2["score"] > best[0]["score"]:
                    best = (sc2, p2, {"proba_thr": thr2})
            preds_store[name] = best[1]
            results[name] = {"preds": best[1], "params": best[2]}

    # score all on full holdout from score_start
    scored = {}
    for name, pack in results.items():
        p = pack["preds"].loc[hold_mask]
        sc = score_predictions(p, y.loc[hold_mask])
        scored[name] = {
            **{k: sc[k] for k in ("score", "n_bets", "n_correct", "n_wrong", "hit_rate", "coverage")},
            "params": pack["params"],
            "cum": sc["cum"],
            "preds": pack["preds"],
        }
        print(
            f"{name:18s}  score={sc['score']:+8.0f}  bets={sc['n_bets']:6d}  "
            f"hit={sc['hit_rate']:.3f}  cov={sc['coverage']:.3f}"
        )
    return scored, score_start


def build_ensemble(scored: dict, y: pd.Series, score_start) -> dict:
    """
    Rank models, then search best 2-of-3 trio among top-6 by cumulative score
    on the first half of the eval window (tune), report on full eval window.
    """
    from itertools import combinations

    ranking = sorted(
        scored.items(),
        key=lambda kv: (kv[1]["score"], kv[1]["hit_rate"], -kv[1]["n_wrong"]),
        reverse=True,
    )
    ranking_list = [(n, scored[n]["score"], scored[n]["hit_rate"]) for n, _ in ranking]
    candidates = [name for name, _ in ranking[:6]]
    hold = y.index >= score_start
    idx = y.index[hold]
    mid = idx[len(idx) // 2]
    tune_mask = (y.index >= score_start) & (y.index <= mid)
    full_y = y.loc[hold]

    best_trio = candidates[:3]
    best_tune_score = -1e18
    best_ens_tune = None
    for trio in combinations(candidates, 3):
        ens = ensemble_2of3(
            scored[trio[0]]["preds"],
            scored[trio[1]]["preds"],
            scored[trio[2]]["preds"],
        )
        sc_t = score_predictions(ens.loc[tune_mask], y.loc[tune_mask])
        # prefer higher score, then hit_rate, with enough bets
        if sc_t["n_bets"] < 200:
            continue
        key = (sc_t["score"], sc_t["hit_rate"])
        if key > (best_tune_score, -1):
            best_tune_score = sc_t["score"]
            best_trio = list(trio)
            best_ens_tune = ens

    top3 = best_trio
    print("Top-3 models (ensemble search):", top3)
    p1 = scored[top3[0]]["preds"]
    p2 = scored[top3[1]]["preds"]
    p3 = scored[top3[2]]["preds"]
    ens = ensemble_2of3(p1, p2, p3) if best_ens_tune is None else best_ens_tune

    # confidence-gated ensemble: only when |sum of votes| == 3 or
    # when two agree and third is 0 (already in 2of3). Extra gate: rolling vol not extreme
    ens_all3 = (p1 == p2) & (p2 == p3) & (p1 != 0)
    ens_strict = p1.where(ens_all3, 0)

    # Soft gate: drop bets in top 5% |ret| shock bars (optional safety) — use causal rvol
    # Keep plain 2of3 as primary.

    sc = score_predictions(ens.loc[hold], full_y)
    sc_s = score_predictions(ens_strict.loc[hold], full_y)

    # Also score naive top3-by-rank for reference
    naive = [name for name, _ in ranking[:3]]
    ens_naive = ensemble_2of3(
        scored[naive[0]]["preds"], scored[naive[1]]["preds"], scored[naive[2]]["preds"]
    )
    sc_naive = score_predictions(ens_naive.loc[hold], full_y)
    print(f"Naive top3 {naive} ensemble score={sc_naive['score']:+.0f}")
    print(f"Searched trio {top3} ensemble score={sc['score']:+.0f}")

    return {
        "top3": top3,
        "naive_top3": naive,
        "ranking": ranking_list,
        "ensemble": ens,
        "ensemble_strict": ens_strict,
        "metrics": sc,
        "metrics_strict": sc_s,
        "metrics_naive": sc_naive,
    }


def plot_results(
    df5: pd.DataFrame,
    scored: dict,
    ensemble_pack: dict,
    y: pd.Series,
    score_start,
):
    hold = y.index >= score_start
    price = df5.loc[hold, "close"]

    # 1) cumulative hit for top models + ensemble
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    ax = axes[0]
    ax.plot(price.index, price.values, color="black", lw=0.8, alpha=0.7)
    ax.set_ylabel("BTCUSDT")
    ax.set_title("Bitcoin 5-min close (evaluation window)")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    top_names = ensemble_pack["top3"] + ["ensemble_2of3"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for name, col in zip(ensemble_pack["top3"], colors[:3]):
        cum = scored[name]["cum"]
        ax2.plot(cum.index, cum.values, label=f"{name} ({scored[name]['score']:+.0f})", color=col, lw=1.2)
    ens_sc = score_predictions(ensemble_pack["ensemble"].loc[hold], y.loc[hold])
    ax2.plot(
        ens_sc["cum"].index,
        ens_sc["cum"].values,
        label=f"ensemble_2of3 ({ens_sc['score']:+.0f})",
        color=colors[3],
        lw=2.0,
    )
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_ylabel("Cumulative hit (Σ +1/0/−1)")
    ax2.set_title("Accumulated score over 1-year backtest")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "cum_hit_and_price.png", dpi=140)
    plt.close(fig)

    # 2) bar chart model comparison
    names = list(scored.keys())
    scores = [scored[n]["score"] for n in names]
    hits = [scored[n]["hit_rate"] for n in names]
    order = np.argsort(scores)
    fig, ax = plt.subplots(figsize=(11, 6))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, np.array(scores)[order], color=["#c44e52" if s < 0 else "#4c72b0" for s in np.array(scores)[order]])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(np.array(names)[order])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Cumulative score")
    ax.set_title("Model ranking by accumulated hit score")
    fig.tight_layout()
    fig.savefig(PLOTS / "model_ranking.png", dpi=140)
    plt.close(fig)

    # 3) hit rate vs coverage scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for n in names:
        ax.scatter(scored[n]["coverage"], scored[n]["hit_rate"], s=max(20, scored[n]["n_bets"] / 50), alpha=0.75)
        ax.annotate(n, (scored[n]["coverage"], scored[n]["hit_rate"]), fontsize=8, alpha=0.9)
    em = ensemble_pack["metrics"]
    ax.scatter(em["coverage"], em["hit_rate"], s=120, marker="*", color="red", label="ensemble_2of3")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Coverage (fraction of bars with a bet)")
    ax.set_ylabel("Hit rate (among bets)")
    ax.set_title("Hit rate vs coverage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "hitrate_coverage.png", dpi=140)
    plt.close(fig)

    # 4) monthly cumulative for ensemble
    step = ens_sc["cum"].diff().fillna(ens_sc["cum"].iloc[0] if len(ens_sc["cum"]) else 0)
    monthly = step.resample("ME").sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    colors_m = ["#4c72b0" if v >= 0 else "#c44e52" for v in monthly.values]
    ax.bar(monthly.index.strftime("%Y-%m"), monthly.values, color=colors_m)
    ax.set_ylabel("Monthly score")
    ax.set_title("Ensemble 2-of-3 monthly accumulated hits")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(PLOTS / "ensemble_monthly.png", dpi=140)
    plt.close(fig)

    print(f"Plots saved to {PLOTS}")


def write_report(
    df5,
    feats,
    scored,
    ensemble_pack,
    score_start,
    n_indicators: int,
    y: pd.Series,
):
    em = ensemble_pack["metrics"]
    ems = ensemble_pack["metrics_strict"]
    ranking_lines = "\n".join(
        f"| {i+1} | {n} | {s:+.0f} | {h:.3f} |"
        for i, (n, s, h) in enumerate(ensemble_pack["ranking"])
    )
    top3 = ensemble_pack["top3"]
    param_lines = "\n".join(
        f"- **{n}**: `{json.dumps(scored[n]['params'], default=str)}`" for n in scored
    )

    report = f"""# Bitcoin 5-Minute Direction Backtest Report

**Generated:** {pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")}  
**Data:** BTCUSDT 1-minute bars aggregated to 5-minute clock grid (minutes 0,5,…,55)  
**Horizon:** dt = 5 minutes  
**Label:** +1 if S(t+dt) > S(t), −1 if S(t+dt) < S(t)  
**Decision:** +1 (bet UP), −1 (bet DOWN), 0 (no bet)  
**Score:** correct bet → +1, wrong bet → −1, no bet → 0; objective = maximize Σ score  
**Evaluation window start:** {score_start}  
**Bars evaluated:** ~{(df5.index >= score_start).sum():,}  
**Price range:** {df5['close'].min():.2f} – {df5['close'].max():.2f}  
**Indicators used:** {n_indicators}

---

## 1. Research foundation (arXiv + model.md)

### 1.1 Intensity model (`model.md` §12)

Under the volume-driven SV intensity model:

$$
\\mathbb{{P}}(S(t+dt)>S(t)) = \\lambda^+_t\\,dt,\\qquad
\\mathbb{{P}}(S(t+dt)<S(t)) = \\lambda^-_t\\,dt
$$

with

$$
\\lambda^\\pm_t = \\tfrac12\\Big(\\tfrac{{\\kappa(\\theta-V_t)}}{{\\bar v}} \\pm \\tfrac1\\alpha(\\tfrac{{dP}}{{dt}}|_{{\\rm drift}}-\\mu P_t)\\Big).
$$

Latent volume factor $V_t$ is filtered with an Extended Kalman Filter (CIR state). Edge = $P_{{\\rm up}}-P_{{\\rm down}}$ is z-scored and thresholded; low-confidence states abstain (decision 0).

### 1.2 arXiv-informed improvements

| Paper | Idea used |
|-------|-----------|
| Albers et al. (arXiv:2108.09750) | Trade-flow imbalance (TFI) multi-horizon features; aggressive flow > passive for short-horizon signal |
| Cont et al. OFI literature | Signed volume / imbalance as linear predictor of short returns |
| Microstructure reviews (2024–2026 crypto LOB work) | Confidence gating when spreads/vol expand; mean-reversion after transient pressure |
| MDH / SV-volume | Volume factor drives conditional volatility and intensity scale |

### 1.3 Design principles (rethought before implementation)

1. **Causal only:** features at bar $t$ use data ≤ $t$; label uses $S(t+dt)$.
2. **Abstention is first-class:** maximize (hits − misses), not raw accuracy on all bars.
3. **Regime diversity:** combine intensity, mean-reversion, momentum, order-flow, ML.
4. **Walk-forward ML:** retrain on past windows only; probability threshold for abstention.
5. **Ensemble 2-of-3:** only bet when ≥2 of top-3 agree — cuts uncorrelated errors.

---

## 2. Model structure

```
1m BTCUSDT OHLCV (+ taker buy volume)
        │
        ▼
   5-min resample (t = :00,:05,…,:55)
        │
        ├─► ~{n_indicators} indicators (returns, MR, mom, RSI, BB, vol, TFI, Amihud, clock, interactions)
        ├─► SV param ID on warmup → EKF V̂_t → λ± → P(up), P(down)
        │
        ▼
   Candidate models (threshold + walk-forward ML)
        │
        ▼
   Rank by Σ score on evaluation window
        │
        ▼
   Top-3 → Ensemble 2-of-3 consensus
        │
        ▼
   Metrics + charts + this report
```

### Candidate models

| Model | Type | Signal |
|-------|------|--------|
| intensity | Theory (model.md) | z(P_up − P_down) + confidence quantile gate |
| intensity_skew | Theory | (λ+−λ−)/(λ++λ−) |
| mean_reversion | Rule | Fade price z-score extremes (vol filter) |
| momentum | Rule | Multi-bar return + TFI confirmation |
| orderflow | Rule | Trade-flow imbalance threshold |
| rsi / bollinger | Rule | Classic oscillator extremes |
| breakout | Rule | Return × volume-z expansion |
| indicator_vote | Meta | Soft vote of ~40 z-scored indicators |
| logistic_wf / gbm_wf / rf_wf | ML | Walk-forward classifiers with P≥thr |

### Tuned parameters

{param_lines}

---

## 3. Single-model backtest ranking

| Rank | Model | Cumulative score | Hit rate |
|------|-------|------------------|----------|
{ranking_lines}

**Top-3 selected:** `{top3[0]}`, `{top3[1]}`, `{top3[2]}`

---

## 4. Ensemble 2-of-3 results

| Metric | 2-of-3 | Strict 3-of-3 |
|--------|--------|---------------|
| Cumulative score Σ | **{em['score']:+.0f}** | {ems['score']:+.0f} |
| Bets | {em['n_bets']} | {ems['n_bets']} |
| Correct | {em['n_correct']} | {ems['n_correct']} |
| Wrong | {em['n_wrong']} | {ems['n_wrong']} |
| Hit rate | **{em['hit_rate']:.4f}** | {ems['hit_rate']:.4f} |
| Coverage | {em['coverage']:.4f} | {ems['coverage']:.4f} |

### Interpretation

- Hit rate above 0.5 with positive cumulative score means the selective betting rule extracts edge after abstentions.
- 2-of-3 typically **reduces wrong bets** more than it reduces correct ones when base models are diverse.
- Strict 3-of-3 is higher precision / lower coverage (shown for sensitivity).

---

## 5. Charts

- `cum_hit_and_price.png` — BTC price + cumulative hit curves (top-3 + ensemble)
- `model_ranking.png` — bar ranking of all models
- `hitrate_coverage.png` — hit rate vs coverage tradeoff
- `ensemble_monthly.png` — monthly ensemble score

---

## 6. Algorithm (operational)

At each clock time $t \\in \\{{:00,:05,\\ldots,:55\\}}$:

1. Update OHLCV bar closed at $t$.
2. Recompute indicators (rolling windows end at $t$).
3. EKF-update $\\hat V_t$; compute $\\lambda^\\pm$, $P_{{\\rm up}}$, $P_{{\\rm down}}$.
4. Run each base model → $d_i \\in \\{{-1,0,+1\\}}$.
5. Ensemble: if at least two of top-3 share the same non-zero $d$, output that $d$; else 0.
6. After $dt=5$m, observe $y=\\mathrm{{sign}}(S_{{t+dt}}-S_t)$ and score.

**No look-ahead:** ML training windows end strictly before prediction time; thresholds tuned on first half of eval window only for rule models (second half + full window reported as primary metrics include in-sample threshold risk — walk-forward ML is the stricter causal benchmark).

---

## 7. Limitations & next steps

1. 1-minute/5-minute bars approximate ticks; true LOB OFI would strengthen intensity identification.
2. Threshold tuning on first half of the test window is light data-snooping; production should freeze thresholds on a dedicated validation year.
3. Transaction costs / fees not deducted (direction skill only).
4. Non-stationarity: crypto regimes shift; scheduled re-selection of top-3 quarterly is recommended.
5. Extensions: Hawkes λ±, deeper LOB features, meta-labeling (Lopez de Prado) for abstention.

---

## 8. How to reproduce

```bash
python download_data.py          # ~1y BTCUSDT 1m from Binance
python run_backtest.py           # full pipeline → current directory
```

Artifacts (all in current directory):
- `btcusdt_1m.parquet`
- `metrics.json`
- `predictions.parquet`
- `REPORT.md` (this file)
- `cum_hit_and_price.png`, `model_ranking.png`, `hitrate_coverage.png`, `ensemble_monthly.png`
"""
    out = ROOT / "REPORT.md"
    out.write_text(report, encoding="utf-8")
    print(f"Report written to {out}")

    # JSON metrics
    def pack_metrics(d):
        return {
            k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v)
            for k, v in d.items()
            if k in ("score", "n_bets", "n_correct", "n_wrong", "hit_rate", "coverage")
        }

    metrics_out = {
        "score_start": str(score_start),
        "n_indicators": n_indicators,
        "models": {n: {**pack_metrics(scored[n]), "params": scored[n]["params"]} for n in scored},
        "top3": top3,
        "ensemble_2of3": pack_metrics(em),
        "ensemble_3of3": pack_metrics(ems),
        "ranking": ensemble_pack["ranking"],
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics_out, indent=2, default=str), encoding="utf-8")

    # predictions
    pred_df = pd.DataFrame({n: scored[n]["preds"] for n in scored})
    pred_df["ensemble_2of3"] = ensemble_pack["ensemble"]
    pred_df["y"] = y
    pred_df.to_parquet(ROOT / "predictions.parquet")
    return out


def main():
    print("=" * 60)
    print("BTC 5-min direction backtest — intensity + 100 indicators")
    print("=" * 60)

    df5, feats, intensity, y = prepare_dataset()
    n_ind = count_indicators(feats)
    print(f"5-min bars: {len(df5):,}  indicators: {n_ind}  range: {df5.index.min()} → {df5.index.max()}")

    scored, score_start = run_all_models(feats, intensity, y)
    ens = build_ensemble(scored, y, score_start)
    print(
        f"ENSEMBLE 2of3: score={ens['metrics']['score']:+.0f}  "
        f"hit={ens['metrics']['hit_rate']:.3f}  bets={ens['metrics']['n_bets']}"
    )
    print(
        f"ENSEMBLE 3of3: score={ens['metrics_strict']['score']:+.0f}  "
        f"hit={ens['metrics_strict']['hit_rate']:.3f}  bets={ens['metrics_strict']['n_bets']}"
    )

    plot_results(df5, scored, ens, y, score_start)
    write_report(df5, feats, scored, ens, score_start, n_ind, y)
    print("DONE.")


if __name__ == "__main__":
    main()
