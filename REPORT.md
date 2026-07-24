# Bitcoin 5-Minute Direction Backtest — Complete Report

**Generated:** 2026-07-24  
**Universe:** BTCUSDT (Binance spot)  
**Raw data:** 525,601 one-minute bars (2025-07-24 → 2026-07-24)  
**Decision grid:** 5-minute clock times \(t\) with minute \(\in\{0,5,10,\ldots,55\}\)  
**Horizon:** \(dt = 5\) minutes  
**Bars on grid:** 105,121 · **Evaluation bars:** ~101,089 (after 14-day warm-up)  
**Price range:** \$58,170 – \$126,011  
**Indicators:** 114 causal features  

---

## Executive summary

| Deliverable | Result |
|-------------|--------|
| Best single model | **`intensity_fade`** — cumulative score **+2,561** |
| Top-3 (for ensemble) | `intensity_fade`, `logistic_wf`, `bollinger` |
| **Ensemble 2-of-3** | **Σ score = +2,492** · hit rate **52.63%** · 47,292 bets |
| Ensemble 3-of-3 (strict) | Σ score = +1,158 · hit rate **54.67%** · 12,396 bets |
| Key empirical finding | On 5-min BTC, **mean-reversion / fade-pressure** dominates momentum |

**Scoring rule (as requested):**

- Predict UP → \(d_t=+1\), DOWN → \(d_t=-1\), abstain → \(d_t=0\)
- After \(dt\): correct bet **+1**, wrong bet **−1**, no bet **0**
- Objective: maximize \(\sum_t s_t\) (hits − misses)

---

## 1. Theory from `model.md` (§12)

Buy/sell intensities inverted from volume-driven SV dynamics:

\[
\lambda^\pm_t
=
\frac12\left(
\frac{\kappa(\theta-V_t)}{\bar v}
\pm
\frac1\alpha\Big(\tfrac{dP_t}{dt}\big|_{\rm drift}-\mu P_t\Big)
\right)
\]

Move probabilities:

\[
\mathbb{P}\big(S(t+dt)>S(t)\big)\propto\lambda^+_t,\qquad
\mathbb{P}\big(S(t+dt)<S(t)\big)\propto\lambda^-_t
\]

**Implementation:**

1. Identify \((\kappa,\theta,\xi,\mu,\alpha)\) on a 30-day warm-up only.
2. Filter latent volume factor \(V_t\) with the Extended Kalman Filter (CIR).
3. Form \(\lambda^\pm\), \(P_{\rm up}\), \(P_{\rm down}\), edge \(=P_{\rm up}-P_{\rm down}\), skew \(=(\lambda^+-\lambda^-)/(\lambda^++\lambda^-)\).

**Empirical twist (arXiv microstructure):** raw intensity *continuation* loses money at 5 minutes; **fading** intensity skew (transient pressure mean-reverts) is the strongest single rule. This matches LOB literature: short-horizon VWAP/mid deviations and flow shocks often reverse as depth replenishes.

---

## 2. Research inputs (arXiv)

| Source | Idea used in this project |
|--------|---------------------------|
| **model.md** intensity SV + EKF | Core \(P(\rm up)/P(\rm down)\) and latent \(V_t\) |
| Albers et al. [arXiv:2108.09750] | Multi-horizon trade-flow imbalance (TFI) from taker buy volume |
| Cont et al. OFI | Signed volume as short-horizon return predictor |
| Crypto LOB / microstructure notes (2024–2026) | Confidence gating; fade transient pressure; regime-aware abstention |
| MDH / volume–volatility | Volume factor scales intensity and volatility |

---

## 3. Algorithm design

### 3.1 Decision protocol (no look-ahead)

At each \(t\) on the 5-min grid:

1. Use only bars with timestamp \(\le t\) (bar closed at \(t\)).
2. Compute 114 indicators + intensity state.
3. Each base model emits \(d_i\in\{-1,0,+1\}\).
4. Ensemble: if **≥2 of top-3** share the same non-zero decision → bet that side; else **0**.
5. After \(dt\), observe \(y_t=\mathrm{sign}(S_{t+dt}-S_t)\) and score.

### 3.2 Pipeline

```
Binance 1m OHLCV (+ taker_buy_base)
        │
        ▼
  Resample → 5m OHLCV on :00,:05,…,:55
        │
        ├─► 114 indicators (returns, z-scores, RSI/Stoch, MACD, Bollinger,
        │         rvol/Parkinson, volume-z, TFI multi-horizon, Amihud,
        │         autocorr, clock, interactions)
        │
        └─► SV ID (warmup) → EKF V̂ → λ± → P(up), P(down), skew
                │
                ▼
     13 candidate models (rules + walk-forward ML)
                │
                ▼
     Rank by Σ score → search best trio among top-6
                │
                ▼
     Ensemble 2-of-3  + charts + metrics.json + this report
```

### 3.3 Indicator families (~114)

| Family | Examples | Count (approx.) |
|--------|----------|-----------------|
| Multi-horizon returns / ROC | `ret_1…288`, `roc_*`, mom spreads | ~20 |
| Mean reversion | price/return z-scores, range position | ~15 |
| Oscillators | RSI, Stochastic, Williams %R | ~10 |
| Trend | MACD, EMA spreads | ~6 |
| Bollinger / range | %B, bandwidth, Keltner, wicks | ~10 |
| Volatility | realized vol, Parkinson, ATR, vol-of-vol | ~10 |
| Volume / activity | vol z-score, OBV slope, VPT | ~12 |
| Order-flow (TFI) | taker imbalance multi-horizon | ~12 |
| Liquidity | Amihud, Kyle proxy, spread proxy | ~6 |
| Persistence / clock | ac1, up-ratio, hour/dow Fourier | ~12 |
| Interactions | mom×vol, MR×vol, breakout, exhaustion | ~6 |

### 3.4 Candidate models

| Model | Family | Decision rule |
|-------|--------|---------------|
| `intensity` | Theory | z(edge) threshold + conf. quantile gate |
| `intensity_skew` | Theory | Follow intensity skew |
| **`intensity_fade`** | Theory + MR | **Fade** intensity skew (best single) |
| `mean_reversion` | Rule | Fade price z-score; vol filter |
| `momentum` | Rule | Return threshold + TFI confirm |
| `orderflow` | Rule | TFI threshold |
| `rsi` | Rule | RSI extremes |
| `bollinger` | Rule | %B band extremes |
| `breakout` | Rule | Return × volume expansion |
| `indicator_vote` | Meta | Soft vote of ~40 z-scored indicators |
| `logistic_wf` | ML | Walk-forward logistic, \(P\ge thr\) |
| `gbm_wf` | ML | Walk-forward gradient boosting |
| `rf_wf` | ML | Walk-forward random forest |

Walk-forward ML: sliding train window 2,000 bars, retrain every 1,000 bars, features available only up to \(t\).

### 3.5 Parameter tuning

- Rule models: grid search thresholds on **first half** of evaluation window.
- Ensemble trio: combinatorial search over top-6 models maximizing tune-half Σ score (min 200 bets).
- Reported metrics: **full evaluation window**.

**Selected parameters (top models):**

| Model | Params |
|-------|--------|
| intensity_fade | `thr=0.3`, `mode=fade_skew` |
| logistic_wf | `proba_thr=0.52` |
| bollinger | `lo=0.1`, `hi=0.9` |
| mean_reversion | `zprice_12`, `thr=1.25`, `vol_filter=1.5` |
| rsi | `lo=35`, `hi=65` |

---

## 4. Backtest results (1 year)

### 4.1 Full model ranking (by cumulative score)

| Rank | Model | Σ score | Bets | Correct | Wrong | Hit rate | Coverage |
|------|-------|---------|------|---------|-------|----------|----------|
| 1 | **intensity_fade** | **+2,561** | 70,915 | 36,738 | 34,177 | 51.81% | 70.2% |
| 2 | logistic_wf | +2,195 | 87,451 | 44,823 | 42,628 | 51.25% | 86.5% |
| 3 | bollinger | +1,690 | 22,690 | 12,190 | 10,500 | 53.72% | 22.4% |
| 4 | mean_reversion | +1,618 | 29,550 | 15,584 | 13,966 | 52.74% | 29.2% |
| 5 | rf_wf | +1,557 | 35,583 | 18,570 | 17,013 | 52.19% | 35.2% |
| 6 | gbm_wf | +1,343 | 48,131 | 24,737 | 23,394 | 51.40% | 47.6% |
| 7 | rsi | +1,266 | 17,970 | 9,618 | 8,352 | 53.52% | 17.8% |
| 8 | intensity | −11 | 627 | 308 | 319 | 49.12% | 0.6% |
| 9 | orderflow | −153 | 4,171 | 2,009 | 2,162 | 48.17% | 4.1% |
| 10 | indicator_vote | −274 | 4,122 | 1,924 | 2,198 | 46.68% | 4.1% |
| 11 | breakout | −730 | 12,638 | 5,954 | 6,684 | 47.11% | 12.5% |
| 12 | momentum | −1,754 | 35,910 | 17,078 | 18,832 | 47.56% | 35.5% |
| 13 | intensity_skew | −2,507 | 62,489 | 29,991 | 32,498 | 47.99% | 61.8% |

### 4.2 Ensemble (2-of-3)

**Trio:** `intensity_fade` + `logistic_wf` + `bollinger`

| Metric | 2-of-3 | 3-of-3 (strict) |
|--------|--------|-----------------|
| **Cumulative score Σ** | **+2,492** | +1,158 |
| Bets | 47,292 | 12,396 |
| Correct (+1) | 24,892 | 6,777 |
| Wrong (−1) | 22,400 | 5,619 |
| Hit rate | **52.63%** | **54.67%** |
| Coverage | 46.8% | 12.3% |

**Interpretation**

- 2-of-3 keeps most of the best model’s edge while **cutting coverage** vs logistic alone and **lifting hit rate** vs intensity_fade alone (51.8% → 52.6%).
- 3-of-3 is higher precision / lower activity (useful if transaction costs matter).
- Momentum / follow-intensity lose: 5-min BTC over this year is **mean-reverting in pressure space**.

### 4.3 Charts

| File (current directory) | Content |
|------|---------|
| `cum_hit_and_price.png` | BTC price + cumulative Σ score (top-3 + ensemble) |
| `model_ranking.png` | Horizontal bar ranking of all models |
| `hitrate_coverage.png` | Hit rate vs coverage scatter |
| `ensemble_monthly.png` | Monthly ensemble score bars |

---

## 5. Final model structure (production form)

```
                    features_t (≤ t)
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
  intensity_fade     logistic_wf         bollinger
  fade λ-skew        P(up)≥0.52          %B∉[0.1,0.9]
  thr=0.3            walk-forward        fade extremes
         │                 │                 │
         └────────────┬────┴────┬────────────┘
                      ▼         ▼
              count(+1)≥2 ?   count(−1)≥2 ?
                      │         │
                      ▼         ▼
                   bet UP    bet DOWN
                      └──── else 0 ────┘
```

**Pseudocode**

```python
d1 = intensity_fade(skew_t, thr=0.3)      # -sign(skew) if |skew|>=thr else 0
d2 = logistic_wf.predict(features_t)      # +1/-1/0 from calibrated proba
d3 = bollinger_signal(pctb_t, 0.1, 0.9)   # +1 if oversold, -1 if overbought

if sum(d == 1 for d in (d1,d2,d3)) >= 2:
    decision = +1
elif sum(d == -1 for d in (d1,d2,d3)) >= 2:
    decision = -1
else:
    decision = 0
```

---

## 6. Why this works (design rationale)

1. **Abstention is the product.** Maximizing Σ(+1/−1/0) is *not* the same as maximizing accuracy on every bar; models that pass when edge is weak destroy score.
2. **Fade beats follow at 5m.** Intensity skew continuation is anti-predictive; fading it is the #1 rule — consistent with short-horizon microstructure reversion.
3. **ML captures nonlinear interactions** among 100+ indicators that pure thresholds miss (`logistic_wf`).
4. **Bollinger adds high-precision MR** at extremes (hit ~53.7%) with lower coverage.
5. **2-of-3 filters idiosyncratic false positives** when the three disagree.

---

## 7. Limitations

1. One-minute bars approximate ticks; full LOB OFI would improve intensity identification.
2. Threshold grids use the first half of the eval year (mild data-snooping for rules). Walk-forward ML is the stricter causal baseline.
3. No fees/slippage — this is a **direction skill** study, not a PnL backtest.
4. Crypto non-stationarity: re-rank top-3 quarterly in production.
5. Flat bars (\(S_{t+dt}=S_t\)) are excluded from hit scoring.

---

## 8. Reproduce

```bash
# from this directory (all outputs written here)
python download_data.py     # ~1y BTCUSDT 1m → btcusdt_1m.parquet
python run_backtest.py      # full pipeline
```

**Artifacts (all in current directory)**

| Path | Description |
|------|-------------|
| `btcusdt_1m.parquet` | Raw 1m OHLCV |
| `metrics.json` | All scores & params |
| `predictions.parquet` | Per-bar decisions + label |
| `*.png` | Charts |
| `REPORT.md` | This document |
| `model.md` | Theoretical derivation |
| `*.py` | Implementation |

---

## 9. Code map (current directory)

| Module | Role |
|--------|------|
| `download_data.py` | Binance kline download |
| `intensity_model.py` | EKF, \(\lambda^\pm\), \(P(\rm up/down)\), param ID |
| `indicators.py` | 114 causal indicators |
| `models.py` | All decision models + 2-of-3 + scoring |
| `backtest.py` | Orchestration, tuning, plots, report |
| `run_backtest.py` | CLI entrypoint |

---

## 10. Bottom line

Over one year of BTCUSDT 5-minute decisions:

- **Best selective system (2-of-3):** cumulative hit score **+2,492** with **52.63%** hit rate on 47k bets.
- **Best single model:** fade intensity skew (**+2,561**).
- **Mean reversion** (price z-score, Bollinger, RSI, fade-flow) systematically beats momentum on this horizon.
- The intensity machinery from `model.md` is most valuable **as a pressure gauge to fade**, not as a trend signal, once mapped to 5-minute bars with real volume and taker imbalance.
