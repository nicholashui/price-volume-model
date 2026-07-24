# price-volume-model

Price–volume stochastic model (Black–Scholes → intensity SV → EKF) plus a **1-year Bitcoin 5-minute direction backtest**.

All source code, data, metrics, plots, and reports live in **this directory** (no nested package required).

## Theory

See **[model.md](model.md)** for the full derivation:

- Joint \((P,V)\) dynamics and marked point processes
- Buy/sell intensities \(\lambda^\pm\)
- Volume-driven stochastic volatility (CIR)
- Extended Kalman Filter for latent volume
- \(\mathbb{P}(S(t+dt)>S(t))\) and \(\mathbb{P}(S(t+dt)<S(t))\)

## Backtest (direction skill)

Predict at clock times \(t\) with minute \(\in\{0,5,\ldots,55\}\), horizon \(dt=5\) min:

| Decision | Score if outcome known |
|----------|------------------------|
| +1 (UP) / −1 (DOWN) correct | **+1** |
| wrong bet | **−1** |
| 0 (no bet) | **0** |

**Final ensemble (2-of-3):** `intensity_fade` + `logistic_wf` + `bollinger`  
**Result (1y BTCUSDT):** cumulative score **+2,492**, hit rate **52.63%** on 47,292 bets.

Full write-up: **[REPORT.md](REPORT.md)**.

## Quick start

```bash
python download_data.py    # Binance BTCUSDT 1m (~1 year) → btcusdt_1m.parquet
python run_backtest.py     # full pipeline; all outputs in current directory
```

## Current-directory layout

| File | Description |
|------|-------------|
| `model.md` | Theoretical model |
| `REPORT.md` | Complete backtest report |
| `README.md` | This file |
| `download_data.py` | Data download |
| `intensity_model.py` | EKF + P(up)/P(down) |
| `indicators.py` | ~114 features |
| `models.py` | Rules + walk-forward ML + 2-of-3 |
| `backtest.py` | Pipeline |
| `run_backtest.py` | CLI entrypoint |
| `btcusdt_1m.parquet` | 1m OHLCV |
| `metrics.json` | All scores & params |
| `predictions.parquet` | Per-bar decisions + label |
| `cum_hit_and_price.png` | Price + cumulative hit |
| `model_ranking.png` | Model ranking |
| `hitrate_coverage.png` | Hit rate vs coverage |
| `ensemble_monthly.png` | Monthly ensemble score |
| `buy_sell_probability.png` | Prior intensity illustration |
