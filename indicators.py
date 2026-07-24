"""
~100 causal technical / microstructure indicators for 5-min BTC bars.

All rolling stats use past-only windows (no leakage into target bar).
arXiv-informed features:
  - Order-flow / trade imbalance proxies (Albers et al. 2108.09750)
  - Mean-reversion & momentum at multi-horizon
  - Volume-driven volatility (MDH / SV)
  - Signed-volume pressure, Amihud illiquidity
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    ma_up = up.ewm(alpha=1 / n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = ma_up / (ma_dn + 1e-12)
    return 100 - (100 / (1 + rs))


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    ll = low.rolling(n, min_periods=max(2, n // 2)).min()
    hh = high.rolling(n, min_periods=max(2, n // 2)).max()
    return 100 * (close - ll) / (hh - ll + 1e-12)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=max(2, n // 2)).mean()


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ~100 indicators from OHLCV (+ optional taker buy volume).

    Expected columns: open, high, low, close, volume
    Optional: taker_buy_base, n_trades, quote_volume
    """
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    feats = pd.DataFrame(index=df.index)

    # --- returns multi-horizon (12) ---
    logc = np.log(c.replace(0, np.nan))
    for n in [1, 2, 3, 6, 12, 24, 36, 48, 72, 96, 144, 288]:
        feats[f"ret_{n}"] = logc.diff(n)

    # --- momentum / ROC (8) ---
    for n in [3, 6, 12, 24, 48, 96]:
        feats[f"roc_{n}"] = c.pct_change(n)
    feats["mom_12_48"] = c.pct_change(12) - c.pct_change(48)
    feats["mom_6_24"] = c.pct_change(6) - c.pct_change(24)

    # --- mean reversion / z-scores (12) ---
    for n in [6, 12, 24, 48, 96]:
        m = c.rolling(n, min_periods=max(3, n // 2)).mean()
        s = c.rolling(n, min_periods=max(3, n // 2)).std()
        feats[f"zprice_{n}"] = (c - m) / (s + 1e-12)
    for n in [12, 24, 48]:
        r = logc.diff()
        m = r.rolling(n, min_periods=max(3, n // 2)).mean()
        s = r.rolling(n, min_periods=max(3, n // 2)).std()
        feats[f"zret_{n}"] = (r - m) / (s + 1e-12)
    # distance to rolling max/min
    for n in [24, 48, 96]:
        hh = h.rolling(n, min_periods=max(3, n // 2)).max()
        ll = l.rolling(n, min_periods=max(3, n // 2)).min()
        feats[f"pos_range_{n}"] = (c - ll) / (hh - ll + 1e-12)

    # --- RSI / Stochastic / Williams (10) ---
    for n in [6, 14, 28]:
        feats[f"rsi_{n}"] = _rsi(c, n)
    for n in [9, 14, 21]:
        feats[f"stoch_{n}"] = _stoch(h, l, c, n)
    for n in [14, 28]:
        hh = h.rolling(n, min_periods=max(3, n // 2)).max()
        feats[f"willr_{n}"] = -100 * (hh - c) / (hh - l.rolling(n, min_periods=max(3, n // 2)).min() + 1e-12)
    feats["rsi_div"] = feats["rsi_6"] - feats["rsi_28"]

    # --- MACD family (6) ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feats["macd"] = macd
    feats["macd_signal"] = signal
    feats["macd_hist"] = macd - signal
    feats["macd_hist_slope"] = feats["macd_hist"].diff(3)
    ema8 = c.ewm(span=8, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    feats["ema_spread_8_21"] = (ema8 - ema21) / c
    feats["price_vs_ema48"] = c / c.ewm(span=48, adjust=False).mean() - 1

    # --- Bollinger / Keltner style (8) ---
    for n, k in [(20, 2.0), (48, 2.0)]:
        m = c.rolling(n, min_periods=max(3, n // 2)).mean()
        s = c.rolling(n, min_periods=max(3, n // 2)).std()
        feats[f"bb_pctb_{n}"] = (c - (m - k * s)) / (2 * k * s + 1e-12)
        feats[f"bb_bw_{n}"] = (2 * k * s) / (m + 1e-12)
    atr14 = _atr(h, l, c, 14)
    mid = c.ewm(span=20, adjust=False).mean()
    feats["keltner_pos"] = (c - mid) / (atr14 + 1e-12)
    feats["close_location"] = (c - l) / (h - l + 1e-12)
    feats["hl_range_pct"] = (h - l) / (c + 1e-12)
    feats["body_pct"] = (c - o) / (c + 1e-12)

    # --- volatility (10) ---
    r1 = logc.diff()
    for n in [6, 12, 24, 48, 96]:
        feats[f"rvol_{n}"] = r1.rolling(n, min_periods=max(3, n // 2)).std()
    # Parkinson
    for n in [12, 48]:
        feats[f"parkinson_{n}"] = np.sqrt(
            ((np.log(h / l)) ** 2).rolling(n, min_periods=max(3, n // 2)).mean() / (4 * np.log(2))
        )
    feats["vol_of_vol"] = feats["rvol_12"].rolling(48, min_periods=12).std()
    feats["atr_14"] = atr14 / c
    feats["atr_ratio"] = atr14 / (_atr(h, l, c, 48) + 1e-12)

    # --- volume / activity (12) ---
    for n in [6, 12, 24, 48, 96]:
        vm = v.rolling(n, min_periods=max(3, n // 2)).mean()
        vs = v.rolling(n, min_periods=max(3, n // 2)).std()
        feats[f"vol_z_{n}"] = (v - vm) / (vs + 1e-12)
    feats["vol_roc_12"] = v.pct_change(12)
    feats["vol_roc_48"] = v.pct_change(48)
    # OBV slope
    obv = (np.sign(c.diff().fillna(0)) * v).cumsum()
    feats["obv_slope_12"] = obv.diff(12) / (v.rolling(12).mean() + 1e-12)
    feats["obv_slope_48"] = obv.diff(48) / (v.rolling(48).mean() + 1e-12)
    # volume-price trend
    feats["vpt"] = (v * c.pct_change()).fillna(0).cumsum()
    feats["vpt_slope_12"] = feats["vpt"].diff(12)

    # --- signed volume / order-flow proxies (arXiv 2108.09750 style) (10) ---
    # Bar-level trade imbalance proxy from candle body + taker buy if available
    if "taker_buy_base" in df.columns:
        tb = df["taker_buy_base"].astype(float)
        sell_v = (v - tb).clip(lower=0)
        feats["tfi"] = (tb - sell_v) / (v + 1e-12)
        feats["tfi_abs"] = tb - sell_v
        for n in [3, 6, 12, 24, 48]:
            b = tb.rolling(n, min_periods=1).sum()
            s = sell_v.rolling(n, min_periods=1).sum()
            feats[f"tfi_{n}"] = (b - s) / (b + s + 1e-12)
            feats[f"tfi_usd_{n}"] = b - s
    else:
        # tick-rule proxy: sign of return * volume
        sgn = np.sign(c.diff()).replace(0, np.nan).ffill().fillna(0)
        signed = sgn * v
        feats["tfi"] = signed / (v + 1e-12)
        feats["tfi_abs"] = signed
        for n in [3, 6, 12, 24, 48]:
            s = signed.rolling(n, min_periods=1).sum()
            tot = v.rolling(n, min_periods=1).sum()
            feats[f"tfi_{n}"] = s / (tot + 1e-12)
            feats[f"tfi_usd_{n}"] = s

    # buy pressure from candle
    feats["buy_pressure"] = ((c - l) - (h - c)) / (h - l + 1e-12)
    feats["upper_wick"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / (h - l + 1e-12)
    feats["lower_wick"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / (h - l + 1e-12)

    # --- liquidity / impact (6) ---
    # Amihud illiquidity
    feats["amihud_12"] = (r1.abs() / (v + 1e-12)).rolling(12, min_periods=3).mean()
    feats["amihud_48"] = (r1.abs() / (v + 1e-12)).rolling(48, min_periods=6).mean()
    if "n_trades" in df.columns:
        nt = df["n_trades"].astype(float)
        feats["trade_intensity"] = nt / (v + 1e-12)
        feats["trade_z_24"] = (nt - nt.rolling(24).mean()) / (nt.rolling(24).std() + 1e-12)
    else:
        feats["trade_intensity"] = v / (h - l + 1e-12)
        feats["trade_z_24"] = feats["vol_z_24"]
    feats["kyle_proxy"] = r1.abs() / (np.sqrt(v) + 1e-12)
    feats["spread_proxy"] = (h - l) / c

    # --- autocorrelation / persistence (6) ---
    # Fast lag-1 corr via rolling cov/var (no Python apply)
    r_lag = r1.shift(1)
    for n in [12, 24, 48]:
        cov = r1.rolling(n, min_periods=max(5, n // 2)).cov(r_lag)
        var = r1.rolling(n, min_periods=max(5, n // 2)).var()
        feats[f"ac1_{n}"] = cov / (var + 1e-12)
    sgn = np.sign(r1)
    feats["sign_persist_12"] = (sgn == sgn.shift(1)).astype(float).rolling(12, min_periods=3).mean()
    feats["up_ratio_24"] = (r1 > 0).astype(float).rolling(24, min_periods=6).mean()
    feats["up_ratio_96"] = (r1 > 0).astype(float).rolling(96, min_periods=12).mean()

    # --- regime / clock features (known at t) (6) ---
    feats["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    feats["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    feats["minute_slot"] = df.index.minute  # 0,5,...,55
    feats["is_weekend"] = (df.index.dayofweek >= 5).astype(float)

    # --- interaction features (6) ---
    feats["mom_x_volz"] = feats["ret_12"] * feats["vol_z_12"]
    feats["zprice_x_tfi"] = feats["zprice_24"] * feats["tfi_12"]
    feats["mr_signal"] = -feats["zprice_24"] * (feats["rvol_12"] / (feats["rvol_48"] + 1e-12))
    feats["breakout"] = feats["ret_6"] * (feats["vol_z_12"].clip(lower=0))
    feats["exhaustion"] = feats["ret_12"] * feats["rsi_14"].sub(50) / 50
    feats["vol_expand_mom"] = feats["ret_12"] * (feats["bb_bw_20"] / (feats["bb_bw_20"].rolling(48).mean() + 1e-12))

    # clean + defragment
    feats = feats.replace([np.inf, -np.inf], np.nan).copy()
    # causal: features at bar t use data ≤ t; label is sign(close[t+dt]-close[t]).
    return feats


def count_indicators(feats: pd.DataFrame) -> int:
    return feats.shape[1]
