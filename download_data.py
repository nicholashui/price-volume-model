"""Download ~1 year of BTCUSDT 1-minute OHLCV from Binance public API."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

# All artifacts live in the project / current directory
ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "btcusdt_1m.parquet"
BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000  # max per request


def _ms(dt: pd.Timestamp) -> int:
    return int(dt.timestamp() * 1000)


def fetch_klines(start_ms: int, end_ms: int) -> list:
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": LIMIT,
    }
    for attempt in range(5):
        try:
            r = requests.get(BINANCE_URL, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return []


def download_one_year(end: pd.Timestamp | None = None) -> pd.DataFrame:
    if end is None:
        end = pd.Timestamp.utcnow().floor("min")
    if end.tzinfo is not None:
        end = end.tz_localize(None)
    start = end - pd.Timedelta(days=365)

    rows: list = []
    cursor = _ms(start)
    end_ms = _ms(end)
    print(f"Downloading BTCUSDT 1m from {start} to {end} ...")

    while cursor < end_ms:
        batch = fetch_klines(cursor, end_ms)
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        next_cursor = last_open + 60_000
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(rows) % 20_000 < LIMIT:
            print(f"  bars: {len(rows):,}  last={pd.to_datetime(last_open, unit='ms')}")
        time.sleep(0.12)

    if not rows:
        raise RuntimeError("No data downloaded from Binance")

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "n_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]:
        df[c] = df[c].astype(float)
    df["n_trades"] = df["n_trades"].astype(int)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.drop_duplicates(subset=["open_time"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    df = df[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "n_trades",
            "taker_buy_base",
            "taker_buy_quote",
        ]
    ]
    df.to_parquet(OUT_PATH)
    print(f"Saved {len(df):,} bars -> {OUT_PATH}")
    return df


def load_or_download() -> pd.DataFrame:
    if OUT_PATH.exists():
        df = pd.read_parquet(OUT_PATH)
        print(f"Loaded cached data: {len(df):,} bars  [{df.index.min()} .. {df.index.max()}]")
        return df
    return download_one_year()


if __name__ == "__main__":
    download_one_year()
