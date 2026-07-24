"""
In-memory SQLite store for BTC OHLCV + backtest series.

Load once from parquet → RAM DB → fast windowed queries for the web chart.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


class MemoryDB:
    """Process-local SQLite :memory: database (shared via reference)."""

    def __init__(self) -> None:
        # check_same_thread=False so HTTP worker threads can read
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=OFF")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.stats: dict = {}
        self.loaded = False

    def load(self, root: Path | None = None) -> dict:
        root = root or ROOT
        preds_path = root / "predictions.parquet"
        bars_path = root / "btcusdt_1m.parquet"
        if not preds_path.exists() or not bars_path.exists():
            raise FileNotFoundError(
                f"Need {preds_path.name} and {bars_path.name} in {root}"
            )

        print("[MemoryDB] reading parquet…")
        preds = pd.read_parquet(preds_path)
        raw = pd.read_parquet(bars_path)
        df5 = (
            raw.resample("5min", label="left", closed="left")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["close"])
        )
        df = preds.join(df5[["open", "high", "low", "close", "volume"]], how="inner")
        dcol = (
            "ensemble_2of3"
            if "ensemble_2of3" in df.columns
            else [c for c in preds.columns if c != "y"][0]
        )
        decision = df[dcol].astype(int).to_numpy()
        y = df["y"].astype(int).to_numpy()
        bet = decision != 0
        valid = np.isin(y, [-1, 1])
        step = np.zeros(len(df), dtype=np.int8)
        step[bet & valid & (decision == y)] = 1
        step[bet & valid & (decision != y)] = -1
        cum = np.cumsum(step).astype(np.int32)

        # 5m full-resolution table
        t_ms = (df.index.asi8 // 10**6).astype(np.int64)
        bars5 = pd.DataFrame(
            {
                "t": t_ms,
                "o": df["open"].to_numpy(float),
                "h": df["high"].to_numpy(float),
                "l": df["low"].to_numpy(float),
                "c": df["close"].to_numpy(float),
                "v": df["volume"].to_numpy(float),
                "decision": decision.astype(np.int8),
                "y": y.astype(np.int8),
                "step": step,
                "cum": cum,
            }
        )

        # 15m display table (overview)
        g = (
            bars5.assign(ts=pd.to_datetime(bars5["t"], unit="ms"))
            .set_index("ts")
            .resample("15min", label="left", closed="left")
            .agg(
                {
                    "t": "first",
                    "o": "first",
                    "h": "max",
                    "l": "min",
                    "c": "last",
                    "v": "sum",
                    "step": "sum",
                    "cum": "last",
                }
            )
            .dropna(subset=["c"])
        )
        g["step"] = np.sign(g["step"].to_numpy()).astype(np.int8)
        g["t"] = (g.index.asi8 // 10**6).astype(np.int64)

        print("[MemoryDB] loading into SQLite :memory: …")
        cur = self.conn.cursor()
        cur.executescript(
            """
            DROP TABLE IF EXISTS bars_5m;
            DROP TABLE IF EXISTS bars_15m;
            DROP TABLE IF EXISTS meta;
            CREATE TABLE bars_5m (
              t INTEGER PRIMARY KEY,
              o REAL, h REAL, l REAL, c REAL, v REAL,
              decision INTEGER, y INTEGER, step INTEGER, cum INTEGER
            );
            CREATE TABLE bars_15m (
              t INTEGER PRIMARY KEY,
              o REAL, h REAL, l REAL, c REAL, v REAL,
              step INTEGER, cum INTEGER
            );
            CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
            CREATE INDEX IF NOT EXISTS idx_5m_t ON bars_5m(t);
            CREATE INDEX IF NOT EXISTS idx_15m_t ON bars_15m(t);
            """
        )

        cur.executemany(
            "INSERT INTO bars_5m VALUES (?,?,?,?,?,?,?,?,?,?)",
            bars5[
                ["t", "o", "h", "l", "c", "v", "decision", "y", "step", "cum"]
            ].itertuples(index=False, name=None),
        )
        cur.executemany(
            "INSERT INTO bars_15m VALUES (?,?,?,?,?,?,?,?)",
            g[["t", "o", "h", "l", "c", "v", "step", "cum"]].itertuples(
                index=False, name=None
            ),
        )

        acted = int((step != 0).sum())
        n_c = int((step == 1).sum())
        n_w = int((step == -1).sum())
        self.stats = {
            "acted": acted,
            "correct": n_c,
            "wrong": n_w,
            "hit_rate": n_c / acted if acted else 0.0,
            "final_cum": int(cum[-1]),
            "n_5m": int(len(bars5)),
            "n_15m": int(len(g)),
            "t0": int(t_ms[0]),
            "t1": int(t_ms[-1]),
            "model": "2of3: intensity_fade + logistic_wf + bollinger",
            "dt": "5m",
        }
        for k, v in self.stats.items():
            cur.execute(
                "INSERT INTO meta(k,v) VALUES (?,?)",
                (k, str(v)),
            )
        self.conn.commit()
        self.loaded = True
        print(
            f"[MemoryDB] ready  5m={self.stats['n_5m']:,}  15m={self.stats['n_15m']:,}  "
            f"cum={self.stats['final_cum']:+d}"
        )
        return self.stats

    def window(
        self,
        t0: int | None = None,
        t1: int | None = None,
        resolution: str = "auto",
    ) -> dict:
        """
        Return OHLCV + hits + cum for [t0, t1] (epoch ms).

        resolution:
          - '5m'  full detail
          - '15m' overview
          - 'auto' pick 5m if span <= 3 days else 15m
        """
        if not self.loaded:
            self.load()
        st = self.stats
        if t0 is None:
            t0 = st["t0"]
        if t1 is None:
            t1 = st["t1"]
        t0 = max(int(t0), st["t0"])
        t1 = min(int(t1), st["t1"])
        if t1 <= t0:
            t0, t1 = st["t0"], st["t1"]

        span_h = (t1 - t0) / 3_600_000
        if resolution == "auto":
            # fine grid when zoomed in (hour–3d); overview otherwise
            table = "bars_5m" if span_h <= 72 else "bars_15m"
        elif resolution == "5m":
            table = "bars_5m"
        else:
            table = "bars_15m"

        cols = "t,o,h,l,c,v,step,cum"
        rows = self.conn.execute(
            f"SELECT {cols} FROM {table} WHERE t>=? AND t<=? ORDER BY t",
            (t0, t1),
        ).fetchall()

        if not rows:
            return {
                "t": [],
                "o": [],
                "h": [],
                "l": [],
                "c": [],
                "v": [],
                "s": [],
                "cum": [],
                "meta": {
                    "t0": t0,
                    "t1": t1,
                    "table": table,
                    "n": 0,
                    "price_min": None,
                    "price_max": None,
                    "vol_max": None,
                    "cum_min": None,
                    "cum_max": None,
                },
            }

        t, o, h, l, c, v, s, cum = [], [], [], [], [], [], [], []
        pmin, pmax = 1e18, -1e18
        vmax = 0.0
        cmin, cmax = 1e18, -1e18
        for r in rows:
            t.append(int(r["t"]))
            o.append(round(float(r["o"]), 2))
            h.append(round(float(r["h"]), 2))
            l.append(round(float(r["l"]), 2))
            c.append(round(float(r["c"]), 2))
            v.append(round(float(r["v"]), 3))
            s.append(int(r["step"]))
            cum.append(int(r["cum"]))
            if r["l"] < pmin:
                pmin = float(r["l"])
            if r["h"] > pmax:
                pmax = float(r["h"])
            if r["v"] > vmax:
                vmax = float(r["v"])
            if r["cum"] < cmin:
                cmin = float(r["cum"])
            if r["cum"] > cmax:
                cmax = float(r["cum"])

        # 5% pad for auto y-scale (no blank)
        pp = (pmax - pmin) * 0.05 or 1.0
        cp = (cmax - cmin) * 0.06 or 1.0
        return {
            "t": t,
            "o": o,
            "h": h,
            "l": l,
            "c": c,
            "v": v,
            "s": s,
            "cum": cum,
            "meta": {
                "t0": t0,
                "t1": t1,
                "data_t0": st["t0"],
                "data_t1": st["t1"],
                "table": table,
                "n": len(t),
                "price_min": pmin - pp,
                "price_max": pmax + pp,
                "vol_max": vmax * 1.08 if vmax else 1.0,
                "cum_min": cmin - cp,
                "cum_max": cmax + cp,
            },
        }

    def get_stats(self) -> dict:
        if not self.loaded:
            self.load()
        return dict(self.stats)


# Singleton for the server process
DB = MemoryDB()
