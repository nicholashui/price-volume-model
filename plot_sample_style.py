"""Generate 3-panel charts matching sample_chat.png style from ensemble backtest."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORK = Path(
    r"C:\Users\NH24831\.grok\worktrees\project-price-volume-model\2026-07-24-04d2d6ec"
)


def _prepare() -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    preds = pd.read_parquet(ROOT / "predictions.parquet")
    raw = pd.read_parquet(ROOT / "btcusdt_1m.parquet")
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
    df = preds.join(df5[["close"]], how="inner")
    dcol = "ensemble_2of3" if "ensemble_2of3" in df.columns else None
    if dcol is None:
        dcol = [c for c in df.columns if c not in ("y", "close")][0]

    decision = df[dcol].astype(int)
    y = df["y"].astype(int)
    close = df["close"].astype(float)

    bet = decision != 0
    valid = y.isin([-1, 1])
    correct = bet & valid & (decision.values == y.values)
    wrong = bet & valid & (decision.values != y.values)

    step = pd.Series(0.0, index=df.index)
    step.iloc[np.where(np.asarray(correct))[0]] = 1.0
    step.iloc[np.where(np.asarray(wrong))[0]] = -1.0
    cum = step.cumsum()

    acted = int(bet.sum())
    n_correct = int(np.sum(correct))
    n_wrong = int(np.sum(wrong))
    stats = {
        "acted": acted,
        "correct": n_correct,
        "wrong": n_wrong,
        "hit_rate": n_correct / acted if acted else 0.0,
        "final_cum": float(cum.iloc[-1]) if len(cum) else 0.0,
        "dcol": dcol,
    }
    return close, step, cum, stats


def plot_window(
    close: pd.Series,
    step: pd.Series,
    cum: pd.Series,
    stats: dict,
    start=None,
    end=None,
    out_name: str = "direction_backtest_chart.png",
    title_extra: str = "",
    marker_size: float = 14,
    stem_lw: float = 0.7,
) -> Path:
    is_window = start is not None or end is not None
    c = close.loc[start:end] if is_window else close
    s = step.reindex(c.index).fillna(0.0)
    # Window charts: local accumulate hit (sample_chat style).
    # Full-year chart: global cumulative from series start.
    if is_window:
        cu = s.cumsum()
        final_cum = float(cu.iloc[-1]) if len(cu) else 0.0
        global_cum = float(cum.reindex(c.index).ffill().iloc[-1]) if len(c) else 0.0
    else:
        cu = cum.reindex(c.index).ffill()
        final_cum = float(cu.iloc[-1]) if len(cu) else 0.0
        global_cum = final_cum

    # window-local stats
    acted_w = int((s != 0).sum())
    n_c = int((s == 1).sum())
    n_w = int((s == -1).sum())
    hr = n_c / acted_w if acted_w else 0.0

    title = (
        f"BTCUSDT direction backtest  dt=5m  bar=5m  ensemble=2of3  "
        f"cum_hit={final_cum:+.0f}{title_extra}"
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 0.9, 1.1]},
    )

    # --- panel 1: price path colored by hit, black line underneath ---
    ax = axes[0]
    ax.plot(c.index, c.values, color="0.35", lw=0.9, zorder=1, label="BTC close (5m)")

    idx_c = c.index[s.values == 1]
    idx_w = c.index[s.values == -1]
    ax.scatter(
        idx_w,
        c.loc[idx_w],
        s=marker_size,
        c="red",
        label="hit -1",
        zorder=2,
        linewidths=0,
        alpha=0.9,
    )
    ax.scatter(
        idx_c,
        c.loc[idx_c],
        s=marker_size,
        c="limegreen",
        label="hit +1",
        zorder=3,
        linewidths=0,
        alpha=0.9,
    )
    # redraw price lightly on top so structure remains visible
    ax.plot(c.index, c.values, color="black", lw=0.45, alpha=0.55, zorder=4)

    ax.set_ylabel("S (USDT)")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    if is_window:
        info = (
            f"acted={acted_w}  correct={n_c}  wrong={n_w}\n"
            f"hit_rate={hr:.3f}  window_cum={final_cum:+.0f}  year_cum={global_cum:+.0f}\n"
            f"dt=5m  models=intensity_fade+logistic_wf+bollinger  vote=2of3"
        )
    else:
        info = (
            f"acted={acted_w}  correct={n_c}  wrong={n_w}\n"
            f"hit_rate={hr:.3f}  final_cum={final_cum:+.0f}\n"
            f"dt=5m  models=intensity_fade+logistic_wf+bollinger  vote=2of3"
        )
    ax.text(
        0.99,
        0.03,
        info,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="steelblue",
            alpha=0.92,
        ),
    )

    # --- panel 2: hit stems (sample style) ---
    ax2 = axes[1]
    t_c = s.index[s.values == 1]
    t_w = s.index[s.values == -1]
    ax2.vlines(t_c, 0, 1, colors="limegreen", lw=stem_lw, alpha=0.85)
    ax2.vlines(t_w, 0, -1, colors="red", lw=stem_lw, alpha=0.85)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_ylabel("hit")
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["-1 wrong", "0 none", "+1 correct"])
    ax2.set_ylim(-1.35, 1.35)
    ax2.grid(True, alpha=0.25)

    # --- panel 3: accumulate hit ---
    ax3 = axes[2]
    ax3.plot(cu.index, cu.values, color="royalblue", lw=1.15, label="accumulate hit")
    ax3.axhline(0, color="gray", lw=0.8)
    ax3.set_ylabel("cum hit")
    ax3.set_xlabel("time (UTC)")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.25)

    fig.tight_layout()
    out = ROOT / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if WORK.exists():
        fig.savefig(WORK / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}  (window acted={acted_w} cum={final_cum:+.0f})")
    return out


def main() -> None:
    close, step, cum, stats = _prepare()
    print(
        "full series:",
        f"acted={stats['acted']} correct={stats['correct']} wrong={stats['wrong']} "
        f"hit_rate={stats['hit_rate']:.4f} cum={stats['final_cum']:+.0f}",
    )

    # 1) Full-year chart (smaller markers)
    plot_window(
        close,
        step,
        cum,
        stats,
        out_name="direction_backtest_chart.png",
        title_extra="  [full year]",
        marker_size=6,
        stem_lw=0.25,
    )

    # 2) Sample-like recent window (~3 weeks) — readable density like sample_chat.png
    end = close.index.max()
    start = end - pd.Timedelta(days=21)
    plot_window(
        close,
        step,
        cum,
        stats,
        start=start,
        end=end,
        out_name="sample_style_chart.png",
        title_extra="  [last 21d]",
        marker_size=16,
        stem_lw=0.85,
    )

    # also alias name user asked for
    plot_window(
        close,
        step,
        cum,
        stats,
        start=start,
        end=end,
        out_name="direction_chat.png",
        title_extra="  [last 21d]",
        marker_size=16,
        stem_lw=0.85,
    )


if __name__ == "__main__":
    main()
