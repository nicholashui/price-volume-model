"""
Reliable interactive chart — fixed blank page issues:
- valid auto-fit JS (no broken braces)
- SVG scatter (not WebGL scattergl — WebGL often = blank)
- lighter 15m series for full-year navigation
- CDN plotly + local fallback message
- auto min/max price & time on zoom
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORK = Path(
    r"C:\Users\NH24831\.grok\worktrees\project-price-volume-model\2026-07-24-04d2d6ec"
)


def load_frame() -> pd.DataFrame:
    preds = pd.read_parquet(ROOT / "predictions.parquet")
    raw = pd.read_parquet(ROOT / "btcusdt_1m.parquet")
    # 5m OHLC
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

    out = pd.DataFrame(
        {
            "open": df["open"].to_numpy(float),
            "high": df["high"].to_numpy(float),
            "low": df["low"].to_numpy(float),
            "close": df["close"].to_numpy(float),
            "volume": df["volume"].to_numpy(float),
            "step": step,
            "cum": np.cumsum(step).astype(np.int32),
        },
        index=df.index,
    )
    return out


def to_payload(df: pd.DataFrame) -> dict:
    """
    Compact arrays for the browser.

    Use 15-minute bars for the full-year interactive chart so the page
    stays responsive (5m x 1y ≈ 105k points freezes many browsers → blank).
    Hits are re-scored on the 15m grid from underlying 5m decisions.
    """
    acted = int((df["step"] != 0).sum())
    nc = int((df["step"] == 1).sum())
    nw = int((df["step"] == -1).sum())

    # Aggregate 5m → 15m for display
    g = df.resample("15min", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "step": "sum",  # net hit contribution in the bin
            "cum": "last",
        }
    ).dropna(subset=["close"])

    # Per-bin hit marker: sign of net step if non-zero
    step15 = np.sign(g["step"].to_numpy()).astype(int)

    t = (g.index.asi8 // 10**6).astype(np.int64)
    return {
        "t": t.tolist(),
        "o": np.round(g["open"].to_numpy(), 2).tolist(),
        "h": np.round(g["high"].to_numpy(), 2).tolist(),
        "l": np.round(g["low"].to_numpy(), 2).tolist(),
        "c": np.round(g["close"].to_numpy(), 2).tolist(),
        "v": np.round(g["volume"].to_numpy(), 3).tolist(),
        "s": step15.tolist(),
        "cum": g["cum"].astype(int).tolist(),
        "stats": {
            "acted": acted,
            "correct": nc,
            "wrong": nw,
            "hit_rate": nc / acted if acted else 0.0,
            "final_cum": int(df["cum"].iloc[-1]),
            "n": int(len(g)),
            "n_source_5m": int(len(df)),
            "bar": "15m (display) / 5m backtest",
            "t0": int(t[0]),
            "t1": int(t[-1]),
        },
    }


def build_html(data: dict) -> str:
    # Embed data; escape </script>
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")

    # Fully offline Plotly (no CDN — blank page if offline otherwise)
    try:
        from plotly.offline import get_plotlyjs

        # Prevent early script termination if library contains the closing tag sequence
        plotly_js = get_plotlyjs().replace("</script>", "<\\/script>")
    except Exception:
        plotly_js = ""

    plotly_tag = (
        f'<script type="text/javascript">\n{plotly_js}\n</script>'
        if plotly_js
        else """
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.27.0/plotly.min.js"></script>
"""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BTCUSDT Price + Backtest</title>
<style>
  html,body{{margin:0;height:100%;background:#0b0e11;color:#eaecef;font-family:Segoe UI,Arial,sans-serif}}
  #msg{{padding:16px;font-size:14px;color:#f0b90b}}
  #err{{display:none;margin:12px;padding:12px;background:#2a0f14;border:1px solid #f6465d;
        color:#ffc1c8;border-radius:6px;white-space:pre-wrap;font-size:13px}}
  header{{padding:10px 14px;border-bottom:1px solid #2b3139}}
  h1{{margin:0 0 4px;font-size:16px}}
  #sub{{font-size:12px;color:#848e9c}}
  .stats{{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}}
  .chip{{background:#161a1e;border:1px solid #2b3139;border-radius:4px;padding:3px 8px;font-size:12px;color:#848e9c}}
  .chip b{{color:#eaecef}}.g b{{color:#0ecb81}}.r b{{color:#f6465d}}
  .bar{{display:flex;flex-wrap:wrap;gap:6px;padding:8px 14px;border-bottom:1px solid #2b3139;background:#12161c;align-items:center}}
  .bar button{{background:#161a1e;color:#eaecef;border:1px solid #2b3139;border-radius:4px;padding:5px 10px;font-size:12px;cursor:pointer}}
  .bar button.active{{border-color:#f0b90b;color:#f0b90b}}
  .bar label{{font-size:12px;color:#848e9c}}
  #chart{{width:100%;height:calc(100vh - 145px);min-height:560px}}
  #status{{padding:4px 14px;font-size:11px;color:#848e9c}}
</style>
{plotly_tag}
</head>
<body>
<div id="msg">Loading chart… (offline Plotly embedded — please wait a few seconds)</div>
<div id="err"></div>
<header id="hdr" style="display:none">
  <h1>BTCUSDT price chart + direction backtest</h1>
  <div id="sub"></div>
  <div class="stats" id="stats"></div>
</header>
<div class="bar" id="toolbar" style="display:none">
  <label>Zoom</label>
  <button type="button" data-h="1">1H</button>
  <button type="button" data-h="4">4H</button>
  <button type="button" data-h="12">12H</button>
  <button type="button" data-h="24" class="active">1D</button>
  <button type="button" data-h="72">3D</button>
  <button type="button" data-h="168">1W</button>
  <button type="button" data-h="720">1M</button>
  <button type="button" data-h="0">ALL</button>
</div>
<div id="status"></div>
<div id="chart"></div>

<script>
window.CHART_DATA = {blob};
</script>
<script>
(function () {{
  var msg = document.getElementById("msg");
  var err = document.getElementById("err");
  function fail(e) {{
    console.error(e);
    err.style.display = "block";
    err.textContent = "Error: " + (e && e.stack ? e.stack : e);
    msg.textContent = "Chart failed to render.";
  }}
  window.addEventListener("error", function (ev) {{ fail(ev.error || ev.message); }});

  function ready(fn) {{
    if (typeof Plotly !== "undefined") return fn();
    var n = 0;
    var id = setInterval(function () {{
      n++;
      if (typeof Plotly !== "undefined") {{ clearInterval(id); fn(); }}
      else if (n > 50) {{
        clearInterval(id);
        fail("Plotly library did not load. Connect to the internet (needs cdn.plot.ly) and refresh.");
      }}
    }}, 100);
  }}

  ready(function () {{
    try {{ boot(window.CHART_DATA); }}
    catch (e) {{ fail(e); }}
  }});

  function boot(D) {{
    msg.style.display = "none";
    document.getElementById("hdr").style.display = "block";
    document.getElementById("toolbar").style.display = "flex";

    var S = D.stats;
    document.getElementById("sub").textContent =
      "ensemble 2of3 · dt=5m · bars=" + S.n.toLocaleString() +
      " · " + new Date(S.t0).toISOString().slice(0, 10) + " → " + new Date(S.t1).toISOString().slice(0, 10);

    document.getElementById("stats").innerHTML =
      '<span class="chip">acted <b>' + S.acted.toLocaleString() + "</b></span>" +
      '<span class="chip g">correct <b>' + S.correct.toLocaleString() + "</b></span>" +
      '<span class="chip r">wrong <b>' + S.wrong.toLocaleString() + "</b></span>" +
      '<span class="chip">hit_rate <b>' + (S.hit_rate * 100).toFixed(2) + "%</b></span>" +
      '<span class="chip g">cum_hit <b>' + (S.final_cum >= 0 ? "+" : "") + S.final_cum + "</b></span>";

    // Precompute Date objects once
    var T = new Array(D.t.length);
    for (var i = 0; i < D.t.length; i++) T[i] = new Date(D.t[i]);

    var iC = [], iW = [];
    for (var j = 0; j < D.s.length; j++) {{
      if (D.s[j] === 1) iC.push(j);
      else if (D.s[j] === -1) iW.push(j);
    }}
    function take(idx, arr) {{
      var o = new Array(idx.length);
      for (var k = 0; k < idx.length; k++) o[k] = arr[idx[k]];
      return o;
    }}
    var TC = take(iC, T), TW = take(iW, T);
    var CC = take(iC, D.c), CW = take(iW, D.c);
    var SC = take(iC, D.s), SW = take(iW, D.s);

    // Downsample volume for speed
    var Tv = [], Vv = [];
    for (var u = 0; u < T.length; u += 2) {{ Tv.push(T[u]); Vv.push(D.v[u]); }}

    var x0 = S.t1 - 24 * 3600 * 1000;
    if (x0 < S.t0) x0 = S.t0;
    var x1 = S.t1;
    var fitting = false;

    function yRangePrice(a, b) {{
      var lo = Infinity, hi = -Infinity, n = 0;
      for (var i = 0; i < D.t.length; i++) {{
        var ti = D.t[i];
        if (ti < a || ti > b) continue;
        if (D.l[i] < lo) lo = D.l[i];
        if (D.h[i] > hi) hi = D.h[i];
        n++;
      }}
      if (!n) return null;
      if (lo === hi) {{ lo -= 1; hi += 1; }}
      var pad = (hi - lo) * 0.05;
      return [lo - pad, hi + pad];
    }}
    function yRangeArr(arr, a, b, floor0) {{
      var lo = Infinity, hi = -Infinity, n = 0;
      for (var i = 0; i < D.t.length; i++) {{
        var ti = D.t[i];
        if (ti < a || ti > b) continue;
        var v = arr[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        n++;
      }}
      if (!n) return null;
      if (lo === hi) {{ lo -= 1; hi += 1; }}
      var pad = (hi - lo) * 0.06;
      if (floor0) return [Math.min(0, lo), hi + pad];
      return [lo - pad, hi + pad];
    }}

    function clampTime(a, b) {{
      a = Math.max(a, S.t0);
      b = Math.min(b, S.t1);
      if (b <= a) {{ a = S.t0; b = S.t1; }}
      return [a, b];
    }}

    function buildTraces() {{
      return [
        {{
          type: "scatter", mode: "lines", name: "BTC close (price)",
          x: T, y: D.c,
          line: {{ color: "#eaecef", width: 1.3 }},
          xaxis: "x", yaxis: "y",
          hovertemplate: "%{{x|%Y-%m-%d %H:%M}}<br>price=%{{y:.2f}}<extra></extra>"
        }},
        {{
          type: "scatter", mode: "markers", name: "hit +1",
          x: TC, y: CC,
          marker: {{ color: "#0ecb81", size: 5 }},
          xaxis: "x", yaxis: "y",
          hovertemplate: "%{{x|%Y-%m-%d %H:%M}}<br>price=%{{y:.2f}} · CORRECT<extra></extra>"
        }},
        {{
          type: "scatter", mode: "markers", name: "hit -1",
          x: TW, y: CW,
          marker: {{ color: "#f6465d", size: 5 }},
          xaxis: "x", yaxis: "y",
          hovertemplate: "%{{x|%Y-%m-%d %H:%M}}<br>price=%{{y:.2f}} · WRONG<extra></extra>"
        }},
        {{
          type: "scatter", mode: "lines", name: "Volume",
          x: Tv, y: Vv,
          line: {{ color: "rgba(240,185,11,0.85)", width: 1 }},
          fill: "tozeroy", fillcolor: "rgba(240,185,11,0.12)",
          xaxis: "x2", yaxis: "y2",
          hovertemplate: "%{{x|%Y-%m-%d %H:%M}}<br>vol=%{{y:.3f}}<extra></extra>"
        }},
        {{
          type: "scatter", mode: "markers", name: "+1", showlegend: false,
          x: TC, y: SC,
          marker: {{ color: "#0ecb81", size: 7, symbol: "line-ns", line: {{ width: 2, color: "#0ecb81" }} }},
          xaxis: "x3", yaxis: "y3",
          hovertemplate: "%{{x|%Y-%m-%d %H:%M}} · +1<extra></extra>"
        }},
        {{
          type: "scatter", mode: "markers", name: "-1", showlegend: false,
          x: TW, y: SW,
          marker: {{ color: "#f6465d", size: 7, symbol: "line-ns", line: {{ width: 2, color: "#f6465d" }} }},
          xaxis: "x3", yaxis: "y3",
          hovertemplate: "%{{x|%Y-%m-%d %H:%M}} · -1<extra></extra>"
        }},
        {{
          type: "scatter", mode: "lines", name: "accumulate hit",
          x: T, y: D.cum,
          line: {{ color: "#f0b90b", width: 1.6 }},
          xaxis: "x4", yaxis: "y4",
          hovertemplate: "%{{x|%Y-%m-%d %H:%M}}<br>cum=%{{y}}<extra></extra>"
        }}
      ];
    }}

    function layoutFor(a, b) {{
      var pr = yRangePrice(a, b) || [0, 1];
      var vr = yRangeArr(D.v, a, b, true) || [0, 1];
      var cr = yRangeArr(D.cum, a, b, false) || [0, 1];
      var ra = [new Date(a), new Date(b)];
      return {{
        paper_bgcolor: "#0b0e11",
        plot_bgcolor: "#0e1217",
        font: {{ color: "#848e9c", size: 11 }},
        margin: {{ l: 50, r: 50, t: 30, b: 40 }},
        showlegend: true,
        legend: {{ orientation: "h", y: 1.08, x: 0, bgcolor: "rgba(0,0,0,0)", font: {{ color: "#eaecef" }} }},
        hovermode: "closest",
        dragmode: "zoom",
        xaxis:  {{ domain: [0,1], anchor: "y",  type: "date", range: ra, matches: "x4", showticklabels: false, gridcolor: "#1e2329", rangeslider: {{ visible: false }} }},
        yaxis:  {{ domain: [0.54, 1.00], title: "Price", range: pr, autorange: false, side: "right", gridcolor: "#1e2329", fixedrange: true }},
        xaxis2: {{ domain: [0,1], anchor: "y2", type: "date", range: ra, matches: "x4", showticklabels: false, gridcolor: "#1e2329" }},
        yaxis2: {{ domain: [0.40, 0.51], title: "Vol", range: vr, autorange: false, side: "right", gridcolor: "#1e2329", fixedrange: true }},
        xaxis3: {{ domain: [0,1], anchor: "y3", type: "date", range: ra, matches: "x4", showticklabels: false, gridcolor: "#1e2329" }},
        yaxis3: {{ domain: [0.28, 0.38], title: "hit", range: [-1.4, 1.4], tickvals: [-1,0,1], side: "right", gridcolor: "#1e2329", fixedrange: true, zeroline: true }},
        xaxis4: {{
          domain: [0,1], anchor: "y4", type: "date", range: ra, title: "time (UTC)",
          gridcolor: "#1e2329",
          rangeslider: {{ visible: true, thickness: 0.06, bgcolor: "#0b0e11", bordercolor: "#2b3139" }}
        }},
        yaxis4: {{ domain: [0.0, 0.24], title: "cum", range: cr, autorange: false, side: "right", gridcolor: "#1e2329", fixedrange: true, zeroline: true }}
      }};
    }}

    var cfg = {{ responsive: true, displaylogo: false, scrollZoom: true, doubleClick: "reset" }};
    var status = document.getElementById("status");
    var ignoreRelayout = false;

    function paint(a, b) {{
      var ab = clampTime(a, b);
      x0 = ab[0]; x1 = ab[1];
      status.textContent = "Rendering…";
      ignoreRelayout = true;
      return Plotly.react("chart", buildTraces(), layoutFor(x0, x1), cfg)
        .then(function () {{
          status.textContent = "Ready — price auto min/max on zoom · drag to zoom · buttons 1H–ALL · no blank margins";
          // ignore synthetic relayout events from react
          setTimeout(function () {{ ignoreRelayout = false; }}, 150);
        }})
        .catch(function (e) {{
          ignoreRelayout = false;
          throw e;
        }});
    }}

    // Zoom buttons
    document.querySelectorAll(".bar button[data-h]").forEach(function (btn) {{
      btn.addEventListener("click", function () {{
        document.querySelectorAll(".bar button[data-h]").forEach(function (b) {{ b.classList.remove("active"); }});
        btn.classList.add("active");
        var h = parseInt(btn.getAttribute("data-h"), 10);
        if (!h) paint(S.t0, S.t1);
        else paint(S.t1 - h * 3600 * 1000, S.t1);
      }});
    }});

    // On user zoom/pan: re-fit y axes + clamp time (no blank area)
    var gd = document.getElementById("chart");
    gd.on("plotly_relayout", function (ev) {{
      if (fitting || ignoreRelayout) return;
      var a = ev["xaxis.range[0]"], b = ev["xaxis.range[1]"];
      if (a === undefined && ev["xaxis4.range[0]"] !== undefined) {{
        a = ev["xaxis4.range[0]"]; b = ev["xaxis4.range[1]"];
      }}
      if (a === undefined && ev["xaxis.range"] !== undefined) {{
        a = ev["xaxis.range"][0]; b = ev["xaxis.range"][1];
      }}
      if (ev["xaxis.autorange"] || ev["xaxis4.autorange"]) {{
        fitting = true;
        paint(S.t1 - 24 * 3600 * 1000, S.t1)
          .then(function () {{ fitting = false; }})
          .catch(function (e) {{ fitting = false; fail(e); }});
        return;
      }}
      if (a === undefined) return;
      var na = +new Date(a), nb = +new Date(b);
      if (!isFinite(na) || !isFinite(nb)) return;
      // skip tiny no-op updates
      if (Math.abs(na - x0) < 500 && Math.abs(nb - x1) < 500) return;
      fitting = true;
      paint(na, nb)
        .then(function () {{ fitting = false; }})
        .catch(function (e) {{ fitting = false; fail(e); }});
    }});

    // First paint: last 1 day
    paint(x0, x1).catch(fail);
  }}
}})();
</script>
</body>
</html>
"""


def main() -> None:
    print("Loading data…")
    df = load_frame()
    print(f"bars={len(df):,} cum={int(df['cum'].iloc[-1]):+d}")
    data = to_payload(df)
    html = build_html(data)

    names = [
        "price_chart_backtest.html",
        "direction_backtest_interactive.html",
        "interactive_chart.html",
        "index.html",
    ]
    for name in names:
        path = ROOT / name
        path.write_text(html, encoding="utf-8")
        mb = path.stat().st_size / 1024 / 1024
        print(f"saved {path.name} ({mb:.1f} MB)")

    # offline helper: embed plotly from local package if user has no network
    try:
        import plotly
        from pathlib import Path as P

        # optional offline bundle note
        (ROOT / "OPEN_ME.txt").write_text(
            "If the page is blank:\n"
            "1) You need internet once to load Plotly from CDN (cdn.plot.ly)\n"
            "2) Or run:  python serve_chart.py\n"
            "3) Prefer Chrome/Edge\n"
            "4) Wait a few seconds on first load (~6MB file)\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    serve = ROOT / "serve_chart.py"
    serve.write_text(
        """\
import http.server, socketserver, webbrowser, os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)
PORT = 8765
url = f"http://127.0.0.1:{PORT}/price_chart_backtest.html"
print("Serving", Path.cwd())
print("Open", url)
webbrowser.open(url)
with socketserver.TCPServer(("127.0.0.1", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
""",
        encoding="utf-8",
    )

    if WORK.exists():
        for name in names + ["serve_chart.py", "OPEN_ME.txt"]:
            src = ROOT / name
            if src.exists():
                (WORK / name).write_bytes(src.read_bytes())

    # sanity: no broken double-brace JS function syntax
    if "fitAll() {{" in html or "function boot(D) {{" not in html.replace("function boot(D) {", "OK"):
        # In the f-string output, JS should have single braces
        if "function boot(D) {{" in html:
            raise SystemExit("BUG: double braces leaked into HTML")
    if "function boot(D) {" not in html:
        # f-string converts {{ to { so we expect single
        pass
    print("JS boot present:", "function boot(D)" in html)
    print("\nOpen: price_chart_backtest.html")
    print("Or:   python serve_chart.py")


if __name__ == "__main__":
    main()
