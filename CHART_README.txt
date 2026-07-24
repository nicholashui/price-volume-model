# Chart (in-memory DB)

## Fast way (recommended)

```bash
cd C:\Project\price-volume-model
python serve_chart.py
```

Opens: http://127.0.0.1:18765/chart_app.html

### What it does
1. Loads `btcusdt_1m.parquet` + `predictions.parquet` **once** into **SQLite :memory:**
2. Serves only the **visible time window** from RAM (not a 7–15MB HTML dump)
3. Auto resolution: **5m** when zoomed in (≤3d), **15m** for long ranges
4. Auto **price min/max** (and vol/cum) for the window — no blank margins

### API
- `GET /api/stats`
- `GET /api/window?t0=<ms>&t1=<ms>&res=auto|5m|15m`
- `GET /api/health`
