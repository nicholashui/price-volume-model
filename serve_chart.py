"""
Fast chart server backed by an in-memory SQLite DB.

  python serve_chart.py

Opens http://127.0.0.1:8765/chart_app.html

API:
  GET /api/stats
  GET /api/window?t0=&t1=&res=auto|5m|15m
"""

from __future__ import annotations

import json
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from memory_db import DB

ROOT = Path(__file__).resolve().parent
PORT = 18765
HOST = "127.0.0.1"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        # quieter
        if args and str(args[0]).startswith("GET /api/"):
            print("[API]", args[0])
        else:
            super().log_message(fmt, *args)

    def _json(self, obj, code=200):
        body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        path = u.path

        if path in ("/", "/index.html"):
            self.path = "/chart_app.html"
            return super().do_GET()

        if path == "/api/stats":
            try:
                self._json(DB.get_stats())
            except Exception as e:
                self._json({"error": str(e)}, 500)
            return

        if path == "/api/window":
            qs = parse_qs(u.query)
            try:
                t0 = int(qs["t0"][0]) if "t0" in qs else None
                t1 = int(qs["t1"][0]) if "t1" in qs else None
                res = qs.get("res", ["auto"])[0]
                self._json(DB.window(t0, t1, resolution=res))
            except Exception as e:
                self._json({"error": str(e)}, 500)
            return

        if path == "/api/health":
            self._json({"ok": True, "loaded": DB.loaded, "stats": DB.stats if DB.loaded else {}})
            return

        return super().do_GET()


def main():
    print("=" * 56)
    print(" BTCUSDT chart server  ·  in-memory SQLite")
    print("=" * 56)
    DB.load(ROOT)
    url = f"http://{HOST}:{PORT}/chart_app.html"
    print(f"Serving {ROOT}")
    print(f"Open    {url}")
    print("API     /api/stats  /api/window?t0=&t1=&res=auto")
    print("Ctrl+C  to stop")
    print("=" * 56)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
