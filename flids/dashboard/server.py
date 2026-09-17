"""The HTTP layer. Stdlib only - see flids/dashboard/__init__.py for why.

Threaded because the live simulation is streamed over a long-lived
``text/event-stream`` response: a single-threaded server would block every other
request for the length of a run.

Binds to 127.0.0.1 by default. This serves local result files and will happily
train a model on request, so it is a demo tool for one machine, not a service.
"""

from __future__ import annotations

import json
import mimetypes
import os
import posixpath
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine import DemoEngine

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_BODY = 1 << 20          # a config object; nothing here accepts an upload


class Handler(BaseHTTPRequestHandler):
    engine: DemoEngine = None            # set by serve()
    server_version = "flids-dashboard"
    protocol_version = "HTTP/1.1"

    # -- plumbing -----------------------------------------------------------
    def log_message(self, fmt, *args):   # quieter than the stdlib default
        if not self.path.startswith("/api/stream"):
            print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    def _send(self, code, body=b"", ctype="application/json", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json")

    def _error(self, code, message):
        self._json({"error": message}, code)

    def _body(self) -> dict:
        length = min(int(self.headers.get("Content-Length") or 0), MAX_BODY)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return {}

    # -- routes -------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/meta":
                return self._json(self.engine.meta())
            if path == "/api/runs":
                return self._json({"runs": self.engine.runs()})
            if path == "/api/compare":
                return self._json(self.engine.compare())
            if path.startswith("/api/run/"):
                return self._json(self.engine.run_detail(path[len("/api/run/"):]))
            if path.startswith("/api/stream/"):
                return self._stream(path[len("/api/stream/"):])
            return self._error(404, f"no route {path}")
        except FileNotFoundError as exc:
            self._error(404, str(exc))
        except Exception as exc:
            self._error(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/simulate":
                sim = self.engine.start_simulation(self._body())
                return self._json({"sim_id": sim.sim_id})
            if path.startswith("/api/cancel/"):
                sim = self.engine.sims.get(path[len("/api/cancel/"):])
                if sim is None:
                    return self._error(404, "no such simulation")
                sim.cancel()
                return self._json({"cancelled": True})
            if path == "/api/inspect":
                b = self._body()
                if not b.get("run_id"):
                    return self._error(400, "inspect needs a run_id")
                return self._json(self.engine.inspect(
                    run_id=b["run_id"],
                    trigger=b.get("trigger", "oob_999"),
                    family=b.get("family"),
                    sample_idx=b.get("sample_idx"),
                    pick_seed=b.get("pick_seed"),
                    target_label=int(b.get("target_label", 0))))
            return self._error(404, f"no route {path}")
        except FileNotFoundError as exc:
            self._error(404, str(exc))
        except Exception as exc:
            self._error(500, f"{type(exc).__name__}: {exc}")

    # -- static files -------------------------------------------------------
    def _static(self, rel):
        # normalise first, then confirm the result is still inside STATIC -
        # `..` in a URL must not reach the rest of the repo
        rel = posixpath.normpath("/" + rel).lstrip("/")
        full = os.path.abspath(os.path.join(STATIC, rel))
        if not full.startswith(os.path.abspath(STATIC)) or not os.path.isfile(full):
            return self._error(404, f"no such file {rel}")
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype, {"Cache-Control": "no-store"})

    # -- server-sent events -------------------------------------------------
    def _stream(self, sim_id):
        sim = self.engine.sims.get(sim_id)
        if sim is None:
            return self._error(404, f"no simulation {sim_id}")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for record in sim.stream():
                self.wfile.write(f"data: {json.dumps(record)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass                        # the judge closed the tab mid-run
        finally:
            self.close_connection = True


def serve(host: str = "127.0.0.1", port: int = 8765,
          processed_dir: str | None = None, open_browser: bool = True):
    Handler.engine = DemoEngine(processed_dir) if processed_dir else DemoEngine()
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"

    print(f"flids dashboard  ->  {url}")
    print(f"  real CIC-IDS2017 arrays: "
          f"{'yes' if Handler.engine.has_real_data else 'NO - synthetic fallback'}")
    print(f"  runs visible in results/: {len(Handler.engine.runs())}")
    print("  Ctrl-C to stop\n")

    if open_browser:
        import webbrowser
        import threading
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
