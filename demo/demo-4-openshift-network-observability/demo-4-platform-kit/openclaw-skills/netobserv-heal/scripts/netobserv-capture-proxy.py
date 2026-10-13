#!/usr/bin/env python3
"""HTTP front-end for in-cluster NetObserv flow capture."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

CAPTURE_PY = os.environ.get(
    "NETOBSERV_CAPTURE_PY", "/opt/capture/netobserv-cluster-capture.py"
)
OUTPUT = Path(os.environ.get("CAPTURE_OUTPUT", "/data/latest-evidence.json"))
HOST = os.environ.get("NETOBSERV_CAPTURE_BIND", "0.0.0.0")
PORT = int(os.environ.get("NETOBSERV_CAPTURE_PORT", "8080"))
MAX_DURATION = int(os.environ.get("CAPTURE_MAX_DURATION", "120"))


def run_capture(duration: int, burst: bool) -> tuple[int, str]:
    env = os.environ.copy()
    env["NETOBSERV_IN_CLUSTER"] = "1"
    env["CAPTURE_OUTPUT"] = str(OUTPUT)
    env["CAPTURE_BURST"] = "1" if burst else "0"
    env["CAPTURE_DURATION"] = str(duration)
    proc = subprocess.run(
        [sys.executable, CAPTURE_PY, "capture", str(duration)],
        capture_output=True,
        text=True,
        env=env,
        timeout=MAX_DURATION + 420,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: str, content_type: str = "text/plain") -> None:
        data = body.encode("utf-8", errors="replace")
        self.send_response(code)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/healthz":
            self._send(200, "ok\n")
            return
        if path == "/latest":
            if not OUTPUT.is_file():
                self._send(
                    404,
                    "no capture yet — POST /capture first\n",
                )
                return
            self._send(200, OUTPUT.read_text(), "application/json")
            return
        if path == "/capture":
            self._handle_capture()
            return
        self._send(404, "try GET /healthz /latest or GET|POST /capture?duration=60\n")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/capture":
            self._handle_capture()
            return
        self.do_GET()

    def _handle_capture(self) -> None:
        qs = parse_qs(urlparse(self.path).query)
        try:
            duration = int((qs.get("duration") or ["60"])[0])
        except ValueError:
            duration = 60
        duration = max(30, min(duration, MAX_DURATION))
        burst = (qs.get("burst") or ["1"])[0] not in ("0", "false", "no")
        try:
            rc, out = run_capture(duration, burst)
        except Exception:
            self._send(500, traceback.format_exc())
            return
        if rc != 0:
            self._send(500, out if out.endswith("\n") else out + "\n")
            return
        summary = out
        if OUTPUT.is_file():
            try:
                ev = json.loads(OUTPUT.read_text())
                net = ev.get("network_evidence") or {}
                rtt = (net.get("rtt_ms") or {}).get("avg")
                summary += (
                    f"\nEVIDENCE_READY: true\n"
                    f"total_db_flows={net.get('total_db_flows')}\n"
                    f"avg_rtt_ms={rtt}\n"
                    f"NEXT: call MCP netobserv_analyze_evidence (do not read summarize-evidence.py)\n"
                )
            except json.JSONDecodeError:
                summary += "\nEVIDENCE_READY: true (parse warning)\n"
        self._send(200, summary if summary.endswith("\n") else summary + "\n")


def main() -> int:
    if not Path(CAPTURE_PY).is_file():
        print(f"missing {CAPTURE_PY}", file=sys.stderr)
        return 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"netobserv-capture-proxy listening on {HOST}:{PORT}", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
