#!/usr/bin/env python3
"""Tiny HTTP front-end for netobserv-cluster-heal.py (stdlib only)."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HEAL_PY = os.environ.get(
    "NETOBSERV_HEAL_PY", "/opt/heal/netobserv-cluster-heal.py"
)
HOST = os.environ.get("NETOBSERV_HEAL_BIND", "0.0.0.0")
PORT = int(os.environ.get("NETOBSERV_HEAL_PORT", "8080"))


def run_cmd(cmd: str) -> tuple[int, str]:
    env = os.environ.copy()
    env["NETOBSERV_IN_CLUSTER"] = "1"
    proc = subprocess.run(
        [sys.executable, HEAL_PY, cmd],
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # quieter
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: str) -> None:
        data = body.encode("utf-8", errors="replace")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/healthz":
            self._send(200, "ok\n")
            return
        if path in ("/heal", "/probe", "/status"):
            cmd = path.lstrip("/")
            try:
                rc, out = run_cmd(cmd)
            except Exception:
                self._send(500, traceback.format_exc())
                return
            self._send(200 if rc == 0 else 500, out if out.endswith("\n") else out + "\n")
            return
        self._send(404, "try GET /heal /probe /status /healthz\n")

    def do_POST(self) -> None:  # noqa: N802
        self.do_GET()


def main() -> int:
    if not os.path.isfile(HEAL_PY):
        print(f"missing {HEAL_PY}", file=sys.stderr)
        return 1
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"netobserv-heal-proxy listening on {HOST}:{PORT}", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
