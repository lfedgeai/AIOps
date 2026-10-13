#!/usr/bin/env python3
"""SPIFFE mTLS front door for OpenClaw /hooks/agent — Bearer token stays on this proxy only."""
from __future__ import annotations

import json
import os
import signal
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn


LISTEN_HOST = os.environ.get("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "18790"))
OPENCLAW_HOOKS_URL = os.environ.get(
    "OPENCLAW_HOOKS_URL",
    "http://openclaw.openclaw.svc.cluster.local:18789/hooks/agent",
)
OPENCLAW_HOOKS_TOKEN = os.environ.get("OPENCLAW_HOOKS_TOKEN", "")
SPIFFE_CERT_DIR = os.environ.get("SPIFFE_CERT_DIR", "/spiffe/svid")
SPIFFE_CERT = os.path.join(SPIFFE_CERT_DIR, "svid.pem")
SPIFFE_KEY = os.path.join(SPIFFE_CERT_DIR, "svid.key")
SPIFFE_BUNDLE = os.path.join(SPIFFE_CERT_DIR, "svid_bundle.pem")
# Comma-separated SPIFFE ID prefixes allowed as mTLS clients (empty = any verified client).
ALLOWED_CLIENT_SPIFFE_PREFIXES = [
    p.strip()
    for p in os.environ.get("ALLOWED_CLIENT_SPIFFE_PREFIXES", "").split(",")
    if p.strip()
]


def wait_for_spiffe_certs(timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if all(os.path.isfile(p) for p in (SPIFFE_CERT, SPIFFE_KEY, SPIFFE_BUNDLE)):
            return
        time.sleep(1)
    sys.stderr.write("SPIFFE cert files not ready under %s\n" % SPIFFE_CERT_DIR)
    sys.exit(1)


def client_spiffe_id(peercert: dict) -> str:
    for subj in peercert.get("subject", ()):
        for key, value in subj:
            if key == "commonName" and value.startswith("spiffe://"):
                return value
    sans = peercert.get("subjectAltName") or ()
    for kind, value in sans:
        if kind == "URI" and str(value).startswith("spiffe://"):
            return str(value)
    return ""


def client_allowed(peercert: dict) -> bool:
    if not ALLOWED_CLIENT_SPIFFE_PREFIXES:
        return True
    spiffe_id = client_spiffe_id(peercert)
    if not spiffe_id:
        return False
    return any(spiffe_id.startswith(prefix) for prefix in ALLOWED_CLIENT_SPIFFE_PREFIXES)


def forward_to_openclaw(body: bytes, headers: dict[str, str]) -> tuple[int, bytes]:
    if not OPENCLAW_HOOKS_TOKEN:
        return 503, b"OPENCLAW_HOOKS_TOKEN unset"
    req = urllib.request.Request(
        OPENCLAW_HOOKS_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {OPENCLAW_HOOKS_TOKEN}",
            "Content-Type": headers.get("Content-Type", "application/json"),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def build_ssl_context() -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile=SPIFFE_CERT, keyfile=SPIFFE_KEY)
    ctx.load_verify_locations(cafile=SPIFFE_BUNDLE)
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    ssl_context: ssl.SSLContext | None = None

    def get_request(self):
        newsocket, addr = super().get_request()
        if self.ssl_context is None:
            raise RuntimeError("SSL context not initialized")
        return self.ssl_context.wrap_socket(newsocket, server_side=True), addr

    def reload_ssl_context(self) -> None:
        self.ssl_context = build_ssl_context()
        sys.stderr.write("reloaded SPIFFE server certificate\n")


def watch_spiffe_cert_rotation(server: ThreadingHTTPServer, interval_s: int = 30) -> None:
    last_mtime = os.path.getmtime(SPIFFE_CERT) if os.path.isfile(SPIFFE_CERT) else 0.0

    def loop() -> None:
        nonlocal last_mtime
        while True:
            time.sleep(interval_s)
            try:
                mtime = os.path.getmtime(SPIFFE_CERT)
            except OSError:
                continue
            if mtime > last_mtime:
                last_mtime = mtime
                try:
                    server.reload_ssl_context()
                except Exception as exc:
                    sys.stderr.write("SPIFFE cert reload failed: %s\n" % exc)

    threading.Thread(target=loop, daemon=True, name="spiffe-cert-watch").start()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _reject(self, code: int, message: str) -> None:
        body = (message + "\n").encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in ("", "/health", "/healthz"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")
            return
        self._reject(404, "not found")

    def do_POST(self) -> None:
        if self.path.rstrip("/") not in ("", "/hooks/agent", "/hooks/agent/"):
            self._reject(404, "not found")
            return
        peercert = self.connection.getpeercert()
        if not peercert:
            self._reject(403, "client certificate required")
            return
        if not client_allowed(peercert):
            spiffe_id = client_spiffe_id(peercert) or "unknown"
            sys.stderr.write("rejected client spiffe_id=%s\n" % spiffe_id)
            self._reject(403, "client SPIFFE ID not allowed")
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        code, out = forward_to_openclaw(raw, dict(self.headers))
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def main() -> None:
    wait_for_spiffe_certs()
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    server.ssl_context = build_ssl_context()
    watch_spiffe_cert_rotation(server)

    def on_sighup(_signum, _frame) -> None:
        try:
            server.reload_ssl_context()
        except Exception as exc:
            sys.stderr.write("SPIFFE cert reload failed: %s\n" % exc)

    signal.signal(signal.SIGHUP, on_sighup)

    sys.stderr.write(
        "openclaw-hooks-mtls-proxy listening on %s:%s (client cert required)\n"
        % (LISTEN_HOST, LISTEN_PORT)
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
