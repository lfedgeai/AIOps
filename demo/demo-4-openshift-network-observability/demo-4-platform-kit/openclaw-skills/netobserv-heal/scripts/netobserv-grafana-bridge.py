#!/usr/bin/env python3
"""Receive Grafana unified alerting webhooks; trigger OpenClaw /hooks/agent → Slack."""
from __future__ import annotations

import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# Shared MLflow tracing (mounted at /tracing when wired by wire-openclaw-mlflow.sh)
sys.path.insert(0, os.environ.get("MLFLOW_TRACING_PATH", "/tracing"))
try:
    from mlflow_tracing import (
        active_incident_trace_json,
        incident_root_span,
        init_mlflow,
    )
except ImportError:
    active_incident_trace_json = None  # type: ignore[misc, assignment]
    incident_root_span = None  # type: ignore[misc, assignment]
    init_mlflow = lambda: False  # type: ignore[misc, assignment]


OPENCLAW_HOOKS_URL = os.environ.get(
    "OPENCLAW_HOOKS_URL",
    "http://openclaw.openclaw.svc.cluster.local:18789/hooks/agent",
)
OPENCLAW_HOOKS_TOKEN = os.environ.get("OPENCLAW_HOOKS_TOKEN", "")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID", "")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8080"))
SPIFFE_MTLS = os.environ.get("SPIFFE_MTLS", "").lower() in ("1", "true", "yes")
SPIFFE_CERT_DIR = os.environ.get("SPIFFE_CERT_DIR", "/spiffe/svid")
SPIFFE_CERT = os.path.join(SPIFFE_CERT_DIR, "svid.pem")
SPIFFE_KEY = os.path.join(SPIFFE_CERT_DIR, "svid.key")
SPIFFE_BUNDLE = os.path.join(SPIFFE_CERT_DIR, "svid_bundle.pem")


def wait_for_spiffe_certs(timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if all(os.path.isfile(p) for p in (SPIFFE_CERT, SPIFFE_KEY, SPIFFE_BUNDLE)):
            return
        time.sleep(1)
    sys.stderr.write("SPIFFE cert files not ready under %s\n" % SPIFFE_CERT_DIR)
    sys.exit(1)


def hooks_ssl_context() -> ssl.SSLContext | None:
    if not SPIFFE_MTLS:
        return None
    wait_for_spiffe_certs()
    ctx = ssl.create_default_context(cafile=SPIFFE_BUNDLE)
    ctx.load_cert_chain(certfile=SPIFFE_CERT, keyfile=SPIFFE_KEY)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def investigate_message(payload: dict) -> str:
    status = payload.get("status") or "firing"
    title = payload.get("title") or payload.get("commonLabels", {}).get("alertname") or "Grafana alert"
    message = payload.get("message") or payload.get("commonAnnotations", {}).get("summary") or ""
    labels = payload.get("commonLabels") or {}
    return (
        "Automated AIOps incident from Grafana.\n\n"
        f"Status: {status}\n"
        f"Title: {title}\n"
        f"Summary: {message}\n"
        f"Labels: {json.dumps(labels, sort_keys=True)}\n\n"
        "Operators report the todo application is slow talking to PostgreSQL. "
        "Investigate using NetObserv flow evidence and MCP triage. "
        "Post a concise diagnosis in this channel.\n"
        "Do NOT heal or call remediate tools unless a human explicitly confirms in a follow-up message."
    )


def idempotency_key(payload: dict) -> str:
    raw = json.dumps(
        {
            "status": payload.get("status"),
            "title": payload.get("title"),
            "labels": payload.get("commonLabels"),
            "alerts": [
                {
                    "fingerprint": (a or {}).get("fingerprint"),
                    "status": (a or {}).get("status"),
                }
                for a in (payload.get("alerts") or [])[:3]
            ],
        },
        sort_keys=True,
    )
    return "grafana-" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def is_firing(payload: dict) -> bool:
    status = str(payload.get("status") or "").lower()
    state = str(payload.get("state") or "").lower()
    if status in ("firing", "active") or state in ("firing", "active", "alerting"):
        return True
    for alert in payload.get("alerts") or []:
        if str((alert or {}).get("status") or "").lower() == "firing":
            return True
    return False


def trigger_openclaw(payload: dict, extra_headers: dict[str, str] | None = None) -> tuple[int, str]:
    if not SPIFFE_MTLS and not OPENCLAW_HOOKS_TOKEN:
        return 503, "OPENCLAW_HOOKS_TOKEN unset"
    if not SLACK_CHANNEL_ID:
        return 503, "SLACK_CHANNEL_ID unset"
    body = {
        "message": investigate_message(payload),
        "name": "Grafana-NetObserv",
        "sessionMode": "isolated",
        "deliver": True,
        "channel": "slack",
        "to": f"channel:{SLACK_CHANNEL_ID}",
        "idempotencyKey": idempotency_key(payload),
        "wakeMode": "now",
    }
    headers = {"Content-Type": "application/json"}
    if OPENCLAW_HOOKS_TOKEN:
        headers["Authorization"] = f"Bearer {OPENCLAW_HOOKS_TOKEN}"
    if extra_headers:
        for k, v in extra_headers.items():
            if k.lower() in ("traceparent", "tracestate") and v:
                headers[k] = v
    req = urllib.request.Request(
        OPENCLAW_HOOKS_URL,
        data=json.dumps(body).encode(),
        method="POST",
        headers=headers,
    )
    ctx = hooks_ssl_context()
    try:
        with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def trigger_openclaw_traced(payload: dict) -> tuple[int, str]:
    """Fire OpenClaw hook under an MLflow investigate_session root span."""
    if incident_root_span is None or not init_mlflow():
        return trigger_openclaw(payload)

    idem = idempotency_key(payload)
    title = payload.get("title") or payload.get("commonLabels", {}).get("alertname") or "Grafana alert"
    with incident_root_span(
        "investigate_session",
        inputs={
            "title": title,
            "status": payload.get("status"),
            "labels": payload.get("commonLabels") or {},
        },
        attributes={"audit.trigger": "grafana-webhook"},
        idempotency_key=idem,
    ) as trace_headers:
        code, out = trigger_openclaw(payload, extra_headers=trace_headers)
    return code, out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        path = self.path.rstrip("/")
        if path in ("", "/health", "/healthz"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")
            return
        if path == "/mlflow/trace-context":
            payload = active_incident_trace_json() if active_incident_trace_json else {}
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path.rstrip("/") not in ("", "/grafana", "/webhook"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"invalid json\n")
            return
        status = payload.get("status") or payload.get("state") or "unknown"
        if not is_firing(payload):
            sys.stderr.write(f"skip webhook status={status!r}\n")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"skipped": "not firing", "status": status}).encode() + b"\n")
            return
        code, out = trigger_openclaw_traced(payload)
        self.send_response(200 if code in (200, 202) else 502)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"openclaw_status": code, "openclaw_body": out[:500]}).encode())


def main() -> None:
    init_mlflow()
    server = HTTPServer(("0.0.0.0", LISTEN_PORT), Handler)
    sys.stderr.write(f"netobserv-grafana-bridge listening on :{LISTEN_PORT}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
