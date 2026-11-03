"""Jaeger collector via Envoy (/jaeger/ui/api/...). Response shape verified on 3.1.0:
traces[].spans[] with operationName, references[CHILD_OF], tags[], duration (us), processID;
traces[].processes{pid: {serviceName}}.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from tier2.collectors.prometheus import EXCLUDED_SERVICES


def _http_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)


def _is_error(span: dict) -> bool:
    tags = {t["key"]: t["value"] for t in span.get("tags", [])}
    return tags.get("error") in (True, "true") or tags.get("otel.status_code") == "ERROR"


def _error_text(span: dict) -> str:
    tags = {t["key"]: t["value"] for t in span.get("tags", [])}
    for k in ("otel.status_description", "exception.message", "rpc.grpc.status_code",
              "http.response.status_code", "http.status_code"):
        if k in tags:
            return f"{k}={tags[k]}"
    return "error"


def parse_traces(raw: list[dict], max_traces: int = 20) -> tuple[list[dict], list[dict]]:
    """Return (failing_traces, dependency_edges) from a Jaeger trace list."""
    edges: dict[tuple[str, str], dict] = {}
    failing = []
    for tr in raw:
        procs = {k: v["serviceName"] for k, v in tr["processes"].items()}
        by_id = {s["spanID"]: s for s in tr["spans"]}
        failed, path = [], []
        for s in sorted(tr["spans"], key=lambda s: s["startTime"]):
            svc = procs.get(s["processID"], "?")
            if svc in EXCLUDED_SERVICES:
                continue
            if not path or path[-1] != svc:
                path.append(svc)
            parent = next((by_id.get(r["spanID"]) for r in s.get("references", [])
                           if r.get("refType") == "CHILD_OF"), None)
            err = _is_error(s)
            if parent:
                psvc = procs.get(parent["processID"], "?")
                if psvc != svc and psvc not in EXCLUDED_SERVICES:
                    e = edges.setdefault((psvc, svc), {"from": psvc, "to": svc, "calls": 0, "errors": 0})
                    e["calls"] += 1
                    e["errors"] += int(err)
            if err:
                failed.append({"service": svc, "operation": s["operationName"],
                               "error": _error_text(s)})
        if failed:
            root = min(tr["spans"], key=lambda s: s["startTime"])
            failing.append({"trace_id": tr["traceID"][:8], "path": path,
                            "duration_ms": round(root["duration"] / 1000, 1),
                            "failed_spans": failed[:4]})
    failing.sort(key=lambda t: (-len(t["failed_spans"]), t["trace_id"]))
    return failing[:max_traces], sorted(edges.values(), key=lambda e: (e["from"], e["to"]))


class Jaeger:
    def __init__(self, base_url: str = "http://localhost:8080", get=_http_get):
        self.base, self._get = base_url.rstrip("/"), get

    def services(self) -> list[str]:
        return self._get(f"{self.base}/jaeger/ui/api/services")["data"] or []

    def traces(self, service: str, start_s: float, end_s: float, errors_only=True,
               limit: int = 20) -> list[dict]:
        q = {"service": service, "start": int(start_s * 1e6), "end": int(end_s * 1e6),
             "limit": limit}
        if errors_only:
            q["tags"] = json.dumps({"error": "true"})
        return self._get(f"{self.base}/jaeger/ui/api/traces?" + urllib.parse.urlencode(q))["data"] or []
