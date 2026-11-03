"""OpenSearch log collector. Index otel-logs*, fields verified on 3.1.0:
resource.service.name, body, severity.number, severity.text, @timestamp.
severity.text is inconsistent across services (INFO/info/Information/error/Warning),
so filtering uses severity.number: >= 13 warn, >= 17 error.
OpenSearch's host port is assigned by Docker; resolve it with `docker port opensearch 9200`.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone

from tier2.collectors.prometheus import EXCLUDED_SERVICES

WARN_NUMBER = 13


def _http_post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def level_of(number: int) -> str:
    return "ERROR" if number >= 17 else "WARN"


class OpenSearch:
    def __init__(self, base_url: str, post=_http_post):
        self.base, self._post = base_url.rstrip("/"), post

    def error_logs(self, start_s: float, end_s: float, size: int = 500) -> list[dict]:
        iso = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat()
        body = {"size": size, "sort": [{"@timestamp": "asc"}],
                "query": {"bool": {"filter": [
                    {"range": {"severity.number": {"gte": WARN_NUMBER}}},
                    {"range": {"@timestamp": {"gte": iso(start_s), "lt": iso(end_s)}}}]}}}
        hits = self._post(f"{self.base}/otel-logs*/_search", body)["hits"]["hits"]
        out = []
        for h in hits:
            s = h["_source"]
            svc = s.get("resource", {}).get("service.name") or s.get("resource", {}).get("service", {}).get("name")
            if not svc or svc in EXCLUDED_SERVICES:
                continue
            out.append({"ts": s.get("@timestamp"), "service": svc,
                        "level": level_of(int(s.get("severity", {}).get("number", 0))),
                        "message": str(s.get("body", ""))})
        return out
