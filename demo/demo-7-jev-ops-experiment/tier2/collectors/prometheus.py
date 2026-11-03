"""Prometheus collector. Metrics are the demo's span metrics, verified on 3.1.0:
traces_span_metrics_calls_total and traces_span_metrics_duration_milliseconds_bucket,
labels service_name, span_kind, span_name, status_code.

SDKs export every 60 s, so rate() over short ranges is empty. Everything here uses
increase() over a window of at least 180 s, evaluated at a fixed timestamp.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from tier2.observation import ServiceStats

# Harness and test infrastructure: not part of what the AI should reason about.
EXCLUDED_SERVICES = frozenset({"flagd", "flagd-ui", "telemetry-docs", "load-generator"})
SERVER_KINDS = "SPAN_KIND_SERVER|SPAN_KIND_CONSUMER"


class PrometheusError(RuntimeError):
    pass


def _http_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.load(r)


class Prometheus:
    def __init__(self, base_url: str = "http://localhost:9090", get=_http_get):
        self.base, self._get = base_url.rstrip("/"), get

    def query(self, expr: str, at: float) -> list[dict]:
        url = f"{self.base}/api/v1/query?" + urllib.parse.urlencode({"query": expr, "time": at})
        try:
            body = self._get(url)
        except OSError as e:
            raise PrometheusError(f"prometheus unreachable: {e}") from e
        if body.get("status") != "success":
            raise PrometheusError(f"prometheus query failed: {body.get('error')}")
        return body["data"]["result"]

    def service_stats(self, at: float, window_s: int) -> dict[str, ServiceStats]:
        w = f"{int(window_s)}s"
        sel = f'span_kind=~"{SERVER_KINDS}"'
        total = self._by_service(
            f'sum by (service_name)(increase(traces_span_metrics_calls_total{{{sel}}}[{w}]))', at)
        errors = self._by_service(
            f'sum by (service_name)(increase(traces_span_metrics_calls_total'
            f'{{{sel},status_code="STATUS_CODE_ERROR"}}[{w}]))', at)
        out: dict[str, ServiceStats] = {}
        quant = {q: self._by_service(
            f'histogram_quantile({q}, sum by (le, service_name)'
            f'(increase(traces_span_metrics_duration_milliseconds_bucket{{{sel}}}[{w}])))', at)
            for q in (0.5, 0.95)}
        for svc, calls in total.items():
            if svc in EXCLUDED_SERVICES or not calls:
                continue
            out[svc] = ServiceStats(
                request_rate=calls / window_s,
                error_rate=errors.get(svc, 0.0) / calls,
                p50_ms=quant[0.5].get(svc), p95_ms=quant[0.95].get(svc))
        return out

    def _by_service(self, expr: str, at: float) -> dict[str, float]:
        out = {}
        for r in self.query(expr, at):
            v = float(r["value"][1])
            if v == v:  # drop NaN
                out[r["metric"].get("service_name", "")] = v
        return out

    def span_metrics_present(self, at: float) -> bool:
        return bool(self.query('count(traces_span_metrics_calls_total)', at))
