import json

import pytest

from tier2.collect import collect
from tier2.collectors.jaeger import Jaeger, parse_traces
from tier2.collectors.opensearch import OpenSearch, level_of
from tier2.collectors.prometheus import Prometheus, PrometheusError
from tier2.testbed import run_checks
from tier2.workload import Workload
from tests.tier2.helpers import FakeFlagStore


# --- Prometheus -------------------------------------------------------------
def prom_get(total, errors, p50, p95):
    """Fake HTTP get keyed on which query is being asked."""
    def vec(d):
        return {"status": "success", "data": {"result": [
            {"metric": {"service_name": k}, "value": [0, str(v)]} for k, v in d.items()]}}

    def get(url):
        q = url
        if "histogram_quantile%280.5" in q:
            return vec(p50)
        if "histogram_quantile%280.95" in q:
            return vec(p95)
        if "STATUS_CODE_ERROR" in q:
            return vec(errors)
        return vec(total)
    return get


def test_service_stats_math_and_exclusions():
    get = prom_get(total={"payment": 18, "flagd": 999, "load-generator": 50, "checkout": 0},
                   errors={"payment": 9}, p50={"payment": 40}, p95={"payment": "NaN"})
    s = Prometheus(get=get).service_stats(1000.0, 180)
    assert set(s) == {"payment"}  # harness services and zero-call services dropped
    assert s["payment"].request_rate == pytest.approx(0.1)
    assert s["payment"].error_rate == pytest.approx(0.5)
    assert s["payment"].p50_ms == 40 and s["payment"].p95_ms is None  # NaN dropped, not zero


def test_prometheus_error_is_named():
    with pytest.raises(PrometheusError, match="unreachable"):
        Prometheus(get=lambda u: (_ for _ in ()).throw(OSError("refused"))).query("up", 1)
    with pytest.raises(PrometheusError, match="failed"):
        Prometheus(get=lambda u: {"status": "error", "error": "bad"}).query("up", 1)


def test_prometheus_uses_wide_increase_windows_not_rate():
    seen = []
    Prometheus(get=lambda u: seen.append(u) or {"status": "success", "data": {"result": []}}
               ).service_stats(1000.0, 180)
    assert seen and all("increase" in u and "rate%28" not in u for u in seen)
    assert all("180s" in u for u in seen)


# --- Jaeger -----------------------------------------------------------------
def span(sid, parent, pid, op, start, err=False, dur=1000):
    tags = [{"key": "error", "value": True}] if err else []
    refs = [{"refType": "CHILD_OF", "spanID": parent}] if parent else []
    return {"traceID": "t" * 32, "spanID": sid, "operationName": op, "references": refs,
            "startTime": start, "duration": dur, "tags": tags, "processID": pid}


TRACE = {"traceID": "abcdef0123456789", "processes": {
    "p1": {"serviceName": "frontend"}, "p2": {"serviceName": "checkout"},
    "p3": {"serviceName": "payment"}, "p4": {"serviceName": "flagd"}},
    "spans": [
        span("s1", None, "p1", "POST /api/checkout", 1, err=True, dur=3_000_000),
        span("s2", "s1", "p2", "PlaceOrder", 2, err=True),
        span("s3", "s2", "p3", "Charge", 3, err=True),
        span("s4", "s3", "p4", "flagd.evaluation.ResolveFloat", 4),  # must be ignored
    ]}


def test_parse_traces_path_edges_and_flagd_exclusion():
    failing, edges = parse_traces([TRACE])
    assert failing[0]["path"] == ["frontend", "checkout", "payment"]
    assert failing[0]["duration_ms"] == 3000.0
    assert {(e["from"], e["to"]) for e in edges} == {("frontend", "checkout"), ("checkout", "payment")}
    assert next(e for e in edges if e["to"] == "payment")["errors"] == 1
    assert "flagd" not in json.dumps([failing, edges])


def test_parse_traces_healthy_trace_gives_edges_but_no_failure():
    ok = json.loads(json.dumps(TRACE))
    for s in ok["spans"]:
        s["tags"] = []
    failing, edges = parse_traces([ok])
    assert failing == [] and edges


def test_jaeger_query_shape():
    seen = []
    Jaeger(get=lambda u: seen.append(u) or {"data": []}).traces("cart", 100.0, 200.0)
    assert "service=cart" in seen[0] and "error" in seen[0] and "limit=" in seen[0]


# --- OpenSearch -------------------------------------------------------------
def test_error_logs_levels_services_and_filter():
    hits = [
        {"_source": {"@timestamp": "t1", "body": "boom", "severity": {"number": 17},
                     "resource": {"service.name": "payment"}}},
        {"_source": {"@timestamp": "t2", "body": "slow", "severity": {"number": 13},
                     "resource": {"service.name": "cart"}}},
        {"_source": {"@timestamp": "t3", "body": "x", "severity": {"number": 17},
                     "resource": {"service.name": "load-generator"}}},
        {"_source": {"@timestamp": "t4", "body": "no service", "severity": {"number": 17}, "resource": {}}},
    ]
    sent = []
    os_ = OpenSearch("http://x", post=lambda u, b: sent.append(b) or {"hits": {"hits": hits}})
    logs = os_.error_logs(0, 100)
    assert [(l["service"], l["level"]) for l in logs] == [("payment", "ERROR"), ("cart", "WARN")]
    flt = sent[0]["query"]["bool"]["filter"][0]["range"]["severity.number"]["gte"]
    assert flt == 13  # severity.number, not severity.text (text is inconsistent)
    assert level_of(21) == "ERROR" and level_of(13) == "WARN"


# --- assembler and health ---------------------------------------------------
def test_collect_assembles_telemetry_without_fault_knowledge():
    prom = Prometheus(get=prom_get({"payment": 18}, {"payment": 9}, {"payment": 40}, {"payment": 90}))
    jaeger = Jaeger(get=lambda u: {"data": [TRACE]})
    logs = OpenSearch("http://x", post=lambda u, b: {"hits": {"hits": []}})
    t = collect(prom, jaeger, logs, window_end=2000.0, window_s=180, baseline_end=1800.0, baseline_s=180)
    assert "payment" in t.services and "payment" in t.baseline_services
    assert t.traces and t.edges
    assert t.window_end > t.window_start and t.baseline_end < t.window_end
    assert len({tr["trace_id"] for tr in t.traces}) == len(t.traces)  # deduplicated


class OkJaeger:
    def services(self):
        return ["frontend"]


class OkProm:
    def __init__(self, present=True):
        self.present = present

    def span_metrics_present(self, now):
        return self.present


def checks(**kw):
    defaults = dict(store=FakeFlagStore(), prom=OkProm(), jaeger=OkJaeger(),
                    workload=Workload(get=lambda u: {"state": "running", "user_count": 5}),
                    os_url="http://os", now=1.0,
                    get=lambda url, timeout=8.0: (200, b'{"status":"green"}'))
    defaults.update(kw)
    return {c.name: c for c in run_checks(**defaults)}


def test_health_all_ok_and_ordered():
    c = checks()
    assert list(c) == ["frontend", "prometheus", "span-metrics", "jaeger", "opensearch",
                       "flags-off", "locust"]
    assert all(x.ok for x in c.values())


def test_health_names_failing_checks():
    assert not checks(prom=OkProm(present=False))["span-metrics"].ok
    assert not checks(workload=Workload(get=lambda u: {"state": "stopped", "user_count": 0}))["locust"].ok
    assert not checks(get=lambda url, timeout=8.0: (503, b"{}"))["frontend"].ok


def test_health_fails_on_leftover_fault_and_names_it():
    from tier2.faults import FaultInjector
    from tier2.scenarios import FaultSpec
    store = FakeFlagStore()
    FaultInjector(store).inject(FaultSpec("cartFailure", "100%"))
    c = checks(store=store)["flags-off"]
    assert not c.ok and "cartFailure" in c.detail
