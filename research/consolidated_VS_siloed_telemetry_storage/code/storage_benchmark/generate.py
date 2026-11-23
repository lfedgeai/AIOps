#!/usr/bin/env python3
"""Deterministic OTel-Demo-shaped telemetry with a known fault, sent as OTLP/HTTP JSON.

One seed gives one corpus. Spans and logs share real trace IDs. Metrics have no trace ID,
as in normal OTLP. The expected answer of each benchmark task is computed here, from the
generated events, and not from either storage backend.
"""
from __future__ import annotations
import argparse, collections, json, random, time
from pathlib import Path
import requests

MIN = 60 * 10**9
# entry route -> (weight, call tree). A node is (service, span name, children).
ROUTES = [
    (45, ("frontend", "GET /product", [("product-catalog", "GetProduct", []), ("recommendation", "ListRecommendations", [("product-catalog", "ListProducts", [])]), ("ad", "GetAds", [])])),
    (30, ("frontend", "GET /cart", [("cart", "GetCart", []), ("currency", "Convert", [])])),
    (25, ("frontend", "POST /checkout", [("checkout", "PlaceOrder", [("cart", "GetCart", []), ("product-catalog", "GetProduct", []), ("currency", "Convert", []), ("quote", "GetQuote", []), ("payment", "Charge", []), ("shipping", "ShipOrder", []), ("email", "SendConfirmation", [])])])),
]
SERVICES = ["frontend", "checkout", "cart", "product-catalog", "recommendation", "ad", "currency", "quote", "payment", "shipping", "email", "accounting"]
PODS = 3
GAUGES = ["app_cpu_utilization", "app_memory_utilization", "app_queue_depth", "app_request_error_ratio"]
FAULT_SERVICE = "payment"


def attrs(d): return [{"key": k, "value": {"stringValue": str(v)}} for k, v in d.items()]
def resource(svc, pod): return {"attributes": attrs({"service.name": svc, "service.namespace": "otel-demo", "k8s.pod.name": f"{svc}-{pod}"})}


class Corpus:
    def __init__(self, seed: int, minutes: int, traces: int, end_ns: int):
        self.r = random.Random(seed)
        self.minutes, self.n_traces = minutes, traces
        self.t0 = end_ns // MIN * MIN - minutes * MIN
        # Fault for five minutes, at two thirds of the corpus. Tasks query this window.
        self.f0 = self.t0 + (minutes * 2 // 3) * MIN
        self.f1 = self.f0 + 5 * MIN
        self.spans, self.logs, self.metrics = [], [], []
        self._traces(); self._background_logs(); self._metrics()
        for x in (self.spans, self.logs, self.metrics): x.sort(key=lambda e: e["t"])

    def _id(self, bits): return f"{self.r.getrandbits(bits) | 1:0{bits // 4}x}"

    def _log(self, t, svc, sev, body, tid="", sid=""):
        # Never on a minute boundary: backends differ on which bucket owns the edge.
        if t % MIN < 10**6: t += 10**6
        self.logs.append({"t": t, "svc": svc, "sev": sev, "body": body, "tid": tid, "sid": sid, "pod": self.r.randrange(PODS)})

    def _span(self, tid, parent, node, t, failing):
        svc, name, kids = node
        sid, fail = self._id(64), failing and svc == FAULT_SERVICE
        own = int(self.r.lognormvariate(1.2, 0.7) * 1e6) + 50_000
        if fail: own += 250 * 10**6
        cur = t + own // 2
        for k in kids:
            end, bad = self._span(tid, sid, k, cur, failing)
            fail |= bad
            cur = end + 20_000
            if bad: break  # callers stop at the first failed call
        end = cur + own // 2
        self.spans.append({"t": t, "end": end, "tid": tid, "sid": sid, "parent": parent, "svc": svc, "name": name, "err": fail, "pod": self.r.randrange(PODS)})
        self._log(t + 1000, svc, "INFO", f"{name} received, session={self._id(64)}", tid, sid)
        if fail:
            cause = "Payment request failed. Invalid token. app.loyalty.level=gold" if svc == FAULT_SERVICE else f"{name} failed: downstream call returned an error"
            self._log(end - 1000, svc, "ERROR", cause, tid, sid)
        elif self.r.random() < .3:
            self._log(end - 1000, svc, "INFO", f"{name} completed in {(end - t) / 1e6:.2f} ms", tid, sid)
        return end, fail

    def _traces(self):
        w = [r[0] for r in ROUTES]
        for _ in range(self.n_traces):
            # Stay 2 s clear of each minute edge so that a whole trace is in one minute.
            t = self.t0 + self.r.randrange(self.minutes) * MIN + self.r.randrange(2 * 10**9, 56 * 10**9)
            route = self.r.choices(ROUTES, w)[0][1]
            in_fault = self.f0 <= t < self.f1
            # 40% payment failure in the fault, 0.5% noise outside it.
            failing = "checkout" in route[1] and self.r.random() < (.40 if in_fault else .005)
            self._span(self._id(128), "", route, t, failing)

    def _background_logs(self):
        # Logs without a trace ID, including errors that must NOT join to a trace.
        for _ in range(self.n_traces * 2):
            t = self.t0 + self.r.randrange(self.minutes * MIN)
            svc = self.r.choice(SERVICES)
            if self.r.random() < .02: self._log(t, svc, "ERROR", "connection reset by peer while polling config")
            else: self._log(t, svc, "INFO", f"heartbeat ok seq={self.r.getrandbits(24)}")

    def _metrics(self):
        for m in range(self.minutes):
            for s in (5, 20, 35, 50):
                t = self.t0 + m * MIN + s * 10**9
                fault = self.f0 <= t < self.f1
                for svc in SERVICES:
                    for pod in range(PODS):
                        for g in GAUGES:
                            v = self.r.uniform(.05, .6)
                            if g == "app_request_error_ratio":
                                v = self.r.uniform(0, .01)
                                # payment is worst; checkout and frontend inherit part of it.
                                if fault: v += {"payment": .40, "checkout": .30, "frontend": .08}.get(svc, 0)
                            self.metrics.append({"t": t, "svc": svc, "pod": pod, "name": g, "v": round(v, 6)})

    # ---- ground truth ----
    def truth(self):
        f0, f1 = self.f0, self.f1
        win = lambda e: f0 <= e["t"] < f1
        errs = [s for s in self.spans if win(s) and s["err"]]
        elogs = collections.defaultdict(list)
        for l in self.logs:
            if l["sev"] == "ERROR" and l["tid"]: elogs[l["tid"]].append(l)
        tid = sorted(s["tid"] for s in errs if s["svc"] == FAULT_SERVICE)[0]
        ctx = {"spans": sorted([s["sid"], s["svc"]] for s in self.spans if s["tid"] == tid),
               "logs": sorted([l["sid"], l["svc"], l["sev"], l["body"]] for l in self.logs if l["tid"] == tid)}
        sw = {}
        def cell(svc, t): return sw.setdefault(f"{svc}|{(t - f0) // MIN}", [0, 0, 0, 0.0])
        for s in self.spans:
            if win(s):
                c = cell(s["svc"], s["t"]); c[0] += 1; c[1] += s["err"]
        for l in self.logs:
            if win(l) and l["sev"] == "ERROR": cell(l["svc"], l["t"])[2] += 1
        for m in self.metrics:
            if win(m) and m["name"] == "app_request_error_ratio":
                c = cell(m["svc"], m["t"]); c[3] = max(c[3], m["v"])
        # drilldown: worst service by error ratio -> its error traces in fault minute 2 -> their error logs
        d0, d1 = f0 + MIN, f0 + 2 * MIN
        mean = collections.defaultdict(list)
        for m in self.metrics:
            if win(m) and m["name"] == "app_request_error_ratio": mean[m["svc"]].append(m["v"])
        worst = max(mean, key=lambda s: sum(mean[s]) / len(mean[s]))
        dt = {s["tid"] for s in self.spans if s["svc"] == worst and s["err"] and d0 <= s["t"] < d1}
        drill = {"service": worst, "traces": {t: len(elogs[t]) for t in sorted(dt)}}
        # bulk join: every error span in the window x error logs of the same trace
        bulk = collections.Counter()
        for s in errs:
            for l in elogs[s["tid"]]: bulk[f"{s['svc']}|{l['svc']}"] += 1
        return {"trace_context": ctx, "service_window": sw, "error_drilldown": drill, "bulk_join": dict(bulk)}, tid

    # ---- OTLP ----
    def otlp(self, kind, batch):
        groups = collections.defaultdict(list)
        for e in batch: groups[(e["svc"], e["pod"])].append(e)
        out = []
        for (svc, pod), es in groups.items():
            res = resource(svc, pod)
            if kind == "traces":
                items = [{"traceId": e["tid"], "spanId": e["sid"], "parentSpanId": e["parent"], "name": e["name"], "kind": 2,
                          "startTimeUnixNano": str(e["t"]), "endTimeUnixNano": str(e["end"]), "status": {"code": 2 if e["err"] else 1},
                          "attributes": attrs({"http.response.status_code": 500 if e["err"] else 200})} for e in es]
                out.append({"resource": res, "scopeSpans": [{"scope": {"name": "bench"}, "spans": items}]})
            elif kind == "logs":
                items = [{"timeUnixNano": str(e["t"]), "observedTimeUnixNano": str(e["t"]), "severityText": e["sev"], "severityNumber": 17 if e["sev"] == "ERROR" else 9,
                          "body": {"stringValue": e["body"]}, **({"traceId": e["tid"], "spanId": e["sid"]} if e["tid"] else {})} for e in es]
                out.append({"resource": res, "scopeLogs": [{"scope": {"name": "bench"}, "logRecords": items}]})
            else:
                by = collections.defaultdict(list)
                for e in es: by[e["name"]].append({"timeUnixNano": str(e["t"]), "asDouble": e["v"]})
                out.append({"resource": res, "scopeMetrics": [{"scope": {"name": "bench"}, "metrics": [{"name": n, "gauge": {"dataPoints": d}} for n, d in by.items()]}]})
        return {{"traces": "resourceSpans", "logs": "resourceLogs", "metrics": "resourceMetrics"}[kind]: out}

    def send(self, url):
        """Send minute by minute, in time order. Returns bytes and seconds."""
        s, total, t = requests.Session(), 0, time.perf_counter()
        for kind, events in (("metrics", self.metrics), ("traces", self.spans), ("logs", self.logs)):
            size = 2000
            for i in range(0, len(events), size):
                body = json.dumps(self.otlp(kind, events[i:i + size])).encode()
                total += len(body)
                for attempt in range(60):
                    r = s.post(f"{url}/v1/{kind}", data=body, headers={"Content-Type": "application/json"}, timeout=60)
                    if r.status_code == 200: break
                    if r.status_code not in (429, 503): r.raise_for_status()
                    time.sleep(.5)
                else: raise RuntimeError(f"collector kept refusing {kind}")
        return total, time.perf_counter() - t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traces", type=int, default=20000); p.add_argument("--minutes", type=int, default=30)
    p.add_argument("--seed", type=int, default=45); p.add_argument("--url", default="http://127.0.0.1:34318")
    p.add_argument("--out", default="out/corpus.json"); p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    c = Corpus(a.seed, a.minutes, a.traces, time.time_ns() - MIN)
    truth, tid = c.truth()
    meta = {"seed": a.seed, "traces": a.traces, "minutes": a.minutes, "t0": c.t0, "t1": c.t0 + a.minutes * MIN, "f0": c.f0, "f1": c.f1, "trace_id": tid,
            "counts": {"spans": len(c.spans), "logs": len(c.logs), "metrics": len(c.metrics)}, "truth": truth}
    if not a.dry_run:
        meta["sent_bytes"], meta["send_s"] = c.send(a.url)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(meta))
    print({k: v for k, v in meta.items() if k != "truth"})
    print({"drill_service": truth["error_drilldown"]["service"], "drill_traces": len(truth["error_drilldown"]["traces"]), "bulk": truth["bulk_join"]})

if __name__ == "__main__": main()
