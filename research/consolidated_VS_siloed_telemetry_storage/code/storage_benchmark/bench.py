#!/usr/bin/env python3
"""Correlation benchmark without an agent: ClickHouse against Thanos + Loki + Tempo.

Every answer is compared with the generator's ground truth. A latency counts only for a
correct answer. Siloed time is split into backend time (HTTP) and correlation time (code).
"""
import argparse, json, os, shutil, statistics as st, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
import correlate
from generate import GAUGES

CH, LOKI, TEMPO, THANOS = "http://127.0.0.1:38123", "http://127.0.0.1:33100", "http://127.0.0.1:33200", "http://127.0.0.1:39090"
NS, MIN = 10**9, 60 * 10**9
TASKS = ["trace_context", "service_window", "error_drilldown", "bulk_join"]
SILOED = ["loki", "tempo", "thanos-receive", "thanos-query"]
ALL = '{service_name=~".+"}'
VARIANTS = {"trace_context": ["broad_seq", "broad_par", "scoped"], "service_window": ["broad_seq", "broad_par"],
            "error_drilldown": ["broad_seq", "scoped"], "bulk_join": ["broad_seq", "broad_par", "scoped"]}
def scope(svcs): return '{service_name=~"%s"}' % "|".join(sorted(svcs))
PAGE = 5000  # Loki's default max_entries_limit_per_query


class Siloed:
    """One siloed client. Variants:
    broad_seq   all-services Loki selector, one call after another
    broad_par   the same calls; independent calls run concurrently
    scoped      Tempo first; Loki is then limited to the services (and, for one trace, the
                time range) that Tempo returned. The calls depend on each other.
    """
    def __init__(self, m):
        self.m = m
        self.s = {n: requests.Session() for n in ("loki", "tempo", "thanos")}
        self.pool = ThreadPoolExecutor(3)

    def get(self, name, url, **params):
        t = time.perf_counter()
        r = self.s[name].get(url, params=params, headers={"Accept": "application/json"}, timeout=120)
        body = r.content
        ms = (time.perf_counter() - t) * 1e3
        if r.status_code != 200: raise RuntimeError(f"{name} {r.status_code}: {body[:300]}")
        self.raw.setdefault(name, []).append(body); self.calls.append((name, ms))
        return body

    def logs(self, q, t0, t1):
        cur = t0
        while True:
            body = self.get("loki", f"{LOKI}/loki/api/v1/query_range", query=q, start=cur, end=t1, limit=PAGE, direction="forward")
            ts = [int(v[0]) for s in json.loads(body)["data"]["result"] for v in s["values"]]
            if len(ts) < PAGE: return
            cur = max(ts)  # inclusive; the correlator removes the overlap

    def all(self, par, *fs):
        if par: list(self.pool.map(lambda f: f(), fs))
        else: [f() for f in fs]

    def run(self, task, variant):
        self.raw, self.calls = {}, []
        t = time.perf_counter()
        getattr(self, task)(variant)
        wall = (time.perf_counter() - t) * 1e3
        per = {}
        for n, ms in self.calls: per[n] = per.get(n, 0) + ms
        return {"wall_ms": wall, "calls": len(self.calls), "bytes": sum(len(b) for v in self.raw.values() for b in v), "per_backend": per}

    def trace_context(self, v):
        m, f0, f1 = self.m, self.m["f0"], self.m["f1"]
        tempo = lambda: self.get("tempo", f"{TEMPO}/api/v2/traces/{m['trace_id']}")
        sel, a, b = ALL, f0, f1 + MIN
        if v == "scoped":
            tempo()
            svcs, a, b = correlate.trace_scope(self.raw)
            sel, a, b = scope(svcs), a - NS, b + NS
        loki = lambda: self.logs('%s | trace_id="%s"' % (sel, m["trace_id"]), a, b)
        loki() if v == "scoped" else self.all(v == "broad_par", tempo, loki)

    def service_window(self, v):
        f0, f1 = self.m["f0"], self.m["f1"]
        self.all(v == "broad_par",
            lambda: self.get("tempo", f"{TEMPO}/api/metrics/query_range", q="{} | count_over_time() by (resource.service.name, status)", start=f0 // NS, end=f1 // NS, step="60s"),
            lambda: self.get("loki", f"{LOKI}/loki/api/v1/query_range", query='sum by (service_name) (count_over_time(%s | severity_text="ERROR" [1m]))' % ALL, start=f0 + MIN, end=f1, step="60"),
            lambda: self.get("thanos", f"{THANOS}/api/v1/query_range", query="max by (service_name) (max_over_time(app_request_error_ratio[1m]))", start=(f0 + MIN) // NS, end=f1 // NS, step=60))

    def error_drilldown(self, v):
        f0, f1 = self.m["f0"], self.m["f1"]
        d0, d1 = f0 + MIN, f0 + 2 * MIN
        self.get("thanos", f"{THANOS}/api/v1/query", query="topk(1, avg by (service_name) (avg_over_time(app_request_error_ratio[5m])))", time=f1 // NS)
        q = '{resource.service.name="%s" && status=error}' % correlate.worst_service(self.raw)
        if v == "scoped": q = "(%s && {status=error}) | select(resource.service.name)" % q
        self.get("tempo", f"{TEMPO}/api/search", q=q, start=d0 // NS, end=d1 // NS, limit=10000, spss=100)
        ids = correlate.search_ids(self.raw)
        sel = scope(correlate.search_services(self.raw)) if v == "scoped" else ALL
        for i in range(0, len(ids), 150):  # keep the URL short
            self.logs('%s | severity_text="ERROR" | trace_id=~"%s"' % (sel, "|".join(ids[i:i + 150])), d0, d1 + MIN)

    def bulk_join(self, v):
        f0, f1 = self.m["f0"], self.m["f1"]
        tempo = lambda: self.get("tempo", f"{TEMPO}/api/search", q="{status=error} | select(resource.service.name)", start=f0 // NS, end=f1 // NS, limit=100000, spss=100)
        sel = ALL
        if v == "scoped":
            tempo(); sel = scope(correlate.search_services(self.raw))
        loki = lambda: self.logs('%s | severity_text="ERROR" | trace_id!=""' % sel, f0, f1 + MIN)
        loki() if v == "scoped" else self.all(v == "broad_par", tempo, loki)


def ch_sql(task, m):
    f0, f1 = m["f0"], m["f1"]
    T = lambda ns: f"fromUnixTimestamp64Nano({ns})"
    if task == "trace_context":
        return f"""SELECT 'span' k, SpanId, ServiceName, '' sev, '' body FROM otel.otel_traces WHERE TraceId='{m['trace_id']}'
UNION ALL SELECT 'log', SpanId, ServiceName, SeverityText, Body FROM otel.otel_logs
WHERE TraceId='{m['trace_id']}' AND Timestamp >= {T(f0)} AND Timestamp < {T(f1 + MIN)}"""
    if task == "service_window":
        # toUnixTimestamp works for DateTime64 (logs, traces) and DateTime (metrics).
        b = lambda c: f"intDiv(toInt64(toUnixTimestamp({c})) - {f0 // NS}, 60) AS b"
        return f"""SELECT svc, b, sum(n), sum(e), sum(l), max(r) FROM (
SELECT ServiceName svc, {b('Timestamp')}, count() n, countIf(StatusCode='Error') e, 0 l, 0. r FROM otel.otel_traces WHERE Timestamp >= {T(f0)} AND Timestamp < {T(f1)} GROUP BY svc, b
UNION ALL SELECT ServiceName svc, {b('Timestamp')}, 0, 0, count(), 0. FROM otel.otel_logs WHERE Timestamp >= {T(f0)} AND Timestamp < {T(f1)} AND SeverityText='ERROR' GROUP BY svc, b
UNION ALL SELECT ServiceName svc, {b('TimeUnix')}, 0, 0, 0, max(Value) FROM otel.otel_metrics_gauge WHERE MetricName='app_request_error_ratio' AND TimeUnix >= {T(f0)} AND TimeUnix < {T(f1)} GROUP BY svc, b
) GROUP BY svc, b"""
    if task == "error_drilldown":
        d0, d1 = f0 + MIN, f0 + 2 * MIN
        return f"""WITH worst AS (SELECT ServiceName FROM otel.otel_metrics_gauge WHERE MetricName='app_request_error_ratio' AND TimeUnix >= {T(f0)} AND TimeUnix < {T(f1)} GROUP BY ServiceName ORDER BY avg(Value) DESC LIMIT 1),
t AS (SELECT DISTINCT TraceId FROM otel.otel_traces WHERE Timestamp >= {T(d0)} AND Timestamp < {T(d1)} AND StatusCode='Error' AND ServiceName IN (SELECT ServiceName FROM worst))
SELECT (SELECT ServiceName FROM worst), t.TraceId, countIf(l.SeverityText='ERROR') FROM t LEFT JOIN
(SELECT TraceId, SeverityText FROM otel.otel_logs WHERE Timestamp >= {T(d0)} AND Timestamp < {T(d1 + MIN)} AND SeverityText='ERROR' AND TraceId IN (SELECT TraceId FROM t)) l
ON l.TraceId=t.TraceId GROUP BY t.TraceId"""
    return f"""SELECT s.ServiceName, l.ServiceName, count() FROM
(SELECT TraceId, ServiceName FROM otel.otel_traces WHERE Timestamp >= {T(f0)} AND Timestamp < {T(f1)} AND StatusCode='Error') s JOIN
(SELECT TraceId, ServiceName FROM otel.otel_logs WHERE Timestamp >= {T(f0)} AND Timestamp < {T(f1 + MIN)} AND SeverityText='ERROR' AND TraceId != '') l
ON s.TraceId=l.TraceId GROUP BY 1, 2"""


def ch_canon(task, body):
    rows = json.loads(body)["data"]
    if task == "trace_context":
        return {"spans": sorted([r[1], r[2]] for r in rows if r[0] == "span"), "logs": sorted(r[1:] for r in rows if r[0] == "log")}
    if task == "service_window": return {f"{r[0]}|{r[1]}": [int(x) for x in r[2:5]] + [r[5]] for r in rows}
    if task == "error_drilldown": return {"service": rows[0][0], "traces": {r[1]: int(r[2]) for r in rows}}
    return {f"{r[0]}|{r[1]}": int(r[2]) for r in rows}


def same(a, b):
    if isinstance(b, dict): return isinstance(a, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in b)
    if isinstance(b, list): return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(b, float) or isinstance(a, float): return abs(a - b) < 1e-9
    return a == b


def summ(xs):
    xs = sorted(xs)
    return {"p50": round(st.median(xs), 3), "p95": round(xs[min(len(xs) - 1, int(.95 * len(xs)))], 3), "n": len(xs)}


def compose(*a): subprocess.run(["docker", "compose", *a], check=True, capture_output=True)


def wait_visible(m, timeout=300):
    """Wait until every signal is complete in both designs. Fail on any loss."""
    s, c, e = requests.Session(), m["counts"], m["t1"] // NS + 60
    rng = f"{m['minutes'] + 5}m"
    def ch(t): return int(s.post(CH, params={"user": "default", "password": "bench"}, data=f"SELECT count() FROM otel.{t}").text)
    def prom(u, q): return int(sum(float(x["value"][1]) for x in s.get(u, params={"query": q, "time": e}).json()["data"]["result"]))
    def tempo():
        d = s.get(f"{TEMPO}/api/metrics/query_range", params={"q": "{} | count_over_time()", "start": m["t0"] // NS, "end": m["t1"] // NS, "step": "60s"}).json()
        return int(sum(p.get("value", 0) for x in d.get("series", []) for p in x["samples"]))
    probes = {"ch.spans": (lambda: ch("otel_traces"), c["spans"]), "ch.logs": (lambda: ch("otel_logs"), c["logs"]), "ch.metrics": (lambda: ch("otel_metrics_gauge"), c["metrics"]),
              "loki.logs": (lambda: prom(f"{LOKI}/loki/api/v1/query", 'sum(count_over_time({service_name=~".+"}[%s]))' % rng), c["logs"]), "tempo.spans": (tempo, c["spans"]),
              "thanos.metrics": (lambda: prom(f"{THANOS}/api/v1/query", " + ".join(f"sum(count_over_time({g}[{rng}]))" for g in GAUGES)), c["metrics"])}
    end = time.time() + timeout
    got = {}
    while time.time() < end:
        for k, (f, want) in probes.items():
            if got.get(k) == want: continue
            try: got[k] = f()
            except Exception as ex: got[k] = repr(ex)[:80]
        if all(got[k] == w for k, (_, w) in probes.items()): return got
        time.sleep(2)
    raise SystemExit(f"data not complete: {[(k, got[k], w) for k, (_, w) in probes.items() if got[k] != w]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="out/corpus.json"); p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--runs", type=int, default=10, help="timed runs per task and variant in each round")
    p.add_argument("--warmup", type=int, default=3); p.add_argument("--out", default="out/results.json")
    p.add_argument("--go", default="out/correlate")
    a = p.parse_args()
    m = json.loads(Path(a.corpus).read_text()); f0_ms = m["f0"] // 10**6
    res = {"corpus": {k: v for k, v in m.items() if k != "truth"}, "rounds": a.rounds, "runs": a.runs, "warmup": a.warmup,
           "host": {"cpuset": os.environ.get("BENCH_CPUSET", "0,2,4,6"), "load": [os.getloadavg()]}, "tasks": {}}
    res["ingested"] = wait_visible(m)
    print("ingest complete, no loss:", res["ingested"])
    errors, acc = [], {}

    def add(task, arm, **kv):
        for k, x in kv.items(): acc.setdefault((task, arm), {}).setdefault(k, []).append(x)

    def clickhouse():
        s = requests.Session()
        for task in TASKS:
            for arm, extra in (("clickhouse", ""), ("clickhouse_no_skip_index", " SETTINGS use_skip_indexes=0")):
                sql = ch_sql(task, m) + extra + " FORMAT JSONCompact"
                for i in range(a.warmup + a.runs):
                    t = time.perf_counter()
                    r = s.post(CH, params={"user": "default", "password": "bench", "output_format_json_quote_64bit_integers": 0}, data=sql, timeout=120); body = r.content
                    t1 = time.perf_counter()
                    if r.status_code != 200: raise SystemExit(f"{task}: {body[:600]}")
                    got = ch_canon(task, body); t2 = time.perf_counter()
                    if not same(got, m["truth"][task]): errors.append((arm, task, got)); break
                    if i >= a.warmup: add(task, arm, total_ms=(t2 - t) * 1e3, backend_ms=(t1 - t) * 1e3, calls=1, bytes=len(body))

    def siloed():
        sil = Siloed(m)
        for task in TASKS:
            fn = getattr(correlate, task)
            for v in VARIANTS[task]:
                for i in range(a.warmup + a.runs):
                    r = sil.run(task, v)
                    t = time.perf_counter(); got = fn(sil.raw, f0_ms); dt = (time.perf_counter() - t) * 1e3
                    if not same(got, m["truth"][task]): errors.append((v, task, got)); break
                    if i >= a.warmup:
                        # wall_ms already includes glue parsing between dependent calls
                        add(task, v, total_ms=r["wall_ms"] + dt, backend_ms=r["wall_ms"], correlate_python_ms=dt, calls=r["calls"], bytes=r["bytes"],
                            **{f"{k}_ms": x for k, x in r["per_backend"].items()})
                if any(e[:2] == (v, task) for e in errors): continue
                d = Path(a.go).parent.resolve() / "raw" / task
                shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
                for n, bodies in sil.raw.items():
                    for i, b in enumerate(bodies): (d / f"{n}_{i:04d}.json").write_bytes(b)
                g = json.loads(subprocess.run([str(Path(a.go).resolve()), task, str(d), str(f0_ms), str(a.warmup + a.runs)], check=True, capture_output=True, text=True).stdout)
                if not same(g["result"], m["truth"][task]): errors.append((v + "-go", task, g["result"])); continue
                for x in g["ns"][a.warmup:]: add(task, v, correlate_go_ms=x / 1e6)

    for rnd in range(a.rounds):
        # Alternate the order. The design that is not measured is paused.
        for design in (("ch", "sil") if rnd % 2 == 0 else ("sil", "ch")):
            paused = SILOED if design == "ch" else ["clickhouse"]
            wait_visible(m)
            compose("pause", *paused)
            try: clickhouse() if design == "ch" else siloed()
            finally: compose("unpause", *paused)
        res["host"]["load"].append(os.getloadavg())
        if errors: break

    for (task, arm), d in acc.items():
        res["tasks"].setdefault(task, {})[arm] = {k: (x[0] if k in ("calls", "bytes") else summ(x)) for k, x in d.items()}
        # p50 per round shows whether the rounds agree
        n = a.runs
        res["tasks"][task][arm]["p50_by_round"] = [round(st.median(d["total_ms"][i:i + n]), 2) for i in range(0, len(d["total_ms"]), n)]
    res["errors"] = [(d, t) for d, t, _ in errors]
    Path(a.out).write_text(json.dumps(res, indent=1))
    for d, t, got in errors: print("WRONG ANSWER", d, t, "\n got ", json.dumps(got)[:700], "\n want", json.dumps(m["truth"][t])[:700])
    print(f"{'task':16}{'arm':26}{'p50':>8}{'p95':>8}{'py':>7}{'go':>7}{'calls':>6}{'bytes':>9}  p50 by round (ms)")
    for t, arms in res["tasks"].items():
        for arm, v in arms.items():
            py, go = (v.get(k, {}).get("p50", 0) for k in ("correlate_python_ms", "correlate_go_ms"))
            print(f"{t:16}{arm:26}{v['total_ms']['p50']:8.1f}{v['total_ms']['p95']:8.1f}{py:7.2f}{go:7.2f}{v['calls']:6}{v['bytes']:9}  {v['p50_by_round']}")
    raise SystemExit(1 if errors else 0)

if __name__ == "__main__": main()
