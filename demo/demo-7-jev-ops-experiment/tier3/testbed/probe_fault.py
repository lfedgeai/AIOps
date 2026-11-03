"""Task 1.6 helper: inject one fault by hand and record what it does, without Jev.

    .venv/bin/python tier3/testbed/probe_fault.py scale_zero cart [--hold 240] [--every 15]

Uses the real injector (ledger, read-back, exact reset) under the dead-man timer.
Every tick records: the harness manifestation check, the target deployment row,
unhealthy or new pods, new events (Normal and Warning), service error rate and p95
over a 120 s window, failing traces, warn/error log counts, and a direct request
to /api/cart through frontend-proxy (what a user sees). After reset it times
recovery. The record goes to results/tier3/probe/<fault>-<target>-<time>.json.
Nothing here is sent to any engine.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tier2.collectors.jaeger import Jaeger  # noqa: E402
from tier2.collectors.opensearch import OpenSearch  # noqa: E402
from tier2.collectors.prometheus import Prometheus  # noqa: E402
from tier3 import testbed  # noqa: E402
from tier3.collect import collect, kube_state  # noqa: E402
from tier3.collectors.kube import Kube  # noqa: E402
from tier3.deadman import Deadman  # noqa: E402
from tier3.faults import FaultInjector  # noqa: E402
from tier3.manifest import manifestation, recovery_problems  # noqa: E402
from tier3.scenarios import FaultSpec  # noqa: E402

WINDOW = 120


def iso(t=None):
    return datetime.fromtimestamp(t or time.time(), timezone.utc).isoformat(timespec="seconds")


def user_request(base: str) -> dict:
    t0 = time.time()
    try:
        with urllib.request.urlopen(f"{base}/api/cart?sessionId=probe-user&currencyCode=USD", timeout=10) as r:
            r.read()
            code = r.status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception as e:  # noqa: BLE001
        code = type(e).__name__
    return {"status": code, "ms": round((time.time() - t0) * 1000)}


def all_events(kube: Kube) -> list[dict]:
    out = []
    for ns in (kube.ns, "default"):
        out += (kube._req("GET", f"/api/v1/namespaces/{ns}/events") or {}).get("items", [])
    return out


def event_key(e):
    return (e["metadata"]["uid"], e.get("count"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fault")
    ap.add_argument("target")
    ap.add_argument("--hold", type=int, default=240, help="seconds the fault stays in")
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--recover-timeout", type=int, default=420)
    a = ap.parse_args()

    kube = Kube.from_kubeconfig(testbed.kubeconfig(), testbed.namespace())
    prom = Prometheus(testbed.prometheus_url())
    jaeger, logs = Jaeger(testbed.jaeger_url()), OpenSearch(testbed.opensearch_url())
    inj = FaultInjector(kube)
    if inj.leftovers():
        print("leftovers:", inj.leftovers(), "- run python -m tier3 reset-faults")
        return 1

    t0 = time.time()
    base_stats = prom.service_stats(t0, 180)
    base_kube = kube_state(kube)
    seen = {event_key(e) for e in all_events(kube)}
    rec = {"fault": a.fault, "target": a.target, "started": iso(t0), "hold_s": a.hold,
           "baseline_user": user_request(testbed.base_url()), "ticks": [], "placement":
           {p["name"]: p["node"] for p in base_kube.pods}}
    print(f"baseline user request: {rec['baseline_user']}")

    dm = Deadman()
    unit = dm.arm(a.hold + a.recover_timeout + 120)
    t_inject = time.time()
    try:
        rec["saved"] = inj.inject(FaultSpec(a.fault, a.target))
        rec["t_inject"] = iso(t_inject)
        print(f"injected {a.fault} on {a.target} at {rec['t_inject']}")
        while time.time() - t_inject < a.hold:
            time.sleep(max(0, t_inject + (len(rec["ticks"]) + 1) * a.every - time.time()))
            now = time.time()
            try:
                tel, ks = collect(prom, jaeger, logs, kube, window_end=now, window_s=WINDOW,
                                  baseline_end=t0, baseline_s=180)
            except Exception as e:  # noqa: BLE001 - recorded; the probe goes on
                rec["ticks"].append({"t": round(now - t_inject), "collect_error": repr(e)[:300],
                                     "user": user_request(testbed.base_url())})
                print(f"t+{round(now - t_inject):>3}s collect error: {e!r}"[:300])
                continue
            why = manifestation(tel.services, base_stats, ks, base_kube)
            new_events = []
            for e in all_events(kube):
                if event_key(e) not in seen:
                    seen.add(event_key(e))
                    new_events.append({"type": e.get("type"), "reason": e.get("reason"),
                                       "object": f"{e['involvedObject'].get('kind')}/{e['involvedObject'].get('name')}",
                                       "message": (e.get("message") or "")[:200], "count": e.get("count")})
            dep = ks.deployments.get(a.target)
            bad_pods = [p for p in ks.pods if not (p["phase"] == "Running" and p["ready"]) or p.get("deleting")
                        or p["name"] not in {q["name"] for q in base_kube.pods}]
            svc = {s: {"err": round(v.error_rate or 0, 3), "p95": round(v.p95_ms or 0)}
                   for s, v in tel.services.items()
                   if (v.error_rate or 0) > 0.001 or s in ("cart", "frontend", "checkout", a.target)}
            tick = {"t": round(now - t_inject), "user": user_request(testbed.base_url()), "manifest": why,
                    "deployment": dep, "pods": [{k: p[k] for k in ("name", "node", "phase", "ready", "restarts",
                                                                      "waiting_reason", "last_termination_reason", "deleting")}
                                                for p in bad_pods],
                    "nodes": [{"name": n["name"], "ready": n["ready"], "taints": [t["effect"] for t in n["taints"]],
                               "pods": n["pods"]} for n in ks.nodes],
                    "events": new_events, "services": svc, "failing_traces": len(tel.traces),
                    "error_logs": dict(Counter(l["service"] for l in tel.logs))}
            rec["ticks"].append(tick)
            print(f"t+{tick['t']:>3}s user={tick['user']['status']}/{tick['user']['ms']}ms "
                  f"dep={dep and (dep['available'], dep['desired'], dep['rollout'])} "
                  f"pods={[(p['name'][-5:], p['phase'], p['ready'], p['waiting_reason'] or p['last_termination_reason']) for p in bad_pods][:3]} "
                  f"svc_err={ {k: v['err'] for k, v in svc.items() if v['err']} } traces={tick['failing_traces']} "
                  f"logs={tick['error_logs']} events={[e['reason'] for e in new_events]} manifest={len(why)}")
    finally:
        t_reset = time.time()
        rec["restored"] = inj.reset()
        rec["t_reset"] = iso(t_reset)
        print("reset:", rec["restored"])
        while True:
            now = time.time()
            probs = recovery_problems(kube_state(kube), base_kube, inj.leftovers(),
                                      prom.service_stats(now, 120), base_stats)
            if not probs or now - t_reset > a.recover_timeout:
                break
            time.sleep(15)
        rec["recovery_s"] = round(time.time() - t_reset)
        rec["recovery_problems"] = probs
        dm.disarm(unit)
        print(f"recovered={not probs} after {rec['recovery_s']}s {probs}")
        out = Path("results/tier3/probe")
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"{a.fault}-{a.target}-{datetime.now().strftime('%H%M%S')}.json"
        p.write_text(json.dumps(rec, indent=2, default=str))
        print("record:", p)
    return 0 if not probs else 2


if __name__ == "__main__":
    sys.exit(main())
