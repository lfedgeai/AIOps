"""Testbed helpers: where the demo lives, how to reach its backends, health checks."""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tier2.collectors.jaeger import Jaeger
from tier2.collectors.prometheus import Prometheus
from tier2.faults import HttpFlagStore, effective_variant
from tier2.workload import Workload

ROOT = Path(__file__).parent
DEMO_DIR = ROOT / "testbed" / "opentelemetry-demo"
FLAG_FILE = DEMO_DIR / "src" / "flagd" / "demo.flagd.json"
DEMO_TAG = "3.1.0"
COMPOSE_FILES = ("compose.yaml", "compose.full.yaml", "compose.observability.yaml")


def base_url() -> str:
    return os.environ.get("TIER2_BASE_URL", "http://localhost:8080")


def prometheus_url() -> str:
    return os.environ.get("TIER2_PROMETHEUS_URL", "http://localhost:9090")


def opensearch_url() -> str:
    """OpenSearch's host port is assigned by Docker, so ask Docker (or use the override)."""
    if os.environ.get("TIER2_OPENSEARCH_URL"):
        return os.environ["TIER2_OPENSEARCH_URL"]
    out = subprocess.run(["docker", "port", "opensearch", "9200"], capture_output=True,
                         text=True, check=False).stdout.split()
    if not out:
        raise RuntimeError("cannot find the OpenSearch port (is the demo running?)")
    return "http://localhost:" + out[0].rsplit(":", 1)[1]


def compose_cmd(*args: str) -> list[str]:
    cmd = ["docker", "compose", "--env-file", ".env", "--env-file", ".env.override"]
    minimal = os.environ.get("TIER2_PROFILE") == "minimal"  # no Kafka, accounting, fraud-detection
    for f in COMPOSE_FILES:
        if not (minimal and f == "compose.full.yaml"):
            cmd += ["-f", f]
    return cmd + list(args)


def compose_env() -> dict:
    # The demo's .env says DEMO_VERSION=latest. A shell variable overrides it, which
    # pins the images to DEMO_TAG without modifying the demo checkout.
    return {**os.environ, "DEMO_VERSION": DEMO_TAG}


def image_versions() -> dict:
    out = subprocess.run(compose_cmd("ps", "--format", "{{.Service}} {{.Image}}"), cwd=DEMO_DIR,
                         env=compose_env(), capture_output=True, text=True, check=False).stdout
    return dict(line.split(" ", 1) for line in out.splitlines() if " " in line)


def git_commit() -> str | None:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT.parent,
                       capture_output=True, text=True, check=False)
    return r.stdout.strip() or None


def flag_names() -> list[str]:
    """All flag names in the pinned flag file; used as scrub terms."""
    return sorted(json.loads(FLAG_FILE.read_text())["flags"]) if FLAG_FILE.exists() else []


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _get(url: str, timeout: float = 8.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def run_checks(store: HttpFlagStore | None = None, prom: Prometheus | None = None,
               jaeger: Jaeger | None = None, workload: Workload | None = None,
               os_url: str | None = None, now: float | None = None, get=_get) -> list[Check]:
    """Frontend, Prometheus, span metrics, Jaeger, OpenSearch, flag state, Locust (in order)."""
    import time
    store = store or HttpFlagStore(base_url())
    prom = prom or Prometheus(prometheus_url())
    jaeger = jaeger or Jaeger(base_url())
    workload = workload or Workload(base_url())
    now = now or time.time()
    checks: list[Check] = []

    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001 - a failed check is a result, not a crash
            ok, detail = False, f"{type(e).__name__}: {e}"
        checks.append(Check(name, bool(ok), detail))

    check("frontend", lambda: ((s := get(base_url() + "/")[0]) == 200, f"HTTP {s}"))
    check("prometheus", lambda: ((s := get(prometheus_url() + "/-/ready")[0]) == 200, f"HTTP {s}"))
    check("span-metrics", lambda: (prom.span_metrics_present(now), "traces_span_metrics_calls_total"))
    check("jaeger", lambda: (bool(jaeger.services()), f"{len(jaeger.services())} services"))
    check("opensearch", lambda: (
        json.loads(get((os_url or opensearch_url()) + "/_cluster/health")[1]).get("status") in
        ("green", "yellow"), "cluster health"))

    def flags():
        active = {n: effective_variant(f) for n, f in store.read()["flags"].items()
                  if effective_variant(f) != "off"}
        return not active, f"active faults: {active}" if active else "all flags off"
    check("flags-off", flags)
    check("locust", lambda: (workload.require_running() > 0, "load generator running"))
    return checks
