#!/usr/bin/env python3
"""
OTEL Demo Ground-Truth Data Collector
Automatically toggles failure flags and exports labeled telemetry
for AIOps model training & evaluation.

Requirements:
pip install requests opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp \
               opentelemetry-exporter-jaeger opentelemetry-exporter-prometheus \
               pandas tqdm
"""

import time
import json
import requests
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ==================== CONFIG ====================
# DEMO_URL = "http://localhost:8080"                    # Astronomy Shop frontend
# FLAGD_PROXY_URL = "http://localhost:8013"              # flagd sidecar proxy (in demo)
# JAEGER_QUERY_URL = "http://localhost:16686/api"        # Jaeger query API
# PROMETHEUS_URL = "http://localhost:9090/api/v1"        # Prometheus API
# LOAD_GENERATOR_ENABLED = True                          # Set False if you control load manually

DEMO_URL = "http://localhost:8080"                    # Astronomy Shop frontend
# Prefer flagd-ui for writing full config in this demo
FLAGD_UI_READ_URL = f"{DEMO_URL}/feature/api/read"
FLAGD_UI_WRITE_URL = f"{DEMO_URL}/feature/api/write"
JAEGER_QUERY_URL = "http://localhost:32789/api"        # Jaeger query API
PROMETHEUS_URL = "http://localhost:9091/api/v1"        # Prometheus API
LOAD_GENERATOR_ENABLED = True                          # Set False if you control load manually

def _resolve_opensearch_url() -> str:
    """
    Resolve OpenSearch endpoint robustly:
    1) Respect OPENSEARCH_URL env if provided
    2) Use docker compose port mapping for service 'opensearch' port 9200
    3) Fallback to localhost:9200 then localhost:32795
    """
    env_url = os.getenv("OPENSEARCH_URL")
    if env_url:
        return env_url.rstrip("/")
    try:
        # docker compose port opensearch 9200 -> "0.0.0.0:32803"
        cp = subprocess.run(
            ["docker", "compose", "port", "opensearch", "9200"],
            capture_output=True, text=True, check=False
        )
        out = (cp.stdout or "").strip()
        if out and ":" in out:
            port = out.split(":")[-1]
            if port.isdigit():
                return f"http://localhost:{port}"
    except Exception:
        pass
    # Reasonable fallbacks
    for port in (9200, 32795):
        try:
            # quick HEAD probe with low timeout
            requests.get(f"http://localhost:{port}", timeout=1)
            return f"http://localhost:{port}"
        except Exception:
            continue
    return "http://localhost:9200"

OPENSEARCH_URL = _resolve_opensearch_url()             # OpenSearch (traces/logs)


# Define your experiments (flag → variant → duration in seconds)
EXPERIMENTS = [
    # Baseline
    {"flag": "cartFailure",                "variant": "off",   "duration": 180},
    # Individual failures
    {"flag": "cartFailure",                "variant": "on",    "duration": 600},
    {"flag": "paymentFailure",             "variant": "10%",   "duration": 600},
    {"flag": "paymentFailure",             "variant": "50%",   "duration": 600},
    {"flag": "paymentFailure",             "variant": "90%",   "duration": 600},
    {"flag": "paymentUnreachable",         "variant": "on",    "duration": 600},
    {"flag": "adFailure",                  "variant": "on",    "duration": 600},
    {"flag": "adHighCpu",                  "variant": "on",    "duration": 600},
    {"flag": "adManualGc",                 "variant": "on",    "duration": 600},
    {"flag": "productCatalogFailure",      "variant": "on",    "duration": 600},
    {"flag": "recommendationCacheFailure", "variant": "on",    "duration": 900},
    {"flag": "imageSlowLoad",              "variant": "5sec",  "duration": 600},
    {"flag": "imageSlowLoad",              "variant": "10sec", "duration": 600},
    {"flag": "emailMemoryLeak",            "variant": "10x",   "duration": 600},
    {"flag": "emailMemoryLeak",            "variant": "100x",  "duration": 600},
    {"flag": "kafkaQueueProblems",         "variant": "on",    "duration": 600},
    {"flag": "llmInaccurateResponse",      "variant": "on",    "duration": 600},
    {"flag": "llmRateLimitError",          "variant": "on",    "duration": 600},
    {"flag": "loadGeneratorFloodHomepage", "variant": "on",    "duration": 600},
    # Recovery
    {"flag": "cartFailure",                "variant": "off",   "duration": 300}
]

OUTPUT_DIR = Path("otel_ground_truth_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# ================================================

def set_flag(flag_name: str, variant: str):
    """Set a feature flag via flagd-ui by rewriting the full config document."""
    try:
        # Read current flags
        r = requests.get(FLAGD_UI_READ_URL, timeout=10)
        r.raise_for_status()
        flags_only = r.json().get("flags", {})
        if flag_name not in flags_only:
            print(f"Flag {flag_name} not found; available: {list(flags_only.keys())}")
        # Build full document for write
        doc = {
            "$schema": "https://flagd.dev/schema/v0/flags.json",
            "flags": flags_only
        }
        if flag_name in doc["flags"]:
            doc["flags"][flag_name]["defaultVariant"] = variant
        w = requests.post(FLAGD_UI_WRITE_URL, json={"data": doc}, timeout=10)
        w.raise_for_status()
        time.sleep(5)  # Allow propagation
        print(f"Flag {flag_name} set to {variant}")
    except Exception as e:
        print(f"Failed to set flag {flag_name}: {e}")

def export_traces_from_opensearch(label: str, since_dt: datetime):
    """Export traces from OpenSearch (collector exports to opensearch) since experiment start."""
    try:
        query = {
            "size": 10000,
            "sort": [{ "endTime": { "order": "desc", "unmapped_type": "date" } }],
            "query": { "match_all": {} }
        }
        resp = requests.post(f"{OPENSEARCH_URL}/otel-traces-*/_search", json=query, timeout=30)
        if not resp.ok:
            print(f"OpenSearch traces export failed: HTTP {resp.status_code}")
            hits = []
        else:
            hits_all = resp.json().get("hits", {}).get("hits", [])
            # Client-side filter by timestamp to avoid mapping quirks with range queries
            since_iso = since_dt.isoformat().replace("+00:00", "Z")
            hits = [h for h in hits_all if h.get("_source", {}).get("endTime", "") >= since_iso]
        path = OUTPUT_DIR / f"traces_{label}.json"
        with open(path, "w") as f:
            json.dump(hits, f, indent=2)
        print(f"Saved {len(hits)} traces → {path.name}")
    except Exception as e:
        print(f"OpenSearch traces export failed: {e}")

def export_prometheus_metrics(label: str):
    """Export key metrics as CSV"""
    queries = {
        # Adjusted to match demo app metrics
        "req_rate": 'sum(rate(app_frontend_requests_total[5m])) by (service_name)',
        "cart_add_p95": 'histogram_quantile(0.95, sum by (le, service_name) (rate(app_cart_add_item_latency_seconds_bucket[5m])))',
        "cpu_usage": 'rate(container_cpu_usage_seconds_total[5m])',
        "memory_usage": 'container_memory_working_set_bytes',
    }
    data = {}
    for name, query in queries.items():
        try:
            url = f"{PROMETHEUS_URL}/query?query={query}"
            result = requests.get(url).json()["data"]["result"]
            data[name] = result
        except:
            data[name] = []
    
    path = OUTPUT_DIR / f"metrics_{label}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Metrics saved → {path.name}")

def export_logs(label: str, since_dt: datetime):
    """If using Loki or OpenSearch, adapt this. Here we just copy from container."""
    # Simple fallback: grab last 10k lines from all containers
    import subprocess
    # Ensure we target the OTEL demo compose file regardless of CWD
    compose_file = (Path(__file__).resolve().parents[1] / "otel-demo" / "docker-compose.yml").as_posix()
    log_path = OUTPUT_DIR / f"logs_{label}.txt"
    since_iso = since_dt.isoformat().replace("+00:00", "Z")
    # Get explicit service list to avoid Compose filtering edge cases
    services_proc = subprocess.run(
        ["docker", "compose", "-f", compose_file, "ps", "--services"],
        capture_output=True, text=True, check=False
    )
    services = [s.strip() for s in (services_proc.stdout or "").splitlines() if s.strip()]
    chunks = []
    def fetch(args):
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout or ""
    for svc in services or ["--all"]:
        # 1) ISO timestamp
        out = fetch(["docker","compose","-f",compose_file,"logs","--no-color","--since",since_iso,"--tail=5000",svc] if svc!="--all" else ["docker","compose","-f",compose_file,"logs","--no-color","--since",since_iso,"--tail=5000"])
        # 2) Relative seconds
        if not out.strip():
            try:
                from datetime import datetime, timezone
                secs = int(max((datetime.now(timezone.utc) - since_dt).total_seconds() + 10, 60))
            except Exception:
                secs = 120
            out = fetch(["docker","compose","-f",compose_file,"logs","--no-color","--since",f"{secs}s","--tail=5000",svc] if svc!="--all" else ["docker","compose","-f",compose_file,"logs","--no-color","--since",f"{secs}s","--tail=5000"])
        # 3) No --since
        if not out.strip():
            out = fetch(["docker","compose","-f",compose_file,"logs","--no-color","--tail=10000",svc] if svc!="--all" else ["docker","compose","-f",compose_file,"logs","--no-color","--tail=10000"])
        if out.strip():
            # Prefix with service name when available
            if svc != "--all":
                prefixed = "\n".join(f"{svc}  | {line}" for line in out.splitlines())
                chunks.append(prefixed)
            else:
                chunks.append(out)
    final_out = ("\n".join(chunks)).strip()
    with open(log_path, "w") as f:
        f.write(final_out)
    print(f"Logs saved → {log_path.name}")

def run_experiment(flag: str, variant: str, duration: int, label: str):
    print(f"\nStarting experiment: {label} ({duration}s)")
    set_flag(flag, variant)
    
    print("Waiting for system to stabilize and generate data...")
    start_dt = datetime.now(timezone.utc)
    for _ in tqdm(range(duration), desc="Collecting", unit="s"):
        time.sleep(1)
    
    # Export all telemetry
    export_traces_from_opensearch(label, start_dt)
    export_prometheus_metrics(label)
    export_logs(label, start_dt)

    # Save metadata
    meta = {
        "label": label,
        "flag": flag,
        "variant": variant,
        "start_time": datetime.now().isoformat(),
        "duration_seconds": duration,
        "ground_truth_root_cause": flag if variant != "off" else "none"
    }
    with open(OUTPUT_DIR / f"metadata_{label}.json", "w") as f:
        json.dump(meta, f, indent=2)

# ==================== MAIN ====================
if __name__ == "__main__":
    print("OTEL Demo Ground-Truth Collector Starting...")
    print(f"Saving to: {OUTPUT_DIR.resolve()}")

    for i, exp in enumerate(EXPERIMENTS):
        flag = exp["flag"]
        variant = exp["variant"]
        duration = exp["duration"]
        is_baseline = variant == "off" and "Failure" in flag
        label = f"exp{i:02d}_{flag}_{variant}"
        if is_baseline:
            label = f"exp{i:02d}_baseline"

        run_experiment(flag, variant, duration, label)

    print("\nAll experiments complete!")
    print(f"Data ready in: {OUTPUT_DIR}")
    print("Use this for training AIOps models (anomaly detection, RCA, etc.)")
