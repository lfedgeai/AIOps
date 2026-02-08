#!/usr/bin/env python3
"""
Collect a full suite of baseline + 18 fault experiments for the OTEL demo app.
- Uses ground_truth_data_collector's set_flag and export functions
- Short, configurable duration per window (default 60s)
Labels:
- train baselines: train01..train05_baseline (optional helper, not used here)
- suite windows: suite00_baseline, suite01_<flag>_<variant> ... suite18_<flag>_<variant>, suite19_baseline
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

import sys
sys.path.append(str(Path(__file__).parent.resolve()))
import ground_truth_data_collector as gdc  # noqa: E402


EXPERIMENTS: List[Dict[str, str | int]] = [
    {"flag": "cartFailure",                "variant": "off"},   # baseline
    {"flag": "cartFailure",                "variant": "on"},
    {"flag": "paymentFailure",             "variant": "10%"},
    {"flag": "paymentFailure",             "variant": "50%"},
    {"flag": "paymentFailure",             "variant": "90%"},
    {"flag": "paymentUnreachable",         "variant": "on"},
    {"flag": "adFailure",                  "variant": "on"},
    {"flag": "adHighCpu",                  "variant": "on"},
    {"flag": "adManualGc",                 "variant": "on"},
    {"flag": "productCatalogFailure",      "variant": "on"},
    {"flag": "recommendationCacheFailure", "variant": "on"},
    {"flag": "imageSlowLoad",              "variant": "5sec"},
    {"flag": "imageSlowLoad",              "variant": "10sec"},
    {"flag": "emailMemoryLeak",            "variant": "10x"},
    {"flag": "emailMemoryLeak",            "variant": "100x"},
    {"flag": "kafkaQueueProblems",         "variant": "on"},
    {"flag": "llmInaccurateResponse",      "variant": "on"},
    {"flag": "llmRateLimitError",          "variant": "on"},
    {"flag": "loadGeneratorFloodHomepage", "variant": "on"},
    {"flag": "cartFailure",                "variant": "off"},   # recovery baseline
]


def run_one(label: str, flag: str, variant: str, duration: int) -> None:
    print(f"Starting: {label} → {flag}={variant} ({duration}s) at {datetime.now().isoformat()}")
    gdc.set_flag(flag, variant)
    start_dt = datetime.now(timezone.utc)
    time.sleep(duration)
    gdc.export_traces_from_opensearch(label, start_dt)
    gdc.export_prometheus_metrics(label)
    gdc.export_logs(label, start_dt)
    meta = {
        "label": label,
        "flag": flag,
        "variant": variant,
        "start_time": datetime.now().isoformat(),
        "duration_seconds": duration,
        "ground_truth_root_cause": flag if variant != "off" else "none",
    }
    (gdc.OUTPUT_DIR / f"metadata_{label}.json").write_text(__import__("json").dumps(meta, indent=2))
    print(f"Completed: {label} at {datetime.now().isoformat()}")


def set_all_flags_off() -> None:
    import requests
    DEMO_URL = gdc.DEMO_URL
    READ = f"{DEMO_URL}/feature/api/read"
    WRITE = f"{DEMO_URL}/feature/api/write"
    r = requests.get(READ, timeout=10)
    r.raise_for_status()
    doc = r.json()
    flags = doc.get("flags", {}) or {}
    for v in flags.values():
        if isinstance(v, dict):
            v["defaultVariant"] = "off"
    payload = {"data": {"$schema": "https://flagd.dev/schema/v0/flags.json", "flags": flags}}
    w = requests.post(WRITE, json=payload, timeout=10)
    w.raise_for_status()
    time.sleep(5)
    print("All flags set to off.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect full 18-fault suite + baselines")
    ap.add_argument("--duration", type=int, default=60, help="Seconds per window")
    args = ap.parse_args()

    print("Full suite collector starting...")
    print(f"Saving to: {gdc.OUTPUT_DIR.resolve()}")

    set_all_flags_off()

    for i, exp in enumerate(EXPERIMENTS):
        flag = str(exp["flag"])
        variant = str(exp["variant"])
        label = f"suite{i:02d}_{'baseline' if variant=='off' else f'{flag}_{variant}'}"
        run_one(label, flag, variant, args.duration)

    print("Full suite collection complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


