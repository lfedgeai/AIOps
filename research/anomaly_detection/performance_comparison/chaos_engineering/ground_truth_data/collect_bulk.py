#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import sys
sys.path.append(str(Path(__file__).parent.resolve()))
import ground_truth_data_collector as gdc  # noqa: E402


DEFAULT_FLAGS: List[str] = [
    "cartFailure",
    "paymentFailure",
    "paymentUnreachable",
    "adFailure",
    "adHighCpu",
    "adManualGc",
    "productCatalogFailure",
    "recommendationCacheFailure",
    "imageSlowLoad",
    "emailMemoryLeak",
    "kafkaQueueProblems",
    "llmInaccurateResponse",
    "llmRateLimitError",
    "loadGeneratorFloodHomepage",
]


def set_all_off() -> None:
    try:
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
        requests.post(WRITE, json={"data": {"$schema": "https://flagd.dev/schema/v0/flags.json", "flags": flags}}, timeout=10)
    except Exception:
        pass
    for f in DEFAULT_FLAGS:
        try:
            gdc.set_flag(f, "off")
        except Exception:
            pass


def collect_baselines(count: int, duration: int, cooldown: int) -> None:
    for i in range(1, count + 1):
        label = f"bulk_baseline_{i:02d}"
        print(f"[baseline] {label}: off for {duration}s")
        set_all_off()
        time.sleep(cooldown)
        start_dt = datetime.now(timezone.utc)
        time.sleep(duration)
        gdc.export_traces_from_opensearch(label, start_dt)
        gdc.export_prometheus_metrics(label)
        gdc.export_logs(label, start_dt)
        meta = {
            "label": label,
            "flag": "baseline",
            "variant": "off",
            "start_time": datetime.now().isoformat(),
            "duration_seconds": duration,
            "ground_truth_root_cause": "none",
        }
        (gdc.OUTPUT_DIR / f"metadata_{label}.json").write_text(__import__("json").dumps(meta, indent=2))


def collect_faults(flags: List[str], count: int, duration: int, cooldown: int) -> None:
    for flag in flags:
        # Use a single variant per flag that is most representative
        if flag == "paymentFailure":
            variants = ["10%", "50%", "90%"]
        elif flag == "imageSlowLoad":
            variants = ["5sec", "10sec"]
        elif flag == "emailMemoryLeak":
            variants = ["10x", "100x"]
        else:
            variants = ["on"]
        for variant in variants:
            for i in range(1, count + 1):
                label = f"bulk_{flag}_{variant}_{i:02d}"
                print(f"[fault] {label}: {flag}={variant} for {duration}s")
                gdc.set_flag(flag, variant)
                time.sleep(cooldown)
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
                    "ground_truth_root_cause": flag,
                }
                (gdc.OUTPUT_DIR / f"metadata_{label}.json").write_text(__import__("json").dumps(meta, indent=2))
            # turn flag off between variants
            gdc.set_flag(flag, "off")
            time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect large dataset: N baselines and N fault windows per flag")
    ap.add_argument("--flags", nargs="*", default=DEFAULT_FLAGS, help="Subset of flags to collect")
    ap.add_argument("--count", type=int, default=30, help="Windows per baseline and per flag variant")
    ap.add_argument("--duration", type=int, default=60, help="Seconds per window")
    ap.add_argument("--cooldown", type=int, default=10, help="Seconds between toggles")
    args = ap.parse_args()

    out = gdc.OUTPUT_DIR.resolve()
    print(f"[bulk] Saving to: {out}")
    collect_baselines(args.count, args.duration, args.cooldown)
    collect_faults(list(args.flags), args.count, args.duration, args.cooldown)
    print("[bulk] Complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

