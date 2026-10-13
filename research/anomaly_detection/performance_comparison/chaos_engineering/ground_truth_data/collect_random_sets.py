#!/usr/bin/env python3
"""
Collect randomized validation and test datasets for the OTEL demo app.
- Uses ground_truth_data_collector's set_flag and export functions
- Randomly selects failure flags/variants
- Adds one or more baseline windows per split
"""
from __future__ import annotations

import argparse
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from pathlib import Path

# Reuse existing collector functions and config
import sys
sys.path.append(str(Path(__file__).parent.resolve()))
import ground_truth_data_collector as gdc  # noqa: E402


# Map of flag -> allowed variants (excluding 'off' which is baseline)
FAIL_FLAGS: Dict[str, List[str]] = {
    "cartFailure": ["on"],
    "paymentFailure": ["10%", "50%", "90%"],
    "paymentUnreachable": ["on"],
    "adFailure": ["on"],
    "adHighCpu": ["on"],
    "adManualGc": ["on"],
    "productCatalogFailure": ["on"],
    "recommendationCacheFailure": ["on"],
    "imageSlowLoad": ["5sec", "10sec"],
    "emailMemoryLeak": ["10x", "100x"],
    "kafkaQueueProblems": ["on"],
    "llmInaccurateResponse": ["on"],
    "llmRateLimitError": ["on"],
    "loadGeneratorFloodHomepage": ["on"],
}


def choose_random_failure(rng: random.Random) -> Tuple[str, str]:
    flag = rng.choice(list(FAIL_FLAGS.keys()))
    variant = rng.choice(FAIL_FLAGS[flag])
    return flag, variant


def run_one(label: str, flag: str, variant: str, duration: int):
    print(f"Starting run: {label} → {flag}={variant} ({duration}s) at {datetime.now().isoformat()}")
    gdc.set_flag(flag, variant)
    print("Collecting...")
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


def collect_split(split_name: str, rng: random.Random, duration: int, baselines: int, failures: int):
    # Baselines first (flag off)
    for i in range(1, baselines + 1):
        label = f"{split_name}{i:02d}_baseline"
        run_one(label, "cartFailure", "off", duration) #cartFailure is the default flag, set to off is collecting faul free baseline data
    # Failures
    for i in range(1, failures + 1):
        flag, variant = choose_random_failure(rng)
        label = f"{split_name}{i:02d}_{flag}_{variant}"
        run_one(label, flag, variant, duration)


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect randomized validation/test datasets")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--duration", type=int, default=180, help="Seconds per window")
    ap.add_argument("--val-baselines", type=int, default=1, help="Number of validation baselines")
    ap.add_argument("--val-failures", type=int, default=3, help="Number of validation failures")
    ap.add_argument("--test-baselines", type=int, default=1, help="Number of test baselines")
    ap.add_argument("--test-failures", type=int, default=3, help="Number of test failures")
    args = ap.parse_args()

    print("OTEL Random Collector starting..."+datetime.now().isoformat())
    print(f"Saving to: {gdc.OUTPUT_DIR.resolve()}"+datetime.now().isoformat())
    rng = random.Random(args.seed)

    collect_split("val", rng, args.duration, args.val_baselines, args.val_failures)
    collect_split("test", rng, args.duration, args.test_baselines, args.test_failures)

    print("Random collection complete at "+datetime.now().isoformat())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


