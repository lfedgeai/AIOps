#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "aiops_harness"
# Keep all OTEL demo artifacts under chaos_engineering/
CHAOS_DIR = Path(__file__).resolve().parent
GROUND = CHAOS_DIR / "ground_truth_data"
DATA = CHAOS_DIR / "otel-demo" / "otel_ground_truth_data"
DATASETS = HARNESS / "datasets"
OUT_ROOT = HARNESS / "out"


def _run(cmd: List[str]) -> int:
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def collect_baselines(windows: int, cooldown: int, duration: int, keep: bool) -> None:
    sys.path.append(str(GROUND))
    # Ensure we can import project modules (e.g., feature_pipeline) when invoked from this subfolder
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import requests  # noqa
    import ground_truth_data_collector as gdc  # type: ignore
    from feature_pipeline import count_log_errors  # type: ignore

    READ = "http://localhost:8080/feature/api/read"
    WRITE = "http://localhost:8080/feature/api/write"
    FLAGS = [
        "cartFailure", "paymentFailure", "paymentUnreachable", "adFailure", "adHighCpu", "adManualGc",
        "productCatalogFailure", "recommendationCacheFailure", "imageSlowLoad", "emailMemoryLeak",
        "kafkaQueueProblems", "llmInaccurateResponse", "llmRateLimitError", "loadGeneratorFloodHomepage",
    ]

    def set_all_off() -> None:
        try:
            r = requests.get(READ, timeout=10); r.raise_for_status()
            flags = (r.json() or {}).get("flags", {})
            for v in flags.values():
                if isinstance(v, dict):
                    v["defaultVariant"] = "off"
            requests.post(WRITE, json={"data": {"$schema": "https://flagd.dev/schema/v0/flags.json", "flags": flags}}, timeout=10)
        except Exception:
            pass
        for f in FLAGS:
            try:
                gdc.set_flag(f, "off")
            except Exception:
                pass

    DATA.mkdir(parents=True, exist_ok=True)
    print(f"== baseline collector (non-deleting) ==", flush=True)
    for i in range(1, windows + 1):
        label = f"train{i:02d}_baseline"
        print(f"Collecting {label} (cooldown {cooldown}s + window {duration}s)", flush=True)
        set_all_off()
        time.sleep(cooldown)
        start = datetime.now(timezone.utc)
        time.sleep(duration)
        gdc.export_traces_from_opensearch(label, start)
        gdc.export_prometheus_metrics(label)
        gdc.export_logs(label, start)
        (gdc.OUTPUT_DIR / f"metadata_{label}.json").write_text(
            f'{{"label":"{label}","flag":"baseline","variant":"off","start_time":"{datetime.now().isoformat()}","duration_seconds":{duration},"ground_truth_root_cause":"none"}}'
        )
        lec = count_log_errors(DATA / f"logs_{label}.txt")
        print(f"{label} log_error_count={lec} (kept)", flush=True)


def run_suite() -> None:
    _run([sys.executable, str(GROUND / "collect_full_suite.py")])


def build_features() -> None:
    env = os.environ.copy()
    env.setdefault("TRAIN_LOG_ERROR_MAX", env.get("TRAIN_LOG_ERROR_MAX", "3"))
    _run([sys.executable, str(GROUND / "feature_pipeline.py")])


def train_eval() -> None:
    build_features()


def visualize() -> None:
    _run([sys.executable, str(GROUND / "visualize_isoforest.py")])


def permute() -> None:
    env = os.environ.copy()
    env.setdefault("PERM_REPEATS", env.get("PERM_REPEATS", "7"))
    env.setdefault("PERM_SEED", env.get("PERM_SEED", "42"))
    _run([sys.executable, str(GROUND / "feature_permutation.py")])
    _run([sys.executable, str(GROUND / "visualize_permutation.py")])


def _resolve_dataset_path(dataset: str | None) -> Path:
    DATASETS.mkdir(parents=True, exist_ok=True)
    if dataset:
        p = Path(dataset)
        if p.exists():
            return p
        return DATASETS / dataset
    subs = [p for p in DATASETS.iterdir() if p.is_dir()]
    if not subs:
        return DATA
    subs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subs[0]


def decoupled_features(dataset: str | None) -> Path:
    ds_path = _resolve_dataset_path(dataset)
    ds_name = ds_path.name
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = OUT_ROOT / ds_name / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OTEL_DATASET_DIR"] = str(ds_path)
    os.environ["OTEL_OUT_FEATURES_CSV"] = str(out_dir / "features.csv")
    os.environ["OTEL_OUT_REPORT_JSON"] = str(out_dir / "report_isoforest.json")
    print(f"[features] dataset={ds_path} out={out_dir}")
    _run([sys.executable, str(GROUND / "feature_pipeline.py")])
    return out_dir


def decoupled_visualize(features_csv: Path, report_json: Path, out_html: Path) -> None:
    os.environ["OTEL_FEATURES_CSV"] = str(features_csv)
    os.environ["OTEL_REPORT_JSON"] = str(report_json)
    os.environ["OTEL_OUT_HTML"] = str(out_html)
    print(f"[visualize] {out_html}")
    _run([sys.executable, str(GROUND / "visualize_isoforest.py")])


def main() -> int:
    p = argparse.ArgumentParser(description="Chaos Orchestrator CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("collect-baselines", help="Collect non-deleting baseline windows")
    b.add_argument("--windows", type=int, default=5)
    b.add_argument("--cooldown", type=int, default=60)
    b.add_argument("--duration", type=int, default=60)
    b.add_argument("--keep", action="store_true", default=True, help="Keep windows (always True)")

    sub.add_parser("suite", help="Run full suite (flagd faults)")
    f = sub.add_parser("features", help="Build features (decoupled dataset)")
    f.add_argument("--dataset", type=str, default=None, help="Dataset name under aiops_harness/datasets or a path")
    te = sub.add_parser("train-eval", help="Train and evaluate model (decoupled dataset)")
    te.add_argument("--dataset", type=str, default=None, help="Dataset name under aiops_harness/datasets or a path")
    v = sub.add_parser("visualize", help="Generate HTML report (decoupled dataset)")
    v.add_argument("--dataset", type=str, default=None, help="Dataset name under aiops_harness/datasets or a path")
    sub.add_parser("permute", help="Permutation importances + HTML")

    args = p.parse_args()
    if args.cmd == "collect-baselines":
        collect_baselines(args.windows, args.cooldown, args.duration, args.keep)
    elif args.cmd == "suite":
        run_suite()
    elif args.cmd == "features":
        out_dir = decoupled_features(getattr(args, "dataset", None))
        decoupled_visualize(out_dir / "features.csv", out_dir / "report_isoforest.json", out_dir / "report_isoforest.html")
    elif args.cmd == "train-eval":
        out_dir = decoupled_features(getattr(args, "dataset", None))
        decoupled_visualize(out_dir / "features.csv", out_dir / "report_isoforest.json", out_dir / "report_isoforest.html")
    elif args.cmd == "visualize":
        ds_path = _resolve_dataset_path(getattr(args, "dataset", None))
        base = OUT_ROOT / ds_path.name
        if base.exists():
            outs = [p for p in base.iterdir() if p.is_dir()]
            outs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if outs:
                out_dir = outs[0]
                decoupled_visualize(out_dir / "features.csv", out_dir / "report_isoforest.json", out_dir / "report_isoforest.html")
            else:
                print(f"[visualize] no outputs found under {base}")
        else:
            print(f"[visualize] no outputs found for dataset {ds_path.name}")
    elif args.cmd == "permute":
        permute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

