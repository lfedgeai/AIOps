#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime
import pandas as pd
 

# Defaults: local harness paths
HARNESS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = HARNESS_DIR / "out"
DEFAULT_DATASETS_DIR = HARNESS_DIR / "datasets"

BASE_DIR = Path(os.getenv("AIOPS_OUT_DIR", str(DEFAULT_OUT_DIR)))
DATA_DIR = Path(os.getenv("OTEL_DATASET_DIR", str(DEFAULT_DATASETS_DIR)))
OUT_FEATURES_CSV = Path(os.getenv("OTEL_OUT_FEATURES_CSV", str(BASE_DIR / "features.csv")))

def _compute_dataset_hash() -> str:
    import hashlib
    try:
        # Hash dataset path + metadata files list + mtimes
        h = hashlib.sha256()
        h.update(str(DATA_DIR.resolve()).encode())
        metas: List[Path] = []
        for p in DATA_DIR.rglob("metadata_*.json"):
            metas.append(p)
        metas.sort()
        for p in metas:
            try:
                st = p.stat()
                h.update(str(p.resolve()).encode())
                h.update(str(int(st.st_mtime)).encode())
                h.update(str(int(st.st_size)).encode())
            except Exception:
                continue
        # Include this script mtime to invalidate on code changes
        try:
            st_self = Path(__file__).resolve().stat()
            h.update(str(int(st_self.st_mtime)).encode())
        except Exception:
            pass
        return h.hexdigest()
    except Exception:
        return ""


@dataclass
class Sample:
    sample_id: str
    label: str
    is_baseline: bool
    root_cause: str
    flag: str
    variant: str
    paths: Dict[str, Path]


def discover_experiment_samples() -> List[Sample]:
    samples: List[Sample] = []
    patterns = [
        "metadata_exp*.json",
        "metadata_val*.json",
        "metadata_test*.json",
        "metadata_suite*.json",
        "metadata_eval*.json",
        "metadata_*baseline*.json",
        "metadata_train*.json",
    ]
    meta_paths_set = set()
    for pat in patterns:
        for p in DATA_DIR.rglob(pat):
            meta_paths_set.add(p.resolve())
    for meta_path in sorted(meta_paths_set):
        with open(meta_path) as f:
            meta = json.load(f)
        label = meta.get("label") or meta_path.stem.replace("metadata_", "")
        root_cause = meta.get("ground_truth_root_cause", "unknown")
        flag = meta.get("flag", "")
        variant = meta.get("variant", "")
        base_dir = meta_path.parent
        paths = {
            "metadata": meta_path,
            "logs": base_dir / f"logs_{label}.txt",
            "traces": base_dir / f"traces_{label}.json",
            "metrics": base_dir / f"metrics_{label}.json",
        }
        is_baseline = root_cause == "none"
        samples.append(
            Sample(
                sample_id=label,
                label=label,
                is_baseline=is_baseline,
                root_cause=root_cause,
                flag=flag,
                variant=variant,
                paths=paths,
            )
        )
    return samples


def count_log_errors(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    error_pattern = re.compile(
        r"(?is)(error[: ]|\berror\b|\bexception\b|\bfail(?:ed|ure)?\b|\bunavailable\b|\binternal\b|\b5\d{2}\b|\[error\]|level=error)"
    )
    infra_prefix = re.compile(r"^\s*(otel-collector|grafana|jaeger|flagd-ui|prometheus|opensearch)\b", re.IGNORECASE)
    count = 0
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            if infra_prefix.search(line):
                continue
            if error_pattern.search(line):
                count += 1
    return count


def count_traces(traces_path: Path) -> int:
    if not traces_path.exists():
        return 0
    try:
        with open(traces_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict) and "hits" in data and "hits" in data["hits"]:
            return len(data["hits"]["hits"])
        return 0
    except Exception:
        return 0


def trace_stats(traces_path: Path) -> Dict[str, int]:
    stats = {
        "trace_total_spans": 0,
        "trace_http_5xx_spans": 0,
        "trace_unique_services": 0,
        "trace_unique_span_names": 0,
    }
    if not traces_path.exists():
        return stats
    try:
        with open(traces_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "hits" in data and "hits" in data["hits"]:
            hits = data["hits"]["hits"]
        elif isinstance(data, list):
            hits = data
        else:
            hits = []
        services: Set[str] = set()
        span_names: Set[str] = set()
        total = 0
        http5xx = 0
        for h in hits:
            src = h.get("_source", {}) if isinstance(h, dict) else {}
            total += 1
            name = src.get("name") or src.get("spanName") or ""
            if not name:
                name = (src.get("attributes", {}) or {}).get("http.route") or ""
            if name:
                span_names.add(str(name))
            res = src.get("resource", {}) or {}
            rattrs = res.get("attributes", {}) or {}
            service_name = rattrs.get("service.name") or rattrs.get("service") or ""
            if not service_name:
                service_name = (src.get("attributes", {}) or {}).get("service.name") or ""
            if service_name:
                services.add(str(service_name))
            attrs = src.get("attributes", {}) or {}
            try:
                code = int(attrs.get("http.status_code") or -1)
                if code >= 500:
                    http5xx += 1
            except Exception:
                pass
        stats["trace_total_spans"] = total
        stats["trace_http_5xx_spans"] = http5xx
        stats["trace_unique_services"] = len(services)
        stats["trace_unique_span_names"] = len(span_names)
        return stats
    except Exception:
        return stats


def count_metric_series(metrics_path: Path) -> int:
    if not metrics_path.exists():
        return 0
    try:
        with open(metrics_path) as f:
            data = json.load(f)
        total = 0
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    total += len(v)
        return total
    except Exception:
        return 0


def metrics_scalar_sum(metrics_path: Path, key: str) -> float:
    if not metrics_path.exists():
        return 0.0
    try:
        with open(metrics_path) as f:
            data = json.load(f)
        entries = data.get(key, [])
        total = 0.0
        if isinstance(entries, list):
            for e in entries:
                v = e.get("value") if isinstance(e, dict) else None
                if isinstance(v, list) and len(v) >= 2:
                    try:
                        total += float(v[1])
                    except Exception:
                        continue
        return total
    except Exception:
        return 0.0


def count_log_lines(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    try:
        with open(log_path, "r", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def count_service_errors(log_path: Path, service_prefix: str) -> int:
    if not log_path.exists():
        return 0
    error_pattern = re.compile(
        r"(?is)(error[: ]|\berror\b|\bexception\b|\bfail(?:ed|ure)?\b|\bunavailable\b|\binternal\b|\b5\d{2}\b|\[error\]|level=error)"
    )
    infra_prefix = re.compile(r"^\s*(otel-collector|grafana|jaeger|flagd-ui|prometheus|opensearch)\b", re.IGNORECASE)
    svc_prefix = re.compile(rf"^\s*{re.escape(service_prefix)}\b", re.IGNORECASE)
    count = 0
    try:
        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                if infra_prefix.search(line):
                    continue
                if not svc_prefix.search(line):
                    continue
                if error_pattern.search(line):
                    count += 1
        return count
    except Exception:
        return 0


def build_features(samples: List[Sample]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for s in samples:
        trace_count = count_traces(s.paths["traces"])
        tstats = trace_stats(s.paths["traces"])
        log_error_count = count_log_errors(s.paths["logs"])
        total_log_lines = count_log_lines(s.paths["logs"])
        log_error_ratio = (log_error_count / total_log_lines) if total_log_lines else 0.0
        fe = count_service_errors(s.paths["logs"], "frontend")
        co = count_service_errors(s.paths["logs"], "checkout")
        rc = count_service_errors(s.paths["logs"], "recommendation")
        pc = count_service_errors(s.paths["logs"], "product-catalog")
        py = count_service_errors(s.paths["logs"], "payment")
        metrics_series_len = count_metric_series(s.paths["metrics"])
        req_rate_sum = metrics_scalar_sum(s.paths["metrics"], "req_rate")
        label_lower = s.label.lower()
        in_training = s.is_baseline and (label_lower.startswith("train") or "_train" in label_lower)
        rows.append(
            {
                "id": s.sample_id,
                "label": s.label,
                "flag": s.flag,
                "variant": s.variant,
                "root_cause": s.root_cause,
                "is_baseline": s.is_baseline,
                "in_training": in_training,
                "trace_count": trace_count,
                "trace_total_spans": tstats["trace_total_spans"],
                "trace_http_5xx_spans": tstats["trace_http_5xx_spans"],
                "trace_unique_services": tstats["trace_unique_services"],
                "trace_unique_span_names": tstats["trace_unique_span_names"],
                "log_error_count": log_error_count,
                "log_total_lines": total_log_lines,
                "log_error_ratio": log_error_ratio,
                "frontend_error_count": fe,
                "checkout_error_count": co,
                "recommendation_error_count": rc,
                "product_catalog_error_count": pc,
                "payment_error_count": py,
                "metrics_series_len": metrics_series_len,
                "req_rate_sum": req_rate_sum,
            }
        )
    df = pd.DataFrame(rows)
    return df


def validate_dataset(samples: List[Sample], df: pd.DataFrame) -> Dict[str, object]:
    total = len(samples)
    num_train = int(df["in_training"].sum()) if "in_training" in df.columns else 0
    num_eval = int(total - num_train)
    missing_logs = 0
    missing_traces = 0
    missing_metrics = 0
    empty_logs = 0
    zero_traces = 0
    for s in samples:
        lp, tp, mp = s.paths["logs"], s.paths["traces"], s.paths["metrics"]
        if not lp.exists():
            missing_logs += 1
        if not tp.exists():
            missing_traces += 1
        if not mp.exists():
            missing_metrics += 1
        # Empty/zero content checks using existing counters
    try:
        empty_logs = int((df.get("log_total_lines", 0) == 0).sum())
    except Exception:
        empty_logs = 0
    try:
        zero_traces = int((df.get("trace_count", 0) == 0).sum())
    except Exception:
        zero_traces = 0
    return {
        "total_samples": int(total),
        "num_train": int(num_train),
        "num_eval": int(num_eval),
        "missing_files": {
            "logs": int(missing_logs),
            "traces": int(missing_traces),
            "metrics": int(missing_metrics),
        },
        "empties": {
            "logs_zero_lines": int(empty_logs),
            "traces_zero_count": int(zero_traces),
        },
    }


def main() -> int:
    samples = discover_experiment_samples()
    # Caching: skip rebuild if dataset+config unchanged and features.csv exists
    cache_path = BASE_DIR / "features.cache.json"
    cache_key = _compute_dataset_hash()
    try:
        if OUT_FEATURES_CSV.exists() and cache_path.exists():
            import json as _json
            cached = _json.loads(cache_path.read_text())
            if cached.get("key") == cache_key:
                print(f"[cache] features up-to-date → {OUT_FEATURES_CSV}")
                return 0
    except Exception:
        pass
    df = build_features(samples)
    try:
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_FEATURES_CSV, index=False)
        # Write validation summary for downstream reports
        try:
            (BASE_DIR / "dataset_validation.json").write_text(json.dumps(validate_dataset(samples, df), indent=2))
        except Exception:
            pass
        # Update cache
        try:
            import json as _json
            cache_path.write_text(_json.dumps({"key": cache_key, "features_csv": str(OUT_FEATURES_CSV)}, indent=2))
        except Exception:
            pass
        print(f"Features saved → {OUT_FEATURES_CSV}")
        return 0
    except Exception as e:
        print(f"[ERROR] failed to write features: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


