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
from joblib import dump

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

# Defaults: local harness paths
HARNESS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = HARNESS_DIR / "out"
DEFAULT_DATASETS_DIR = HARNESS_DIR / "datasets"

BASE_DIR = Path(os.getenv("AIOPS_OUT_DIR", str(DEFAULT_OUT_DIR)))
DATA_DIR = Path(os.getenv("OTEL_DATASET_DIR", str(DEFAULT_DATASETS_DIR)))
OUT_FEATURES_CSV = Path(os.getenv("OTEL_OUT_FEATURES_CSV", str(BASE_DIR / "features.csv")))
OUT_REPORT_JSON = Path(os.getenv("OTEL_OUT_REPORT_JSON", str(BASE_DIR / "report_isoforest.json")))

# Allow slight noise in training baselines when clean data is scarce
TRAIN_LOG_ERROR_MAX = int(os.getenv("TRAIN_LOG_ERROR_MAX", "3"))
THRESHOLD_QUANTILE = float(os.getenv("THRESHOLD_QUANTILE", "0.995"))


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


def train_and_evaluate(df: pd.DataFrame) -> Dict[str, object]:
    if "in_training" not in df.columns:
        df["in_training"] = False
    train_df = df[df["in_training"] == True].copy()  # noqa: E712
    if {"is_baseline", "log_error_count"}.issubset(train_df.columns):
        train_df = train_df[
            (train_df["is_baseline"] == True)  # noqa: E712
            & (train_df["log_error_count"].astype(int) <= TRAIN_LOG_ERROR_MAX)
        ]
    eval_df = df[df["in_training"] == False].copy()  # noqa: E712

    if {"is_baseline", "log_error_count"}.issubset(eval_df.columns):
        mask_bad = (eval_df["is_baseline"] == True) & (eval_df["log_error_count"].astype(int) > 3)  # noqa: E712
        eval_df = eval_df[~mask_bad]

    if len(train_df) < 5:
        return {
            "ok": False,
            "error": f"Not enough baseline samples for training: {len(train_df)} (need >=5)",
        }

    feature_cols = [
        "trace_count",
        "trace_total_spans",
        "trace_http_5xx_spans",
        "trace_unique_services",
        "trace_unique_span_names",
        "log_error_count",
        "log_total_lines",
        "log_error_ratio",
        "frontend_error_count",
        "checkout_error_count",
        "recommendation_error_count",
        "product_catalog_error_count",
        "payment_error_count",
        "metrics_series_len",
        "req_rate_sum",
    ]

    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return {
            "ok": False,
            "error": "scikit-learn not installed. Please run: pip3 install --user scikit-learn",
        }

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )
    model.fit(train_df[feature_cols])

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, str(BASE_DIR / f"IsolationForestModel_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"))

    eval_df = eval_df.copy()
    eval_df["pred"] = model.predict(eval_df[feature_cols])  # +1 normal, -1 anomaly
    eval_df["anomaly_score"] = -model.decision_function(eval_df[feature_cols])
    # Threshold calibrated on train baselines (common operating point)
    scores_train = -model.decision_function(train_df[feature_cols])
    threshold = float(pd.Series(scores_train).quantile(THRESHOLD_QUANTILE))

    y_true = (eval_df["root_cause"] != "none").astype(int)
    y_pred = (eval_df["pred"] == -1).astype(int)
    y_pred_thresh = (eval_df["anomaly_score"] > threshold).astype(int)

    scores = eval_df["anomaly_score"].astype(float)
    roc_auc = None
    pr_auc = None
    curves: Dict[str, Dict[str, List[float]]] = {}
    try:
        if y_true.nunique() == 2:
            roc_auc = float(roc_auc_score(y_true, scores))
            pr_auc = float(average_precision_score(y_true, scores))
            prec, rec, _ = precision_recall_curve(y_true, scores)
            fpr, tpr, _ = roc_curve(y_true, scores)
            curves = {
                "pr": {"precision": list(map(float, prec)), "recall": list(map(float, rec))},
                "roc": {"fpr": list(map(float, fpr)), "tpr": list(map(float, tpr))},
            }
    except Exception:
        pass

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision_internal = tp / (tp + fp) if (tp + fp) else 0.0
    recall_internal = tp / (tp + fn) if (tp + fn) else 0.0
    f1_internal = (2 * precision_internal * recall_internal / (precision_internal + recall_internal)) if (precision_internal + recall_internal) else 0.0
    fpr_internal = fp / (fp + tn) if (fp + tn) else 0.0
    tnr_internal = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy_internal = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    # Metrics at common operating point (threshold)
    tp_t = int(((y_true == 1) & (y_pred_thresh == 1)).sum())
    fp_t = int(((y_true == 0) & (y_pred_thresh == 1)).sum())
    tn_t = int(((y_true == 0) & (y_pred_thresh == 0)).sum())
    fn_t = int(((y_true == 1) & (y_pred_thresh == 0)).sum())
    precision = tp_t / (tp_t + fp_t) if (tp_t + fp_t) else 0.0
    recall = tp_t / (tp_t + fn_t) if (tp_t + fn_t) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp_t / (fp_t + tn_t) if (fp_t + tn_t) else 0.0
    tnr = tn_t / (tn_t + fp_t) if (tn_t + fp_t) else 0.0
    accuracy = (tp_t + tn_t) / (tp_t + tn_t + fp_t + fn_t) if (tp_t + tn_t + fp_t + fn_t) else 0.0

    # Persist features
    df.to_csv(OUT_FEATURES_CSV, index=False)

    report = {
        "ok": True,
        "num_train": int(len(train_df)),
        "num_eval": int(len(eval_df)),
        # Confusion at threshold operating point
        "confusion_matrix": {"tp": tp_t, "fp": fp_t, "tn": tn_t, "fn": fn_t},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "true_negative_rate": tnr,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "curves": curves,
        "threshold": threshold,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "metrics_at_internal": {
            "precision": precision_internal,
            "recall": recall_internal,
            "f1": f1_internal,
            "false_positive_rate": fpr_internal,
            "true_negative_rate": tnr_internal,
            "accuracy": accuracy_internal,
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        },
        "feature_cols": feature_cols,
        "feature_importances": None,
        "feature_importances_ranked": None,
        "features_csv": str(OUT_FEATURES_CSV),
    }
    with open(OUT_REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> int:
    samples = discover_experiment_samples()
    df = build_features(samples)
    report = train_and_evaluate(df)
    if not report.get("ok", False):
        print(f"[ERROR] {report.get('error')}")
        try:
            df.to_csv(OUT_FEATURES_CSV, index=False)
            print(f"Features saved → {OUT_FEATURES_CSV}")
        except Exception:
            pass
        return 1
    print("IsolationForest report:")
    print(json.dumps(report, indent=2))
    print(f"Features saved → {OUT_FEATURES_CSV}")
    print(f"Report saved   → {OUT_REPORT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


