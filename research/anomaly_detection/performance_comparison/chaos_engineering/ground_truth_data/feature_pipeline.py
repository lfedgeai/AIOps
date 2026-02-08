#!/usr/bin/env python3
"""
IsolationForest baseline on OTEL ground-truth dataset.
Builds simple features from logs/metrics/traces and evaluates anomaly detection.
"""

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

BASE_DIR = Path("/home/redhat/git/OTEL/otel-demo/otel_ground_truth_data")
# Allow reading dataset inputs from a different directory while keeping outputs in BASE_DIR
# If not provided, default to BASE_DIR for both.
DATA_DIR = Path(os.getenv("OTEL_DATASET_DIR", str(BASE_DIR)))
OUT_FEATURES_CSV = Path(os.getenv("OTEL_OUT_FEATURES_CSV", str(BASE_DIR / "features.csv")))
OUT_REPORT_JSON = Path(os.getenv("OTEL_OUT_REPORT_JSON", str(BASE_DIR / "report_isoforest.json")))

import os
# Allow slight noise in training baselines when clean data is scarce
# Override via env TRAIN_LOG_ERROR_MAX (default 0 for strict, relaxed e.g. 3)
TRAIN_LOG_ERROR_MAX = int(os.getenv("TRAIN_LOG_ERROR_MAX", "3"))
THRESHOLD_QUANTILE = float(os.getenv("THRESHOLD_QUANTILE", "0.995"))


@dataclass
class Sample:
    sample_id: str
    label: str  # same as sample_id for experiment labels; for baselines: baseline_XX
    is_baseline: bool
    root_cause: str
    flag: str
    variant: str
    paths: Dict[str, Path]


def discover_experiment_samples() -> List[Sample]:
    samples: List[Sample] = []
    # Include experiments, validation/test, and any explicit baseline-labeled files
    patterns = [
        "metadata_exp*.json",
        "metadata_val*.json",
        "metadata_test*.json",
        "metadata_suite*.json",  # legacy name
        "metadata_eval*.json",   # preferred name
        "metadata_*baseline*.json",
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
        # Look for sibling files in the same directory as metadata to support nested layouts
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


def discover_additional_baselines() -> List[Sample]:
    baselines_dir = BASE_DIR / "baselines"
    if not baselines_dir.exists():
        return []
    samples: List[Sample] = []
    for i in range(1, 6):
        suf = f"{i:02d}"
        label = f"baseline_{suf}"
        paths = {
            "metadata": baselines_dir / f"metadata_baseline_{suf}.json",
            "logs": baselines_dir / f"logs_baseline_{suf}.txt",
            "traces": baselines_dir / f"traces_baseline_{suf}.json",
            "metrics": baselines_dir / f"metrics_baseline_{suf}.json",
        }
        if not all(p.exists() for p in paths.values()):
            continue
        try:
            with open(paths["metadata"]) as f:
                meta = json.load(f)
        except Exception:
            meta = {}
        samples.append(
            Sample(
                sample_id=label,
                label=label,
                is_baseline=True,
                root_cause="none",
                flag=meta.get("flag", "baseline"),
                variant=meta.get("variant", "off"),
                paths=paths,
            )
        )
    return samples


def count_log_errors(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    # Expanded, case-insensitive patterns and bracketed/quoted markers, plus 'Error:' forms
    error_pattern = re.compile(
        r"(?is)(error[: ]|\\berror\\b|\\bexception\\b|\\bfail(?:ed|ure)?\\b|\\bunavailable\\b|\\binternal\\b|\\b5\\d{2}\\b|\\[error\\]|level=error)"
    )
    # Exclude infra/service noise prefixes (allow leading spaces)
    infra_prefix = re.compile(r"^\\s*(otel-collector|grafana|jaeger|flagd-ui|prometheus|opensearch)\\b", re.IGNORECASE)
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
        # traces export is a list of hits (OpenSearch search response hits)
        if isinstance(data, list):
            return len(data)
        # sometimes saved as full OS response; be permissive
        if isinstance(data, dict) and "hits" in data and "hits" in data["hits"]:
            return len(data["hits"]["hits"])
        return 0
    except Exception:
        return 0


def trace_stats(traces_path: Path) -> Dict[str, int]:
    """
    Compute richer statistics from traces export:
      - total_spans
      - http_5xx_spans
      - unique_services
      - unique_span_names
    """
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
            # span name
            name = src.get("name") or src.get("spanName") or ""
            if not name:
                # try http.route as a fallback proxy for name
                name = (src.get("attributes", {}) or {}).get("http.route") or ""
            if name:
                span_names.add(str(name))
            # service name (typical locations)
            res = src.get("resource", {}) or {}
            rattrs = res.get("attributes", {}) or {}
            service_name = rattrs.get("service.name") or rattrs.get("service") or ""
            if not service_name:
                # sometimes stored directly under attributes
                service_name = (src.get("attributes", {}) or {}).get("service.name") or ""
            if service_name:
                services.add(str(service_name))
            # http status
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
    """
    For instant-vector Prometheus /query results saved in metrics JSON,
    compute the scalar sum for a given query key by summing numeric 'value'.
    """
    if not metrics_path.exists():
        return 0.0
    try:
        with open(metrics_path) as f:
            data = json.load(f)
        entries = data.get(key, [])
        total = 0.0
        if isinstance(entries, list):
            for e in entries:
                # result shape: {'metric': {...}, 'value': [ts, "number"]}
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
    """
    Count error-like lines for a specific service based on prefix heuristics.
    """
    if not log_path.exists():
        return 0
    error_pattern = re.compile(
        r"(?is)(error[: ]|\\berror\\b|\\bexception\\b|\\bfail(?:ed|ure)?\\b|\\bunavailable\\b|\\binternal\\b|\\b5\\d{2}\\b|\\[error\\]|level=error)"
    )
    infra_prefix = re.compile(r"^\\s*(otel-collector|grafana|jaeger|flagd-ui|prometheus|opensearch)\\b", re.IGNORECASE)
    svc_prefix = re.compile(rf"^\\s*{re.escape(service_prefix)}\\b", re.IGNORECASE)
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
        # richer trace stats
        tstats = trace_stats(s.paths["traces"])
        log_error_count = count_log_errors(s.paths["logs"])
        total_log_lines = count_log_lines(s.paths["logs"])
        log_error_ratio = (log_error_count / total_log_lines) if total_log_lines else 0.0
        # per-service error counts (selected core services)
        fe = count_service_errors(s.paths["logs"], "frontend")
        co = count_service_errors(s.paths["logs"], "checkout")
        rc = count_service_errors(s.paths["logs"], "recommendation")
        pc = count_service_errors(s.paths["logs"], "product-catalog")
        py = count_service_errors(s.paths["logs"], "payment")
        metrics_series_len = count_metric_series(s.paths["metrics"])
        # metrics scalar sums for req_rate
        req_rate_sum = metrics_scalar_sum(s.paths["metrics"], "req_rate")
        # Training selection: only baselines explicitly labeled for training
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
    # Separate training (explicit train baselines) and evaluation (everything else)
    if "in_training" not in df.columns:
        df["in_training"] = False
    train_df = df[df["in_training"] == True].copy()  # noqa: E712
    # Training hygiene: only train on (near) clean baselines
    if {"is_baseline", "log_error_count"}.issubset(train_df.columns):
        train_df = train_df[
            (train_df["is_baseline"] == True)  # noqa: E712
            & (train_df["log_error_count"].astype(int) <= TRAIN_LOG_ERROR_MAX)
        ]
    eval_df = df[df["in_training"] == False].copy()  # noqa: E712

    # Keep eval baselines even with a few benign errors (<=3)
    if {"is_baseline", "log_error_count"}.issubset(eval_df.columns):
        mask_bad = (eval_df["is_baseline"] == True) & (eval_df["log_error_count"].astype(int) > 3)  # noqa: E712
        eval_df = eval_df[~mask_bad]

    if len(train_df) < 5:
        return {
            "ok": False,
            "error": f"Not enough baseline samples for training: {len(train_df)} (need >=5)",
        }

    # Keep schema minimal and independent of service names
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

    dump(model, f"{BASE_DIR}/IsolationForestModel_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl")

    # Tree-based impurity importances (if available)
    importances: List[float] = []
    ranked_importances: List[Tuple[str, float]] = []
    try:
        imp = getattr(model, "feature_importances_", None)
        if imp is not None:
            importances = [float(x) for x in imp]
            ranked_importances = sorted(
                list(zip(feature_cols, importances)),
                key=lambda t: t[1],
                reverse=True,
            )
    except Exception:
        pass

    # Predict on eval set
    eval_df = eval_df.copy()
    eval_df["pred"] = model.predict(eval_df[feature_cols])  # +1 normal, -1 anomaly
    eval_df["anomaly_score"] = -model.decision_function(eval_df[feature_cols])
    # Common operating point threshold from train baselines
    scores_train = -model.decision_function(train_df[feature_cols])
    threshold = float(pd.Series(scores_train).quantile(THRESHOLD_QUANTILE))
    y_pred_thresh = (eval_df["anomaly_score"] > threshold).astype(int)

    # Ground truth: anomaly if root_cause != "none"
    y_true = (eval_df["root_cause"] != "none").astype(int)
    y_pred = (eval_df["pred"] == -1).astype(int)

    # Curve-based metrics (require both classes present)
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

    tp_int = int(((y_true == 1) & (y_pred == 1)).sum())
    fp_int = int(((y_true == 0) & (y_pred == 1)).sum())
    tn_int = int(((y_true == 0) & (y_pred == 0)).sum())
    fn_int = int(((y_true == 1) & (y_pred == 0)).sum())

    precision_internal = tp_int / (tp_int + fp_int) if (tp_int + fp_int) else 0.0
    recall_internal = tp_int / (tp_int + fn_int) if (tp_int + fn_int) else 0.0
    f1_internal = (2 * precision_internal * recall_internal / (precision_internal + recall_internal)) if (precision_internal + recall_internal) else 0.0
    fpr_internal = fp_int / (fp_int + tn_int) if (fp_int + tn_int) else 0.0
    tnr_internal = tn_int / (tn_int + fp_int) if (tn_int + fp_int) else 0.0
    accuracy_internal = (tp_int + tn_int) / (tp_int + tn_int + fp_int + fn_int) if (tp_int + tn_int + fp_int + fn_int) else 0.0

    tp = int(((y_true == 1) & (y_pred_thresh == 1)).sum())
    fp = int(((y_true == 0) & (y_pred_thresh == 1)).sum())
    tn = int(((y_true == 0) & (y_pred_thresh == 0)).sum())
    fn = int(((y_true == 1) & (y_pred_thresh == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    # Additional scalar metrics
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    # Persist features
    df.to_csv(OUT_FEATURES_CSV, index=False)

    report = {
        "ok": True,
        "num_train": int(len(train_df)),
        "num_eval": int(len(eval_df)),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
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
            "confusion_matrix": {"tp": tp_int, "fp": fp_int, "tn": tn_int, "fn": fn_int},
        },
        "feature_cols": feature_cols,
        "feature_importances": dict(ranked_importances) if ranked_importances else None,
        "feature_importances_ranked": ranked_importances if ranked_importances else None,
        "features_csv": str(OUT_FEATURES_CSV),
    }
    with open(OUT_REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> int:
    exp_samples = discover_experiment_samples()
    extra_baselines = discover_additional_baselines()

    # Merge: experiments + additional baselines for training
    all_samples = list(exp_samples) + list(extra_baselines)
    df = build_features(all_samples)

    report = train_and_evaluate(df)
    if not report.get("ok", False):
        print(f"[ERROR] {report.get('error')}")
        # Still write features for inspection
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


