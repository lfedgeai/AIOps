#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
import pickle as pkl
import re


@dataclass
class DetectorResult:
    name: str
    precision: float
    recall: float
    f1: float
    pr_auc: Optional[float]
    roc_auc: Optional[float]
    threshold: float
    pr_curve: Optional[Tuple[List[float], List[float]]] = None  # (recall, precision)
    roc_curve: Optional[Tuple[List[float], List[float]]] = None  # (fpr, tpr)


def load_data(features_csv: Path, isoforest_report_json: Optional[Path]) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(features_csv)
    # Prefer feature_cols from isoforest report to ensure column order consistency
    feature_cols: List[str] = []
    if isoforest_report_json and isoforest_report_json.exists():
        try:
            with open(isoforest_report_json) as f:
                rep = json.load(f)
                if isinstance(rep.get("feature_cols"), list):
                    feature_cols = [str(c) for c in rep["feature_cols"]]
        except Exception:
            pass
    if not feature_cols:
        # Fallback: infer sensible numeric feature columns
        exclude = {
            "id", "label", "flag", "variant", "root_cause", "is_baseline", "in_training",
        }
        for col in df.columns:
            if col in exclude:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
    return df, feature_cols


def split_modalities(feature_cols: List[str]) -> Dict[str, List[str]]:
    """
    Heuristically split feature columns into modalities: logs, traces, metrics.
    Rules:
      - traces: columns starting with 'trace_'
      - logs: starting with 'log_' or ending with '_error_count'/'_error_ratio'
      - metrics: starting with 'metrics_' or common metric prefixes like 'req_', 'cpu_', 'memory_', 'latency_'
    """
    logs: List[str] = []
    traces: List[str] = []
    metrics: List[str] = []
    for c in feature_cols:
        lc = c.lower()
        if lc.startswith("trace_"):
            traces.append(c)
            continue
        if lc.startswith("log_") or lc.endswith("_error_count") or lc.endswith("_error_ratio"):
            logs.append(c)
            continue
        if lc.startswith("metrics_") or lc.startswith("metric_") or lc.startswith("req_") or lc.startswith("cpu_") or lc.startswith("memory_") or lc.startswith("latency_"):
            metrics.append(c)
            continue
    return {"logs": logs, "traces": traces, "metrics": metrics}


def split_modalities(feature_cols: List[str]) -> Dict[str, List[str]]:
    """
    Heuristically split feature columns into modalities: logs, traces, metrics.
    Rules:
      - traces: columns starting with 'trace_'
      - logs: starting with 'log_' or ending with '_error_count'/'_error_ratio'
      - metrics: starting with 'metrics_' or common metric prefixes like 'req_', 'cpu_', 'memory_', 'latency_'
    """
    logs: List[str] = []
    traces: List[str] = []
    metrics: List[str] = []
    for c in feature_cols:
        lc = c.lower()
        if lc.startswith("trace_"):
            traces.append(c)
            continue
        if lc.startswith("log_") or lc.endswith("_error_count") or lc.endswith("_error_ratio"):
            logs.append(c)
            continue
        if lc.startswith("metrics_") or lc.startswith("metric_") or lc.startswith("req_") or lc.startswith("cpu_") or lc.startswith("memory_") or lc.startswith("latency_"):
            metrics.append(c)
            continue
    return {"logs": logs, "traces": traces, "metrics": metrics}


def split_train_eval(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Train on baseline windows marked for training
    if "in_training" not in df.columns:
        df["in_training"] = False
    train_df = df[df["in_training"] == True].copy()  # noqa: E712
    eval_df = df[df["in_training"] == False].copy()  # noqa: E712
    # Ground truth: 1 if anomaly (fault), 0 if baseline
    y_eval = (eval_df["root_cause"] != "none").astype(int).to_numpy()
    X_train = train_df[feature_cols].to_numpy()
    X_eval = eval_df[feature_cols].to_numpy()
    return X_train, X_eval, y_eval


def calibrate_threshold(scores_train: np.ndarray, target_quantile: float = 0.995) -> float:
    target_quantile = min(max(target_quantile, 0.5), 0.9999)
    return float(np.quantile(scores_train, target_quantile))


def compute_metrics(y_true: np.ndarray, scores_eval: np.ndarray, threshold: float, higher_is_anomalous: bool = True) -> Tuple[float, float, float, Optional[float], Optional[float], Optional[Tuple[List[float], List[float]]], Optional[Tuple[List[float], List[float]]]]:
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
    preds = (scores_eval > threshold).astype(int) if higher_is_anomalous else (scores_eval < threshold).astype(int)
    tp = int(((y_true == 1) & (preds == 1)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    fn = int(((y_true == 1) & (preds == 0)).sum())
    tn = int(((y_true == 0) & (preds == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    pr_auc: Optional[float] = None
    roc_auc: Optional[float] = None
    pr_pts: Optional[Tuple[List[float], List[float]]] = None
    roc_pts: Optional[Tuple[List[float], List[float]]] = None
    try:
        if len(np.unique(y_true)) == 2:
            pr_auc = float(average_precision_score(y_true, scores_eval))
            roc_auc = float(roc_auc_score(y_true, scores_eval))
            pr_p, pr_r, _ = precision_recall_curve(y_true, scores_eval)
            fpr, tpr, _ = roc_curve(y_true, scores_eval)
            pr_pts = (list(map(float, pr_r)), list(map(float, pr_p)))  # x=recall, y=precision
            roc_pts = (list(map(float, fpr)), list(map(float, tpr)))   # x=fpr, y=tpr
    except Exception:
        pass
    return precision, recall, f1, pr_auc, roc_auc, pr_pts, roc_pts


def run_iforest_ext(X_train: np.ndarray, X_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str, object]:
    """
    Try Extended Isolation Forest via isotree; fall back to sklearn IsolationForest if unavailable.
    Returns (train_scores, eval_scores, name)
    """
    try:
        from isotree import IsolationForest as EIF
        model = EIF(n_estimators=200, random_state=42)
        model.fit(X_train)
        # isotree outputs higher score => more anomalous via predict_scores
        s_train = model.predict_scores(X_train)
        s_eval = model.predict_scores(X_eval)
        return np.asarray(s_train), np.asarray(s_eval), "ExtendedIsolationForest(isotree)", model
    except Exception:
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
        model.fit(X_train)
        # Use negative decision function so higher => more anomalous
        s_train = -model.decision_function(X_train)
        s_eval = -model.decision_function(X_eval)
        return np.asarray(s_train), np.asarray(s_eval), "IsolationForest(sklearn)", model


def run_copod(X_train: np.ndarray, X_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str, object]:
    from pyod.models.copod import COPOD
    m = COPOD()
    m.fit(X_train)
    s_train = m.decision_function(X_train)
    s_eval = m.decision_function(X_eval)
    return np.asarray(s_train), np.asarray(s_eval), "COPOD(pyod)", m


def run_rrcf(X_train: np.ndarray, X_eval: np.ndarray, n_trees: int = 50, shingle_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, str, object]:
    """
    Batch scoring with rrcf:
    - Build forest on X_train
    - Score train as their average CoDisp
    - Score eval by temporary insertion/removal (costly but ok for moderate sizes)
    Note: Higher CoDisp => more anomalous.
    """
    try:
        import rrcf
    except Exception as e:
        raise RuntimeError("Please install 'rrcf' (pip install rrcf)") from e

    # Normalize features for tree sensitivity
    X_train = np.asarray(X_train, dtype=float)
    X_eval = np.asarray(X_eval, dtype=float)
    # Simple robust scaling
    med = np.median(X_train, axis=0)
    mad = np.median(np.abs(X_train - med), axis=0) + 1e-9
    Xtr = (X_train - med) / mad
    Xev = (X_eval - med) / mad

    sample_size = min(256, len(Xtr)) if len(Xtr) > 0 else 64
    forest = []
    rng = np.random.default_rng(42)
    for _ in range(n_trees):
        if len(Xtr) <= sample_size:
            subsample = Xtr
        else:
            idx = rng.choice(len(Xtr), size=sample_size, replace=False)
            subsample = Xtr[idx]
        tree = rrcf.RCTree()
        for i, x in enumerate(subsample):
            tree.insert_point(x, index=i)
        forest.append(tree)

    # Score train by inserting if not in tree; approximate by sampling subset
    def codisp_point(x: np.ndarray) -> float:
        s = 0.0
        for t in forest:
            # temporary insert
            idx = -1  # dummy
            t.insert_point(x, index=idx)
            s += t.codisp(idx)
            t.forget_point(idx)
        return s / max(1, len(forest))

    # Train scores (approximate on full set)
    s_train = np.array([codisp_point(x) for x in Xtr], dtype=float)
    s_eval = np.array([codisp_point(x) for x in Xev], dtype=float)
    model_state = {
        "type": "rrcf",
        "forest": forest,
        "median": med,
        "mad": mad,
        "sample_size": sample_size,
        "n_trees": n_trees,
        "rng_seed": 42,
    }
    return s_train, s_eval, "RRCF(rrcf)", model_state


def _svg_polyline(xs: List[float], ys: List[float], w: int = 360, h: int = 240, pad: int = 20, stroke: str = "#2563eb") -> str:
    def sx(x: float) -> float: return pad + float(x) * (w - 2*pad)
    def sy(y: float) -> float: return h - pad - float(y) * (h - 2*pad)
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    axes = f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#999" stroke-width="1"/>' \
           f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#999" stroke-width="1"/>'
    poly = f'<polyline fill="none" stroke="{stroke}" stroke-width="2" points="{pts}"/>'
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">{axes}{poly}</svg>'


def write_report(out_dir: Path, results: List[DetectorResult], per_flag_map: Optional[Dict[str, List[Dict[str, object]]]] = None, threshold_quantile: Optional[float] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    validation: Optional[dict] = None
    try:
        val_path = out_dir / "dataset_validation.json"
        if val_path.exists():
            validation = json.loads(val_path.read_text())
    except Exception:
        validation = None
    data = {
        "generated_at": datetime.now().isoformat(),
        "dataset_validation": validation,
        "threshold_quantile": threshold_quantile,
        "results": [
            {
                "detector": r.name,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "pr_auc": r.pr_auc,
                "roc_auc": r.roc_auc,
                "threshold": r.threshold,
                "pr_curve": {"recall": r.pr_curve[0], "precision": r.pr_curve[1]} if r.pr_curve else None,
                "roc_curve": {"fpr": r.roc_curve[0], "tpr": r.roc_curve[1]} if r.roc_curve else None,
                "by_flag": (per_flag_map or {}).get(r.name),
            }
            for r in results
        ],
    }
    (out_dir / "ad_compare.json").write_text(json.dumps(data, indent=2))
    # Simple HTML
    rows = []
    for r in results:
        rows.append(
            f"<tr><td>{r.name}</td>"
            f"<td>{r.precision:.4f}</td>"
            f"<td>{r.recall:.4f}</td>"
            f"<td>{r.f1:.4f}</td>"
            f"<td>{'n/a' if r.pr_auc is None else f'{r.pr_auc:.4f}'}</td>"
            f"<td>{'n/a' if r.roc_auc is None else f'{r.roc_auc:.4f}'}</td>"
            f"<td>{r.threshold:.6f}</td></tr>"
        )
    sections = []
    for r in results:
        pr_svg = ""
        roc_svg = ""
        if r.pr_curve:
            xs = [max(0.0, min(1.0, float(x))) for x in r.pr_curve[0]]
            ys = [max(0.0, min(1.0, float(y))) for y in r.pr_curve[1]]
            pr_svg = _svg_polyline(xs, ys, stroke="#7c3aed")
        if r.roc_curve:
            xs = [max(0.0, min(1.0, float(x))) for x in r.roc_curve[0]]
            ys = [max(0.0, min(1.0, float(y))) for y in r.roc_curve[1]]
            roc_svg = _svg_polyline(xs, ys, stroke="#16a34a")
        # Per-flag table if provided
        pf_rows_html = ""
        if per_flag_map and r.name in per_flag_map and per_flag_map[r.name]:
            pf_rows = []
            for it in per_flag_map[r.name]:
                try:
                    pf_rows.append(
                        f"<tr><td>{str(it.get('flag',''))}</td>"
                        f"<td>{it.get('support_pos','')}</td>"
                        f"<td>{it.get('precision',0):.4f}</td>"
                        f"<td>{it.get('recall',0):.4f}</td>"
                        f"<td>{it.get('f1',0):.4f}</td></tr>"
                    )
                except Exception:
                    continue
            if pf_rows:
                pf_rows_html = f"""
        <div style="margin:6px 0 4px 0">Per-flag metrics (injected faults only):</div>
        <table>
          <tr><th>flag</th><th>injected faults</th><th>P</th><th>R</th><th>F1</th></tr>
          {"".join(pf_rows)}
        </table>"""
        sections.append(f"""
        <h3>{r.name}</h3>
        <div>precision={r.precision:.4f} &nbsp; recall={r.recall:.4f} &nbsp; f1={r.f1:.4f} &nbsp;
             PR-AUC={'n/a' if r.pr_auc is None else f'{r.pr_auc:.4f}'} &nbsp;
             ROC-AUC={'n/a' if r.roc_auc is None else f'{r.roc_auc:.4f}'}
        </div>
        <div style="display:grid;grid-template-columns:repeat(2,minmax(320px,1fr));gap:16px;margin:8px 0 20px 0">
          <div><h4>Precision–Recall</h4>{(pr_svg or '<div style=\"color:#666\">n/a</div>')}</div>
          <div><h4>ROC</h4>{(roc_svg or '<div style=\"color:#666\">n/a</div>')}</div>
        </div>
        {pf_rows_html}
        """)

    # Optional validation summary (HTML)
    val_html = ""
    if validation:
        mv = validation.get("missing_files", {}) if isinstance(validation, dict) else {}
        ev = validation.get("empties", {}) if isinstance(validation, dict) else {}
        val_html = f"""
  <div style="margin:10px 0 16px 0;padding:10px;border:1px solid #e5e7eb;border-radius:6px;">
    <h2 style="margin:0 0 8px 0;font-size:16px;">Dataset validation</h2>
    <div style="display:grid;grid-template-columns:repeat(3,minmax(160px,1fr));gap:8px;">
      <div>Total samples: <b>{validation.get('total_samples','n/a')}</b></div>
      <div>Train rows: <b>{validation.get('num_train','n/a')}</b></div>
      <div>Eval rows: <b>{validation.get('num_eval','n/a')}</b></div>
      <div>Missing logs: <b>{mv.get('logs','n/a')}</b></div>
      <div>Missing traces: <b>{mv.get('traces','n/a')}</b></div>
      <div>Missing metrics: <b>{mv.get('metrics','n/a')}</b></div>
      <div>Zero-line logs: <b>{ev.get('logs_zero_lines','n/a')}</b></div>
      <div>Zero-count traces: <b>{ev.get('traces_zero_count','n/a')}</b></div>
    </div>
  </div>"""

    detectors_html = """
  <div style="margin:10px 0 16px 0;padding:10px;border:1px solid #e5e7eb;border-radius:6px;">
    <h2 style="margin:0 0 8px 0;font-size:16px;">Detectors in this report</h2>
    <ul style="margin:0 0 0 18px;">
      <li><b>IsolationForest (sklearn)</b>: tree-based isolation; higher score ⇒ more anomalous.</li>
      <li><b>Extended IsolationForest (isotree)</b>: enhanced IF (if installed); higher score ⇒ more anomalous.</li>
      <li><b>COPOD (pyod)</b>: copula-based outlier degree; higher score ⇒ more anomalous.</li>
      <li><b>RRCF (rrcf)</b>: random cut forest; CoDisp score; higher score ⇒ more anomalous.</li>
    </ul>
  </div>"""

    intro_html = """
  <div style="margin:10px 0 16px 0;padding:10px;border:1px solid #e5e7eb;border-radius:6px;">
    <h2 style="margin:0 0 8px 0;font-size:16px;">What this report shows</h2>
    <div>This compares anomaly detectors on the selected dataset. Training uses baseline windows (train* labels). Evaluation uses all other windows (baselines not marked train + faults). Thresholds are calibrated on train scores at a fixed quantile; metrics (precision/recall/F1) are at that operating point, and PR/ROC AUCs are computed from continuous scores.</div>
  </div>"""

    quantile_html = f"""
  <div style="margin:10px 0 16px 0;padding:10px;border:1px solid #e5e7eb;border-radius:6px;">
    <h2 style="margin:0 0 8px 0;font-size:16px;">Threshold quantile</h2>
    <div><b>THRESHOLD_QUANTILE</b> = <b>{'n/a' if threshold_quantile is None else f'{threshold_quantile:.4f}'}</b>. We set the anomaly threshold so that this fraction of baseline (train) scores fall below it. Lowering it (e.g., 0.980 → 0.970) typically increases recall at the cost of precision.</div>
  </div>"""

    metrics_html = """
  <div style="margin:10px 0 16px 0;padding:10px;border:1px solid #e5e7eb;border-radius:6px;">
    <h2 style="margin:0 0 8px 0;font-size:16px;">Metric definitions</h2>
    <ul style="margin:0 0 0 18px;">
      <li><b>Precision</b>: TP / (TP + FP) — of predicted anomalies, how many are true faults.</li>
      <li><b>Recall</b>: TP / (TP + FN) — of true faults, how many were detected.</li>
      <li><b>F1</b>: harmonic mean of precision and recall — balances both.</li>
      <li><b>PR-AUC</b>: area under Precision–Recall curve — ranking quality for the positive class.</li>
      <li><b>ROC-AUC</b>: area under ROC curve — overall separability between classes.</li>
    </ul>
  </div>"""

    # Combined vs standalone (F1) table
    def _comp_table(rs: List[DetectorResult]) -> str:
        fams = {
            "IsolationForest": {"combined": "IsolationForest(sklearn)", "logs": "IsolationForest(logs)", "traces": "IsolationForest(traces)", "metrics": "IsolationForest(metrics)"},
            "COPOD": {"combined": "COPOD(pyod)", "logs": "COPOD(logs)", "traces": "COPOD(traces)", "metrics": "COPOD(metrics)"},
            "RRCF": {"combined": "RRCF(rrcf)", "logs": "RRCF(logs)", "traces": "RRCF(traces)", "metrics": "RRCF(metrics)"},
        }
        f1map = {r.name: r.f1 for r in rs}
        head = "<tr><th>Algorithm</th><th>Combined F1</th><th>Logs F1</th><th>Traces F1</th><th>Metrics F1</th></tr>"
        rows_h = []
        for fam, names in fams.items():
            rows_h.append(
                f"<tr><td>{fam}</td>"
                f"<td>{f1map.get(names['combined'], float('nan')):.4f}</td>"
                f"<td>{f1map.get(names['logs'], float('nan')):.4f}</td>"
                f"<td>{f1map.get(names['traces'], float('nan')):.4f}</td>"
                f"<td>{f1map.get(names['metrics'], float('nan')):.4f}</td></tr>"
            )
        return f"""
  <div style="margin:10px 0 16px 0;padding:10px;border:1px solid #e5e7eb;border-radius:6px;">
    <h2 style="margin:0 0 8px 0;font-size:16px;">Combined vs standalone (F1)</h2>
    <table>
      {head}
      {''.join(rows_h)}
    </table>
  </div>"""
    comp_html = _comp_table(results)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Anomaly Detector Comparison</title>
<style>table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 10px;text-align:center}}</style>
</head><body>
  <h1>Anomaly Detector Comparison</h1>
  {intro_html}
  {quantile_html}
  {metrics_html}
  {val_html}
  {comp_html}
  {detectors_html}
  <table>
    <tr><th>detector</th><th>precision</th><th>recall</th><th>f1</th><th>PR-AUC</th><th>ROC-AUC</th><th>threshold</th></tr>
    {"".join(rows)}
  </table>
  {"".join(sections)}
</body></html>"""
    (out_dir / "ad_compare.html").write_text(html)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare anomaly detectors on existing features")
    ap.add_argument("--features", type=Path, required=True, help="Path to features.csv")
    ap.add_argument("--report", type=Path, required=False, help="Path to report_isoforest.json (to get feature_cols)")
    ap.add_argument("--out", type=Path, required=False, help="Output directory (default: aiops_harness/out/ad_compare_<ts>)")
    ap.add_argument("--quantile", type=float, default=0.995, help="Train baseline score quantile for threshold")
    args = ap.parse_args()
    # Allow env override for a global operating point
    try:
        env_q = os.getenv("THRESHOLD_QUANTILE")
        if env_q is not None:
            args.quantile = float(env_q)
    except Exception:
        pass

    df, feature_cols = load_data(args.features, args.report)
    if not feature_cols:
        raise SystemExit("[ERROR] could not determine feature columns")
    # Build train/eval DataFrames with noisy baseline filtering for training
    if "in_training" not in df.columns:
        df["in_training"] = False
    train_df = df[df["in_training"] == True].copy()  # noqa: E712
    # Keep only true baselines in train, and drop noisy baselines
    try:
        train_df = train_df[(train_df.get("is_baseline", True) == True)]  # noqa: E712
    except Exception:
        pass
    try:
        max_err = int(os.getenv("TRAIN_LOG_ERROR_MAX", "3"))
        if "log_error_count" in train_df.columns:
            train_df = train_df[train_df["log_error_count"].astype(int) <= max_err]
    except Exception:
        pass
    eval_df = df[df["in_training"] == False].copy()  # noqa: E712
    # Arrays for models
    X_train = train_df[feature_cols].to_numpy()
    X_eval = eval_df[feature_cols].to_numpy()
    y_eval = (eval_df["root_cause"] != "none").astype(int).to_numpy()
    # Metadata for eval split
    eval_flags: np.ndarray = eval_df.get("flag", pd.Series([], dtype=str)).astype(str).to_numpy()
    if len(X_train) < 5 or len(X_eval) < 2:
        raise SystemExit("[ERROR] insufficient data: need >=5 train and >=2 eval")

    out_dir = args.out or (Path(__file__).resolve().parents[1] / "out" / f"ad_compare_{datetime.now().strftime('%Y%m%d%H%M%S')}")

    results: List[DetectorResult] = []
    model_files: Dict[str, str] = {}

    def _safe_name(n: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", n)

    def _save_model(model_obj: object, name: str) -> Optional[Path]:
        try:
            fname = f"{_safe_name(name)}.pkl"
            fpath = out_dir / fname
            out_dir.mkdir(parents=True, exist_ok=True)
            # Try joblib first; fall back to pickle
            try:
                joblib_dump(model_obj, fpath)
            except Exception:
                with open(fpath, "wb") as fh:
                    pkl.dump(model_obj, fh)
            return fpath
        except Exception:
            return None

    # Extended IF (or fallback IF)
    s_tr, s_ev, name, mdl = run_iforest_ext(X_train, X_eval)
    thr = calibrate_threshold(s_tr, args.quantile)
    p, r, f1, pr_auc, roc_auc, pr_pts, roc_pts = compute_metrics(y_eval, s_ev, thr, higher_is_anomalous=True)
    results.append(DetectorResult(name, p, r, f1, pr_auc, roc_auc, thr, pr_pts, roc_pts))
    mp = _save_model(mdl, name)
    if mp:
        model_files[name] = str(mp)
    # Modality-specific IsolationForest models (logs-only, traces-only, metrics-only)
    def _run_if_for_cols(mod_name: str, cols: List[str]) -> None:
        if not cols:
            return
        try:
            X_tr_m = train_df[cols].to_numpy()
            X_ev_m = eval_df[cols].to_numpy()
            if X_tr_m.shape[1] == 0 or X_ev_m.shape[1] == 0:
                return
            s_tr_m, s_ev_m, _, mdl_m = run_iforest_ext(X_tr_m, X_ev_m)
            thr_m = calibrate_threshold(s_tr_m, args.quantile)
            p_m, r_m, f1_m, pr_auc_m, roc_auc_m, pr_pts_m, roc_pts_m = compute_metrics(y_eval, s_ev_m, thr_m, higher_is_anomalous=True)
            det_name = f"IsolationForest({mod_name})"
            results.append(DetectorResult(det_name, p_m, r_m, f1_m, pr_auc_m, roc_auc_m, thr_m, pr_pts_m, roc_pts_m))
            mp_m = _save_model(mdl_m, det_name)
            if mp_m:
                model_files[det_name] = str(mp_m)
        except Exception:
            pass
    try:
        mods = split_modalities(feature_cols)
        _run_if_for_cols("logs", mods.get("logs", []))
        _run_if_for_cols("traces", mods.get("traces", []))
        _run_if_for_cols("metrics", mods.get("metrics", []))
    except Exception:
        pass
    # Modality-specific IsolationForest models (logs-only, traces-only, metrics-only)
    def _run_if_for_cols(mod_name: str, cols: List[str]) -> None:
        if not cols:
            return
        try:
            X_tr_m = train_df[cols].to_numpy()
            X_ev_m = eval_df[cols].to_numpy()
            if X_tr_m.shape[1] == 0 or X_ev_m.shape[1] == 0:
                return
            s_tr_m, s_ev_m, _, mdl_m = run_iforest_ext(X_tr_m, X_ev_m)
            thr_m = calibrate_threshold(s_tr_m, args.quantile)
            p_m, r_m, f1_m, pr_auc_m, roc_auc_m, pr_pts_m, roc_pts_m = compute_metrics(y_eval, s_ev_m, thr_m, higher_is_anomalous=True)
            det_name = f"IsolationForest({mod_name})"
            results.append(DetectorResult(det_name, p_m, r_m, f1_m, pr_auc_m, roc_auc_m, thr_m, pr_pts_m, roc_pts_m))
            mp_m = _save_model(mdl_m, det_name)
            if mp_m:
                model_files[det_name] = str(mp_m)
        except Exception:
            pass
    try:
        mods = split_modalities(feature_cols)
        _run_if_for_cols("logs", mods.get("logs", []))
        _run_if_for_cols("traces", mods.get("traces", []))
        _run_if_for_cols("metrics", mods.get("metrics", []))
    except Exception:
        pass
    # Per-flag breakdown at the operating point
    def per_flag_breakdown(flags_arr: np.ndarray, y_true_arr: np.ndarray, preds_arr: np.ndarray) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        try:
            uniq = pd.Series(flags_arr).fillna("").astype(str).unique()
        except Exception:
            uniq = np.unique(flags_arr)
        for fflag in uniq:
            mask = flags_arr == fflag
            if mask.sum() == 0:
                continue
            yt = y_true_arr[mask]
            pr = preds_arr[mask]
            tp = int(((yt == 1) & (pr == 1)).sum())
            fp = int(((yt == 0) & (pr == 1)).sum())
            fn = int(((yt == 1) & (pr == 0)).sum())
            precision_f = tp / (tp + fp) if (tp + fp) else 0.0
            recall_f = tp / (tp + fn) if (tp + fn) else 0.0
            f1_f = (2 * precision_f * recall_f / (precision_f + recall_f)) if (precision_f + recall_f) else 0.0
            support_pos = int((yt == 1).sum())
            out.append({"flag": str(fflag), "precision": precision_f, "recall": recall_f, "f1": f1_f, "support_pos": support_pos})
        # Sort by flag name for stability
        try:
            out.sort(key=lambda x: x["flag"])
        except Exception:
            pass
        return out
    preds_if = (s_ev > thr).astype(int)
    by_flag_if = per_flag_breakdown(eval_flags, y_eval, preds_if)

    # COPOD
    try:
        s_tr, s_ev, name, mdl = run_copod(X_train, X_eval)
        thr = calibrate_threshold(s_tr, args.quantile)
        p, r, f1, pr_auc, roc_auc, pr_pts, roc_pts = compute_metrics(y_eval, s_ev, thr, higher_is_anomalous=True)
        results.append(DetectorResult(name, p, r, f1, pr_auc, roc_auc, thr, pr_pts, roc_pts))
        mp = _save_model(mdl, name)
        if mp:
            model_files[name] = str(mp)
        preds_copod = (s_ev > thr).astype(int)
        by_flag_copod = per_flag_breakdown(eval_flags, y_eval, preds_copod)
    except Exception as e:
        print(f"[WARN] COPOD failed: {e}")
        by_flag_copod = []

    # COPOD per-modality
    try:
        def _run_copod_for_cols(mod_name: str, cols: List[str]) -> None:
            if not cols:
                return
            try:
                X_tr_m = train_df[cols].to_numpy()
                X_ev_m = eval_df[cols].to_numpy()
                if X_tr_m.shape[1] == 0 or X_ev_m.shape[1] == 0:
                    return
                s_tr_m, s_ev_m, _, mdl_m = run_copod(X_tr_m, X_ev_m)
                thr_m = calibrate_threshold(s_tr_m, args.quantile)
                p_m, r_m, f1_m, pr_auc_m, roc_auc_m, pr_pts_m, roc_pts_m = compute_metrics(y_eval, s_ev_m, thr_m, higher_is_anomalous=True)
                det_name = f"COPOD({mod_name})"
                results.append(DetectorResult(det_name, p_m, r_m, f1_m, pr_auc_m, roc_auc_m, thr_m, pr_pts_m, roc_pts_m))
                mp_m = _save_model(mdl_m, det_name)
                if mp_m:
                    model_files[det_name] = str(mp_m)
            except Exception:
                pass
        try:
            mods = split_modalities(feature_cols)
            _run_copod_for_cols("logs", mods.get("logs", []))
            _run_copod_for_cols("traces", mods.get("traces", []))
            _run_copod_for_cols("metrics", mods.get("metrics", []))
        except Exception:
            pass
    except Exception:
        pass

    # RRCF
    try:
        s_tr, s_ev, name, mdl = run_rrcf(X_train, X_eval)
        thr = calibrate_threshold(s_tr, args.quantile)
        p, r, f1, pr_auc, roc_auc, pr_pts, roc_pts = compute_metrics(y_eval, s_ev, thr, higher_is_anomalous=True)
        results.append(DetectorResult(name, p, r, f1, pr_auc, roc_auc, thr, pr_pts, roc_pts))
        mp = _save_model(mdl, name)
        if mp:
            model_files[name] = str(mp)
        preds_rrcf = (s_ev > thr).astype(int)
        by_flag_rrcf = per_flag_breakdown(eval_flags, y_eval, preds_rrcf)
        # Modality-specific RRCF
        def _run_rrcf_for_cols(mod_name: str, cols: List[str]) -> None:
            if not cols:
                return
            try:
                X_tr_m = train_df[cols].to_numpy()
                X_ev_m = eval_df[cols].to_numpy()
                if X_tr_m.shape[1] == 0 or X_ev_m.shape[1] == 0:
                    return
                s_tr_m, s_ev_m, _, mdl_m = run_rrcf(X_tr_m, X_ev_m)
                thr_m = calibrate_threshold(s_tr_m, args.quantile)
                p_m, r_m, f1_m, pr_auc_m, roc_auc_m, pr_pts_m, roc_pts_m = compute_metrics(y_eval, s_ev_m, thr_m, higher_is_anomalous=True)
                det_name = f"RRCF({mod_name})"
                results.append(DetectorResult(det_name, p_m, r_m, f1_m, pr_auc_m, roc_auc_m, thr_m, pr_pts_m, roc_pts_m))
                mp_m = _save_model(mdl_m, det_name)
                if mp_m:
                    model_files[det_name] = str(mp_m)
            except Exception:
                pass
        try:
            _run_rrcf_for_cols("logs", mods.get("logs", []))
            _run_rrcf_for_cols("traces", mods.get("traces", []))
            _run_rrcf_for_cols("metrics", mods.get("metrics", []))
        except Exception:
            pass
    except Exception as e:
        print(f"[WARN] RRCF unavailable or failed: {e}")
        by_flag_rrcf = []

    # Extend JSON with per-flag breakdown alongside results
    # Build map detector->by_flag
    per_flag_map: Dict[str, List[Dict[str, object]]] = {}
    try:
        per_flag_map["IsolationForest(sklearn)"] = by_flag_if
    except Exception:
        pass
    try:
        per_flag_map["COPOD(pyod)"] = by_flag_copod
    except Exception:
        pass
    try:
        per_flag_map["RRCF(rrcf)"] = by_flag_rrcf
    except Exception:
        pass
    write_report(out_dir, results, per_flag_map=per_flag_map, threshold_quantile=float(args.quantile))
    # Also record model file locations
    try:
        meta = {"model_files": model_files}
        (out_dir / "models_index.json").write_text(json.dumps(meta, indent=2))
    except Exception:
        pass
    print("Wrote:")
    print(f" - {out_dir/'ad_compare.json'}")
    print(f" - {out_dir/'ad_compare.html'}")
    if model_files:
        print(f" - models: {len(model_files)} saved (index: {out_dir/'models_index.json'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


