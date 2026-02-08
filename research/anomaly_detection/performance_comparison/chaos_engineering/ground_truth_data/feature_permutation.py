#!/usr/bin/env python3
"""
Compute permutation feature importances for the trained IsolationForest using the current
features.csv and evaluation split. Writes a timestamped JSON report so previous reports are preserved.
"""
from __future__ import annotations

import json
import os
import glob
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import average_precision_score

BASE_DIR = Path("/home/redhat/git/OTEL/otel-demo/otel_ground_truth_data")
FEATURES_CSV = BASE_DIR / "features.csv"
MAIN_REPORT_JSON = BASE_DIR / "report_isoforest.json"

# Match training tolerance used in feature_pipeline.py
TRAIN_LOG_ERROR_MAX = int(os.getenv("TRAIN_LOG_ERROR_MAX", "3"))
# Number of shuffles per feature for stability
PERM_N = int(os.getenv("PERM_REPEATS", "5"))
RNG_SEED = int(os.getenv("PERM_SEED", "42"))


def load_latest_model_path() -> Path | None:
    candidates = sorted(
        glob.glob(str(BASE_DIR / "IsolationForestModel_*.pkl"))
    )
    if not candidates:
        return None
    return Path(candidates[-1])


def build_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    # Ensure columns exist
    if "in_training" not in df.columns:
        df["in_training"] = False
    for col in ("is_baseline", "log_error_count"):
        if col not in df.columns:
            df[col] = 0 if col == "log_error_count" else False

    # Training: baselines with low/no log errors
    train_df = df[
        (df["in_training"] == True)  # noqa: E712
        & (df["is_baseline"] == True)  # noqa: E712
        & (df["log_error_count"].astype(int) <= TRAIN_LOG_ERROR_MAX)
    ].copy()

    # Evaluation: not used for training; exclude contaminated baselines
    eval_df = df[(df["in_training"] == False)].copy()  # noqa: E712
    eval_df = eval_df[
        ~((eval_df["is_baseline"] == True) & (eval_df["log_error_count"].astype(int) > 0))  # noqa: E712
    ].copy()

    # Feature columns: prefer those from main report; fallback to numeric columns
    feature_cols: List[str] = []
    try:
        with open(MAIN_REPORT_JSON) as f:
            rep = json.load(f)
            if isinstance(rep.get("feature_cols"), list):
                feature_cols = [str(c) for c in rep["feature_cols"]]
    except Exception:
        pass
    if not feature_cols:
        # Heuristic: drop non-feature cols
        exclude = {
            "id", "label", "flag", "variant", "root_cause",
            "is_baseline", "in_training",
        }
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return train_df, eval_df, feature_cols


def compute_scores(model, X_eval: pd.DataFrame) -> np.ndarray:
    # Higher anomaly_score => more anomalous
    return -model.decision_function(X_eval)


def main() -> int:
    if not FEATURES_CSV.exists():
        print(f"[ERROR] features.csv not found at {FEATURES_CSV}")
        return 1
    df = pd.read_csv(FEATURES_CSV)

    train_df, eval_df, feature_cols = build_splits(df)
    if len(eval_df) < 2:
        print(f"[ERROR] insufficient eval data for permutation: eval={len(eval_df)}")
        return 1

    model_path = load_latest_model_path()
    if not model_path or not model_path.exists():
        print("[ERROR] trained IsolationForest model not found (IsolationForestModel_*.pkl)")
        return 1
    model = load(model_path)

    # Ground truth for eval set
    y_true = (eval_df["root_cause"] != "none").astype(int).to_numpy()
    if len(np.unique(y_true)) < 2:
        print("[ERROR] eval set needs both classes for PR-AUC-based permutation importance")
        return 1

    X_eval = eval_df[feature_cols].copy()
    baseline_scores = compute_scores(model, X_eval)
    try:
        baseline_pr_auc = float(average_precision_score(y_true, baseline_scores))
    except Exception:
        print("[ERROR] failed to compute baseline PR-AUC")
        return 1

    rng = np.random.default_rng(RNG_SEED)
    drops: Dict[str, List[float]] = {c: [] for c in feature_cols}

    for col in feature_cols:
        for _ in range(PERM_N):
            X_shuf = X_eval.copy()
            # Shuffle one column independently
            X_shuf[col] = rng.permutation(X_shuf[col].to_numpy())
            s = compute_scores(model, X_shuf)
            try:
                pr_auc = float(average_precision_score(y_true, s))
            except Exception:
                pr_auc = math.nan
            if not math.isnan(pr_auc):
                drops[col].append(baseline_pr_auc - pr_auc)

    # Aggregate
    agg: Dict[str, Dict[str, float]] = {}
    for col, arr in drops.items():
        if arr:
            mu = float(np.mean(arr))
            sd = float(np.std(arr))
            agg[col] = {"mean_drop_pr_auc": mu, "std_drop_pr_auc": sd, "n": len(arr)}
        else:
            agg[col] = {"mean_drop_pr_auc": 0.0, "std_drop_pr_auc": 0.0, "n": 0}

    ranked = sorted(agg.items(), key=lambda kv: kv[1]["mean_drop_pr_auc"], reverse=True)

    out = {
        "ok": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_path": str(model_path),
        "feature_cols": feature_cols,
        "perm_repeats": PERM_N,
        "baseline_pr_auc": baseline_pr_auc,
        "importances_perm": agg,
        "importances_perm_ranked": ranked,
        "train_count": int(len(train_df)),
        "eval_count": int(len(eval_df)),
    }
    out_path = BASE_DIR / f"report_isoforest_perm_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote permutation importance report → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


