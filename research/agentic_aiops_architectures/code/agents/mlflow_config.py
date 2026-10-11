"""
Default MLflow tracking URI: local server on :5050 (see scripts/ensure_local_mlflow.sh).

Override with MLFLOW_TRACKING_URI or MLFLOW_TRACKING_URL (alias).
Do not point at the OpenShift agentic-aiops/mlflow service for harness runs.
"""
from __future__ import annotations

import os

DEFAULT_MLFLOW_TRACKING_URI = "http://localhost:5050"


def mlflow_tracking_uri() -> str:
    """MLflow URI from MLFLOW_TRACKING_URI / MLFLOW_TRACKING_URL, else local default."""
    for key in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_URL"):
        v = (os.environ.get(key) or "").strip()
        if v:
            return v
    return DEFAULT_MLFLOW_TRACKING_URI
