"""
Default MLflow tracking URI for OpenShift (matches config/harness.yaml).

Override with MLFLOW_TRACKING_URI or MLFLOW_TRACKING_URL (alias).
"""
from __future__ import annotations

import os

# Same default as config/harness.yaml mlflow.tracking_uri
DEFAULT_MLFLOW_TRACKING_URI = "https://mlflow-agentic-aiops.apps.sno1gpu.localdomain"


def mlflow_tracking_uri() -> str:
    """OpenShift MLflow by default; env MLFLOW_TRACKING_URI or MLFLOW_TRACKING_URL overrides."""
    for key in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_URL"):
        v = (os.environ.get(key) or "").strip()
        if v:
            return v
    return DEFAULT_MLFLOW_TRACKING_URI
