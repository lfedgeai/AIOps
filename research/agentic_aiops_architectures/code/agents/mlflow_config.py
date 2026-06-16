"""
Default MLflow tracking URI for OpenShift (matches config/harness.yaml).

Override with MLFLOW_TRACKING_URI or MLFLOW_TRACKING_URL (alias).
"""
from __future__ import annotations

import os

# No cluster-specific default — set MLFLOW_TRACKING_URI or use run_harness.sh route discovery.
DEFAULT_MLFLOW_TRACKING_URI = ""


def mlflow_tracking_uri() -> str:
    """MLflow URI from MLFLOW_TRACKING_URI / MLFLOW_TRACKING_URL, else empty."""
    for key in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_URL"):
        v = (os.environ.get(key) or "").strip()
        if v:
            return v
    return DEFAULT_MLFLOW_TRACKING_URI
