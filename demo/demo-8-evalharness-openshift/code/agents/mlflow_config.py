"""
Default MLflow tracking URI: in-cluster MLflow in ``agentic-aiops``.

Override with MLFLOW_TRACKING_URI or MLFLOW_TRACKING_URL (alias). The standalone
repo deploys MLflow as a normal cluster service with a PVC, so the default is the
cluster DNS name. Set ``MLFLOW_TRACKING_URI=http://127.0.0.1:5050`` when running
the harness from a laptop against a port-forward.
"""
from __future__ import annotations

import os

DEFAULT_MLFLOW_TRACKING_URI = (
    os.environ.get("MLFLOW_TRACKING_URI")
    or os.environ.get("MLFLOW_TRACKING_URL")
    or "https://mlflow.redhat-ods-applications.svc.cluster.local:8443/mlflow"
)


def mlflow_tracking_uri() -> str:
    """MLflow URI from MLFLOW_TRACKING_URI / MLFLOW_TRACKING_URL, else local default."""
    for key in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_URL"):
        v = (os.environ.get(key) or "").strip()
        if v:
            return v
    return DEFAULT_MLFLOW_TRACKING_URI
