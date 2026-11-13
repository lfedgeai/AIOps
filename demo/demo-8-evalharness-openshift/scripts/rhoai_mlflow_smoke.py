"""Verify RHOAI MLflow access from a Data Science Pipeline step."""

from kfp import dsl
from kfp.compiler import Compiler


@dsl.component(
    base_image="image-registry.openshift-image-registry.svc:5000/agentic-aiops-harness/aiops-harness:latest"
)
def verify_mlflow() -> str:
    import json
    import os

    import mlflow
    from mlflow.tracking import MlflowClient

    sa_path = "/var/run/secrets/kubernetes.io/serviceaccount"
    with open(f"{sa_path}/namespace") as f:
        workspace = f.read().strip()
    with open(f"{sa_path}/token") as f:
        os.environ["MLFLOW_TRACKING_TOKEN"] = f.read().strip()
    os.environ["MLFLOW_WORKSPACE"] = workspace
    os.environ["REQUESTS_CA_BUNDLE"] = f"{sa_path}/service-ca.crt"
    os.environ["MLFLOW_TRACKING_URI"] = (
        "https://mlflow.redhat-ods-applications.svc:8443/mlflow"
    )
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    print(f"MLflow client {mlflow.__version__}, workspace {workspace}")
    if not hasattr(mlflow, "set_workspace"):
        raise RuntimeError("The MLflow client does not support workspaces")
    mlflow.set_workspace(workspace)
    mlflow.set_experiment("aiops-pipeline-connectivity")
    with mlflow.start_run(run_name="pipeline-mlflow-smoke") as run:
        mlflow.log_metric("pipeline_connectivity", 1)
        mlflow.log_dict({"workspace": workspace, "ok": True}, "smoke.json")
        run_id = run.info.run_id
    client = MlflowClient()
    stored = client.get_run(run_id)
    if stored.data.metrics.get("pipeline_connectivity") != 1:
        raise RuntimeError("Metric was not stored")
    artifact_path = client.download_artifacts(run_id, "smoke.json")
    with open(artifact_path) as f:
        if json.load(f) != {"workspace": workspace, "ok": True}:
            raise RuntimeError("Artifact did not match")
    print(f"Verified metric and artifact for run {run_id}")
    return run_id


@dsl.pipeline(name="aiops-rhoai-mlflow-smoke")
def mlflow_smoke_pipeline():
    verify_mlflow().set_caching_options(False)


if __name__ == "__main__":
    Compiler().compile(mlflow_smoke_pipeline, "rhoai_mlflow_smoke.yaml")
