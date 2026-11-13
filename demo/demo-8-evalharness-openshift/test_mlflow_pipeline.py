from kfp import dsl
from kfp.compiler import Compiler

@dsl.component(base_image="image-registry.openshift-image-registry.svc:5000/agentic-aiops-harness/aiops-harness:latest")
def test_rhoai_mlflow() -> str:
    import json, os
    mlflow_vars = {k: v for k, v in os.environ.items() if "MLFLOW" in k.upper()}
    print(f"MLflow env vars: {json.dumps(mlflow_vars, indent=2)}")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    print(f"Tracking URI: {tracking_uri}")
    if not tracking_uri:
        return "NO_TRACKING_URI_INJECTED"
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        exps = client.search_experiments()
        return f"SUCCESS: connected, found {len(exps)} experiments"
    except Exception as e:
        return f"FAILED: {type(e).__name__}: {e}"

@dsl.pipeline(name="test-mlflow-access")
def test_pipeline():
    test_rhoai_mlflow()

if __name__ == "__main__":
    Compiler().compile(test_pipeline, "test_mlflow_pipeline.yaml")
    print("Compiled")
