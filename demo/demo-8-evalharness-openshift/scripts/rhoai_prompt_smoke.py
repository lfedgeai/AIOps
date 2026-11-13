"""Test prompt registration and snapshots in a pipeline without injecting faults."""
from kfp import dsl
from kfp.compiler import Compiler


@dsl.component(base_image="image-registry.openshift-image-registry.svc:5000/agentic-aiops-harness/aiops-harness:latest")
def verify_prompts() -> str:
    import json
    import os
    import sys

    import mlflow
    from mlflow.tracking import MlflowClient

    sys.path.insert(0, "/app")
    from code.agents.mlflow_agent_logging import log_agent_trace_to_mlflow
    from code.agents.mlflow_prompt_registry import _cache

    sa_path = "/var/run/secrets/kubernetes.io/serviceaccount"
    with open(f"{sa_path}/namespace") as f:
        workspace = f.read().strip()
    with open(f"{sa_path}/token") as f:
        os.environ["MLFLOW_TRACKING_TOKEN"] = f.read().strip()
    os.environ["MLFLOW_WORKSPACE"] = workspace
    os.environ["REQUESTS_CA_BUNDLE"] = f"{sa_path}/service-ca.crt"
    uri = "https://mlflow.redhat-ods-applications.svc:8443/mlflow"
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_workspace(workspace)
    mlflow.set_experiment("aiops-pipeline-connectivity")
    prompts = {"openai_model": "synthetic-prompt-smoke", "system_prompt": "x" * 6100,
               "user_prompt": "synthetic test; no faults", "context_c": "0"}
    with mlflow.start_run(run_name="pipeline-prompt-smoke") as run:
        run_id = run.info.run_id
        for _ in range(2):
            log_agent_trace_to_mlflow(run_id=run_id, tracking_uri=uri, prompts=prompts,
                                      tool_calls=[], thinking=None, llm_rounds=[])
            _cache.clear()  # The next call must find versions as a new agent process would.
        client = MlflowClient()
        run_data = client.get_run(run_id).data
        if run_data.tags.get("aiops.prompt_registry_status") != "ok":
            raise RuntimeError("Prompt Registry did not report success")
        invocations = client.list_artifacts(run_id, "agent_trace")
        if len(invocations) != 1:
            raise RuntimeError("Expected one agent invocation")
        snapshots = client.list_artifacts(run_id, invocations[0].path)
        if len(snapshots) != 2:
            raise RuntimeError("Expected two distinct snapshots")
        manifests = []
        for snapshot in snapshots:
            filename = client.download_artifacts(run_id, f"{snapshot.path}/prompt_registry.json")
            with open(filename) as f:
                manifests.append(json.load(f))
        if manifests[0] != manifests[1]:
            raise RuntimeError("Identical prompts did not reuse versions")
        for role, entry in manifests[0]["roles"].items():
            text = "".join(client.get_prompt_version(p["name"], p["version"]).template for p in entry["parts"])
            if text != prompts[f"{role}_prompt"]:
                raise RuntimeError(f"{role} prompt reconstruction failed")
    print(f"Verified prompt versions and snapshots in run {run_id}")
    return run_id


@dsl.pipeline(name="aiops-rhoai-prompt-smoke")
def prompt_smoke_pipeline():
    verify_prompts().set_caching_options(False)


if __name__ == "__main__":
    Compiler().compile(prompt_smoke_pipeline, "/tmp/opencode/aiops-rhoai-prompt-smoke.yaml")
