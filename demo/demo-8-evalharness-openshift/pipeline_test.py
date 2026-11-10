"""RHOAI Data Science Pipeline for the AIOps evaluation harness."""
from kfp import dsl
from kfp.compiler import Compiler

HARNESS_IMAGE = "image-registry.openshift-image-registry.svc:5000/agentic-aiops-harness/aiops-harness:latest"
AGENTS = ["rhdp_deepseek-r1-distill-qwen-14b", "rhdp_granite-3-2-8b-instruct"]
FAULTS = ["scale_zero", "kill_pod"]


@dsl.component(base_image=HARNESS_IMAGE)
def harness_cell(agent: str, fault: str) -> dict:
    import base64, json, os, subprocess, sys

    # Load MaaS credentials from the Kubernetes Secret
    from kubernetes import client, config
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    ns = open("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read().strip()
    secret = v1.read_namespaced_secret("aiops-model-credentials", ns)
    for key, value in (secret.data or {}).items():
        os.environ[key] = base64.b64decode(value).decode()

    # Configure harness connections
    os.environ["CLICKHOUSE_HTTP"] = "http://clickhouse.agentic-aiops-harness.svc.cluster.local:8123"
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow.agentic-aiops-harness.svc.cluster.local:5000"
    os.environ["K8S_FAULT_NAMESPACE"] = "otel-demo"
    os.environ["K8S_NAMESPACE"] = "otel-demo"
    os.environ["PYTHONPATH"] = "/app"
    os.environ["AGENT_TIMEOUT"] = "600"
    os.environ["DETECTION_WINDOW"] = "700"

    cmd = [
        sys.executable, "-u", "/app/code/harness/run_harness.py",
        "--scenario", "a",
        "--flag", fault,
        "--variant", "cart",
        "--context-c", "0",
        "--agents", agent,
        "--no-judge",
        "--detection-timeout", "600",
        "--fault-duration", "120",
        "--output", "/tmp/result.json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd="/app")
    print(proc.stdout[-3000:])
    if proc.returncode != 0:
        print(proc.stderr[-3000:], file=sys.stderr)
    try:
        with open("/tmp/result.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "no output file", "detected": False, "mttd_seconds": None}


@dsl.pipeline(name="aiops-eval-test", description="AIOps harness: MaaS agents × faults")
def aiops_eval_test():
    for agent in AGENTS:
        for fault in FAULTS:
            cell = harness_cell(agent=agent, fault=fault)
            cell.set_display_name(f"{agent.split('-')[0][:10]} / {fault}")


if __name__ == "__main__":
    Compiler().compile(aiops_eval_test, "pipeline_test.yaml")
    print("Compiled")
