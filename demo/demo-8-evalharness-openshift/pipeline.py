"""RHOAI Data Science Pipeline for the AIOps evaluation harness.

Sequential execution: reset → cell → reset → cell → ... → report.
Each harness cell runs one agent × one fault. A reset step between cells
ensures the OTEL demo returns to a clean baseline. A final report step
aggregates all results, logs MLflow metrics, and outputs a summary artifact.

Pipeline file: pipeline.py (this file)
Compiled YAML: pipeline.yaml (uploaded to KFP)
"""
import json

from kfp import dsl
from kfp.compiler import Compiler

HARNESS_IMAGE = "image-registry.openshift-image-registry.svc:5000/agentic-aiops-harness/aiops-harness:latest"

AGENTS = ["rhdp_deepseek-r1-distill-qwen-14b", "rhdp_granite-3-2-8b-instruct"]
FAULTS = ["scale_zero", "kill_pod"]


@dsl.component(base_image=HARNESS_IMAGE)
def preflight_models():
    """Require a real response from both models before any fault is injected."""
    import base64
    import json
    import urllib.request

    from kubernetes import client, config

    config.load_incluster_config()
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
        namespace = f.read().strip()
    secret = client.CoreV1Api().read_namespaced_secret("aiops-model-credentials", namespace)
    env = {k: base64.b64decode(v).decode() for k, v in (secret.data or {}).items()}
    endpoints = [
        ("DEEPSEEK", "deepseek-r1-distill-qwen-14b", 1024),
        ("GRANITE", "granite-3-2-8b-instruct", 64),
    ]
    for name, default_model, max_tokens in endpoints:
        base, key = env.get(f"{name}_API_BASE"), env.get(f"{name}_API_KEY")
        if not base or not key:
            raise RuntimeError(f"Missing endpoint or credentials for {name}")
        payload = json.dumps({
            "model": env.get(f"{name}_MODEL") or default_model,
            "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            base.rstrip("/") + "/chat/completions", data=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            choices = json.load(response).get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        if not str(message.get("content") or "").strip() and not message.get("tool_calls"):
            raise RuntimeError(f"{name} returned an empty completion")
        print(f"Model preflight passed: {name}")


@dsl.component(base_image=HARNESS_IMAGE)
def reset_environment() -> str:
    """Reset the OTEL demo to a clean baseline before the next harness cell.

    Scales all fault-target deployments back to 1 replica, removes any
    leftover harness artifacts (network policies, env vars), and waits
    for pods to become Ready.
    """
    import os
    import time

    from kubernetes import client, config

    config.load_incluster_config()
    apps = client.AppsV1Api()
    core = client.CoreV1Api()
    net = client.NetworkingV1Api()

    namespace = "otel-demo"
    targets = ["cart", "checkout", "valkey-cart", "load-generator",
               "product-catalog", "payment", "email", "frontend"]
    issues = []

    # Only restore the deployment that this pipeline injects faults into.
    for dep_name in targets:
        try:
            dep = apps.read_namespaced_deployment(dep_name, namespace)
            if dep_name == "cart" and (dep.spec.replicas or 0) < 1:
                apps.patch_namespaced_deployment(dep_name, namespace, {"spec": {"replicas": 1}})
                print(f"  scaled {dep_name} back to 1")
        except Exception as e:
            issues.append(f"{dep_name}: {e}")

    # Remove leftover harness network policies
    try:
        policies = net.list_namespaced_network_policy(namespace)
        for pol in policies.items:
            if pol.metadata.name == "harness-block-cart":
                net.delete_namespaced_network_policy(pol.metadata.name, namespace)
                print(f"  deleted network policy {pol.metadata.name}")
    except Exception as e:
        issues.append(f"network policies: {e}")

    # Wait for pods to become Ready
    deadline = time.time() + 120
    while time.time() < deadline:
        all_ready = True
        for dep_name in targets:
            try:
                dep = apps.read_namespaced_deployment(dep_name, namespace)
                ready = dep.status.ready_replicas or 0
                if ready < max(1, dep.spec.replicas or 0) or (dep.status.available_replicas or 0) < max(1, dep.spec.replicas or 0):
                    all_ready = False
            except Exception:
                all_ready = False
        if all_ready:
            break
        time.sleep(5)
    else:
        raise RuntimeError("OTEL Demo did not return to a ready baseline")

    if issues:
        raise RuntimeError(f"Reset failed: {'; '.join(issues)}")

    status = "clean" if not issues else f"warnings: {'; '.join(issues)}"
    print(f"[reset] environment ready ({status})")
    return status


@dsl.component(base_image=HARNESS_IMAGE)
def harness_cell(agent: str, fault: str) -> dict:
    """Run one harness cell: inject fault, poll agent, score, output JSON."""
    import base64
    import json
    import os
    import subprocess
    import sys

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
    import certifi
    import mlflow
    import tempfile

    sa_path = "/var/run/secrets/kubernetes.io/serviceaccount"
    os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow.redhat-ods-applications.svc:8443/mlflow"
    os.environ["MLFLOW_WORKSPACE"] = ns
    with open(f"{sa_path}/token") as token_file:
        os.environ["MLFLOW_TRACKING_TOKEN"] = token_file.read().strip()
    # Keep the public CA roots as well as the OpenShift service CA for MaaS.
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as bundle:
        for path in (certifi.where(), f"{sa_path}/service-ca.crt"):
            with open(path) as source:
                bundle.write(source.read() + "\n")
        os.environ["REQUESTS_CA_BUNDLE"] = bundle.name
    os.environ.pop("MLFLOW_TRACKING_INSECURE_TLS", None)
    os.environ.pop("MLFLOW_RUN_ID", None)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_workspace(ns)
    mlflow.set_experiment("agentic_aiops_mttd_mttr")
    os.environ["K8S_FAULT_NAMESPACE"] = "otel-demo"
    os.environ["K8S_NAMESPACE"] = "otel-demo"
    os.environ["PYTHONPATH"] = "/app"
    os.environ["AGENT_TIMEOUT"] = "600"

    # Do not inject a fault unless the selected model can answer requests.
    subprocess.run(
        [sys.executable, "/app/scripts/preflight_agent_apis.py", "--agents", agent, "--require-all"],
        check=True, cwd="/app", timeout=150,
    )

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
    print(proc.stdout[-5000:])
    if proc.returncode != 0:
        print(proc.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"Harness failed with exit code {proc.returncode}")
    if "[harness] MLflow log error:" in proc.stdout or "[harness] MLflow start_run error:" in proc.stdout:
        raise RuntimeError("Harness did not log to MLflow; see the step log")
    with open("/tmp/result.json") as f:
        result = json.load(f)
    if result.get("approach") != agent or result.get("fault_flag") != fault:
        raise RuntimeError("Harness returned a result for the wrong cell")

    from mlflow.tracking import MlflowClient
    experiment = mlflow.get_experiment_by_name("agentic_aiops_mttd_mttr")
    expected_name = f"{result['run_id']}_{agent}_scenarioa_{fault}_cart"
    runs = MlflowClient().search_runs(
        [experiment.experiment_id], filter_string=f"attributes.run_name = '{expected_name}'",
        order_by=["start_time DESC"], max_results=1,
    )
    if not runs or "detected" not in runs[0].data.metrics:
        raise RuntimeError("Harness metrics were not found in RHOAI MLflow")
    if runs[0].data.tags.get("aiops.prompt_registry_status") != "ok":
        raise RuntimeError("Agent prompts were not registered in RHOAI MLflow; see the agent trace artifacts")
    if not any(item.path == "harness_run.json" for item in MlflowClient().list_artifacts(runs[0].info.run_id)):
        raise RuntimeError("Harness result artifact was not found in RHOAI MLflow")

    # Large agent transcripts in KFP parameters broke report deserialization.
    # The full result and transcript remain in the MLflow run artifacts.
    keys = ("run_id", "approach", "fault_flag", "detected", "mttd_seconds",
            "rca_correct", "remediation_executed")
    compact = {key: result.get(key) for key in keys}
    compact["agent_error"] = bool(result.get("agent_error"))
    return compact


@dsl.component(base_image=HARNESS_IMAGE)
def generate_report(
    result_1: dict,
    result_2: dict,
    result_3: dict,
    result_4: dict,
    run_name: str,
    report_artifact: dsl.Output[dsl.Artifact],
) -> dict:
    """Aggregate all cell results into a summary report.

    Computes detection rate, mean MTTD, per-agent breakdown, and logs
    everything to MLflow as a summary run. Outputs the report as a
    pipeline artifact.
    """
    import json
    import os

    import certifi
    import mlflow
    import tempfile

    sa_path = "/var/run/secrets/kubernetes.io/serviceaccount"
    with open(f"{sa_path}/namespace") as namespace_file:
        workspace = namespace_file.read().strip()
    with open(f"{sa_path}/token") as token_file:
        os.environ["MLFLOW_TRACKING_TOKEN"] = token_file.read().strip()
    os.environ["MLFLOW_WORKSPACE"] = workspace
    os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow.redhat-ods-applications.svc:8443/mlflow"
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as bundle:
        for path in (certifi.where(), f"{sa_path}/service-ca.crt"):
            with open(path) as source:
                bundle.write(source.read() + "\n")
        os.environ["REQUESTS_CA_BUNDLE"] = bundle.name
    os.environ.pop("MLFLOW_TRACKING_INSECURE_TLS", None)
    os.environ.pop("MLFLOW_RUN_ID", None)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_workspace(workspace)

    results = [result_1, result_2, result_3, result_4]

    total = len(results)
    detected = sum(1 for r in results if r.get("detected"))
    mttds = [r.get("mttd_seconds") for r in results if r.get("mttd_seconds") is not None]
    mean_mttd = sum(mttds) / len(mttds) if mttds else None
    rca_correct = sum(1 for r in results if r.get("rca_correct"))
    remediation = sum(1 for r in results if r.get("remediation_executed"))
    errors = sum(1 for r in results if r.get("agent_error"))

    # Per-agent breakdown
    by_agent = {}
    for r in results:
        agent = r.get("approach", "unknown")
        if agent not in by_agent:
            by_agent[agent] = {"total": 0, "detected": 0, "mttds": []}
        by_agent[agent]["total"] += 1
        if r.get("detected"):
            by_agent[agent]["detected"] += 1
        if r.get("mttd_seconds") is not None:
            by_agent[agent]["mttds"].append(r["mttd_seconds"])

    report = {
        "run_name": run_name,
        "total_cells": total,
        "detected": detected,
        "detection_rate": round(detected / total, 3) if total else 0,
        "mean_mttd_seconds": round(mean_mttd, 1) if mean_mttd is not None else None,
        "rca_correct": rca_correct,
        "remediation_executed": remediation,
        "agent_errors": errors,
        "per_agent": {
            agent: {
                "total": v["total"],
                "detected": v["detected"],
                "detection_rate": round(v["detected"] / v["total"], 3),
                "mean_mttd_seconds": round(sum(v["mttds"]) / len(v["mttds"]), 1) if v["mttds"] else None,
            }
            for agent, v in by_agent.items()
        },
        "cells": [
            {
                "agent": r.get("approach"),
                "fault": r.get("fault_flag"),
                "detected": r.get("detected"),
                "mttd_seconds": r.get("mttd_seconds"),
                "rca_correct": r.get("rca_correct"),
                "remediation_executed": r.get("remediation_executed"),
                "agent_error": r.get("agent_error"),
            }
            for r in results
        ],
    }

    # Log to MLflow
    mlflow.set_experiment("agentic_aiops_mttd_mttr")
    with mlflow.start_run(run_name=f"{run_name}_summary"):
        mlflow.log_metric("total_cells", total)
        mlflow.log_metric("detected_count", detected)
        mlflow.log_metric("detection_rate", report["detection_rate"])
        if mean_mttd is not None:
            mlflow.log_metric("mean_mttd_seconds", mean_mttd)
        mlflow.log_metric("rca_correct_count", rca_correct)
        mlflow.log_metric("remediation_executed_count", remediation)
        mlflow.log_metric("agent_error_count", errors)
        mlflow.log_dict(report, "summary_report.json")
        mlflow.set_tag("run_type", "pipeline_summary")
        mlflow.set_tag("pipeline", run_name)

    # Write report to file for pipeline artifact
    with open(report_artifact.path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report


@dsl.pipeline(
    name="aiops-eval-matrix",
    description="AIOps harness: sequential agents × faults with environment reset and summary report",
)
def aiops_eval_matrix():
    """Sequential pipeline: reset → cell → reset → cell → ... → report.

    Each cell depends on the previous cell completing (and the reset step
    between them), preventing parallel fault-injection interference.
    """
    final_reset = reset_environment().set_caching_options(False)
    final_reset.set_display_name("final reset")
    with dsl.ExitHandler(exit_task=final_reset):
        results = []
        preflight = preflight_models().set_caching_options(False)
        previous_cell = preflight

        for agent in AGENTS:
            for fault in FAULTS:
                reset = reset_environment().set_caching_options(False)
                reset.set_display_name(f"reset / {fault}")
                cell = harness_cell(agent=agent, fault=fault).set_caching_options(False)
                cell.set_display_name(f"{agent.split('-')[0][:10]} / {fault}")
                reset.after(previous_cell)
                cell.after(reset)
                previous_cell = cell
                results.append(cell.output)

        report = generate_report(
            result_1=results[0],
            result_2=results[1],
            result_3=results[2],
            result_4=results[3],
            run_name="aiops-eval-matrix",
        ).set_caching_options(False)
        report.set_display_name("summary report")
        report.after(previous_cell)


if __name__ == "__main__":
    Compiler().compile(aiops_eval_matrix, "pipeline.yaml")
    print("Compiled: pipeline.yaml")
