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

    # Scale all deployments back to at least 1 replica
    for dep_name in targets:
        try:
            dep = apps.read_namespaced_deployment(dep_name, namespace)
            if (dep.spec.replicas or 0) < 1:
                apps.patch_namespaced_deployment(dep_name, namespace, {"spec": {"replicas": 1}})
                print(f"  scaled {dep_name} back to 1")
        except Exception as e:
            issues.append(f"{dep_name}: {e}")

    # Remove leftover harness network policies
    try:
        policies = net.list_namespaced_network_policy(namespace)
        for pol in policies.items:
            if pol.metadata.name.startswith("harness-block-"):
                net.delete_namespaced_network_policy(pol.metadata.name, namespace)
                print(f"  deleted network policy {pol.metadata.name}")
    except Exception:
        pass

    # Wait for pods to become Ready
    deadline = time.time() + 120
    while time.time() < deadline:
        all_ready = True
        for dep_name in ["cart", "checkout"]:
            try:
                dep = apps.read_namespaced_deployment(dep_name, namespace)
                ready = dep.status.ready_replicas or 0
                if ready < 1:
                    all_ready = False
            except Exception:
                all_ready = False
        if all_ready:
            break
        time.sleep(5)

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
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow.agentic-aiops-harness.svc.cluster.local:5000"
    os.environ["K8S_FAULT_NAMESPACE"] = "otel-demo"
    os.environ["K8S_NAMESPACE"] = "otel-demo"
    os.environ["PYTHONPATH"] = "/app"
    os.environ["AGENT_TIMEOUT"] = "600"

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
    try:
        with open("/tmp/result.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "no output", "detected": False, "mttd_seconds": None,
                "approach": agent, "fault_flag": fault}


@dsl.component(base_image=HARNESS_IMAGE)
def generate_report(
    results: list[dict],
    run_name: str,
) -> dict:
    """Aggregate all cell results into a summary report.

    Computes detection rate, mean MTTD, per-agent breakdown, and logs
    everything to MLflow as a summary run. Outputs the report as a
    pipeline artifact.
    """
    import json
    import os

    # MLflow logging
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow.agentic-aiops-harness.svc.cluster.local:5000"

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
        "mean_mttd_seconds": round(mean_mttd, 1) if mean_mttd else None,
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
    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("agentic_aiops_mttd_mttr")
        with mlflow.start_run(run_name=f"{run_name}_summary"):
            mlflow.log_metric("total_cells", total)
            mlflow.log_metric("detected_count", detected)
            mlflow.log_metric("detection_rate", report["detection_rate"])
            if mean_mttd:
                mlflow.log_metric("mean_mttd_seconds", mean_mttd)
            mlflow.log_metric("rca_correct_count", rca_correct)
            mlflow.log_metric("remediation_executed_count", remediation)
            mlflow.log_metric("agent_error_count", errors)
            mlflow.log_dict(report, "artifacts/summary_report.json")
            mlflow.set_tag("run_type", "pipeline_summary")
            mlflow.set_tag("pipeline", run_name)
    except Exception as e:
        print(f"[report] MLflow logging failed: {e}")

    # Write report to file for pipeline artifact
    with open("/tmp/report.json", "w") as f:
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
    results = []
    previous_cell = None

    for agent in AGENTS:
        for fault in FAULTS:
            # Reset environment before each cell
            reset = reset_environment()
            reset.set_display_name(f"reset / {fault}")

            # Run the harness cell
            cell = harness_cell(agent=agent, fault=fault)
            cell.set_display_name(f"{agent.split('-')[0][:10]} / {fault}")

            # Chain sequentially: reset after previous cell, cell after reset
            if previous_cell is not None:
                reset.after(previous_cell)
            cell.after(reset)

            previous_cell = cell
            results.append(cell.output)

    # Generate the final report after all cells complete
    report = generate_report(
        results=results,
        run_name="aiops-eval-matrix",
    )
    report.set_display_name("summary report")
    report.after(previous_cell)


if __name__ == "__main__":
    Compiler().compile(aiops_eval_matrix, "pipeline.yaml")
    print("Compiled: pipeline.yaml")
