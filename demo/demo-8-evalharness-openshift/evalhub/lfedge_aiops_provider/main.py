#!/usr/bin/env python3
"""EvalHub provider that runs one real AIOps harness cell.

Contract (eval-hub runtime):
  - Read job spec from /meta/job.json (EVALHUB_MODE=k8s).
  - POST benchmark status events to the sidecar proxy at job.callback_url:
      POST {callback_url}/api/v1/evaluations/jobs/{job_id}/events

The job runs ``code/harness/run_harness.py`` for one (agent, scenario, fault,
context) cell, parses the harness JSON, and reports the real MTTD/MTTR/RCA
metrics. If the cell cannot run, it reports ``failed`` — it never fabricates
timings.

Benchmark parameters understood (all optional):
  agent, scenario, fault, variant, context_c, detection_timeout,
  fault_duration, judge, api_key, model_url, model_name, run_timeout,
  clickhouse_http, namespace

The provider needs the same RBAC as the harness: write access in the OTEL demo
namespace. See manifests/rbac/harness-rbac.yaml and docs/DEPLOY.md.
"""
from __future__ import annotations

import json
import os
import ssl
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JOB_PATH = os.environ.get("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
DEFAULT_CALLBACK = "http://127.0.0.1:8080"
SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
SA_NAMESPACE_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

ROOT = Path(__file__).resolve().parents[2]

AGENTS: dict[str, dict[str, str]] = {
    "rhdp_deepseek-r1-distill-qwen-14b": {
        "script": "code/agents/deepseek_agent/agent.py",
        "base_env": "DEEPSEEK_API_BASE",
        "key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-r1-distill-qwen-14b",
    },
    "rhdp_qwen3-14b": {
        "script": "code/agents/qwen3_agent/agent.py",
        "base_env": "QWEN3_API_BASE",
        "key_env": "QWEN3_API_KEY",
        "model_env": "QWEN3_MODEL",
        "default_model": "qwen3-14b",
    },
    "rhdp_qwen38-27b": {
        "script": "code/agents/qwen38_27b_agent/agent.py",
        "base_env": "QWEN38_API_BASE",
        "key_env": "QWEN38_API_KEY",
        "model_env": "QWEN38_MODEL",
        "default_model": "qwen38-27b",
    },
    "litemaas_qwen3.8-27b": {
        "script": "code/agents/qwen36_27b_agent/agent.py",
        "base_env": "QWEN36_API_BASE",
        "key_env": "QWEN36_API_KEY",
        "model_env": "QWEN36_MODEL",
        "default_model": "Qwen3.8-27B",
    },
    "rhdp_granite-3-2-8b-instruct": {
        "script": "code/agents/granite_agent/agent.py",
        "base_env": "GRANITE_API_BASE",
        "key_env": "GRANITE_API_KEY",
        "model_env": "GRANITE_MODEL",
        "default_model": "granite-3-2-8b-instruct",
    },
    "prelude_glm-53-flash": {
        "script": "code/agents/glm_53_flash_agent/agent.py",
        "base_env": "GLM_API_BASE",
        "key_env": "GLM_API_KEY",
        "model_env": "GLM_MODEL",
        "default_model": "glm-53-flash",
    },
}


def _bearer_token() -> str | None:
    try:
        tok = Path(SA_TOKEN_PATH).read_text(encoding="utf-8").strip()
        return tok or None
    except OSError:
        return None


def _tenant_namespace() -> str | None:
    try:
        return Path(SA_NAMESPACE_PATH).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_job() -> dict[str, Any]:
    with open(JOB_PATH, encoding="utf-8") as f:
        return json.load(f)


def _post_event(callback_base: str, job_id: str, payload: dict[str, Any]) -> None:
    base = (callback_base or DEFAULT_CALLBACK).rstrip("/")
    url = f"{base}/api/v1/evaluations/jobs/{urllib.parse.quote(job_id)}/events"
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    tok = _bearer_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    ns = _tenant_namespace()
    if ns:
        headers["X-Tenant"] = ns
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    ctx = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:4000]
        print(f"[lfedge_aiops] HTTP {e.code} posting event: {detail}", flush=True)
        raise


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _base_url(value: str) -> str:
    v = (value or "").strip().rstrip("/")
    if v.endswith("/chat/completions"):
        v = v[: -len("/chat/completions")]
    return v


def _run_cell(params: dict[str, Any], job: dict[str, Any], workdir: Path) -> dict[str, Any]:
    agent = str(params.get("agent") or "rhdp_qwen3-14b").strip()
    if agent not in AGENTS:
        raise ValueError(f"unknown agent {agent!r}; choose one of {sorted(AGENTS)}")
    spec = AGENTS[agent]

    scenario = str(params.get("scenario") or "a").strip()
    fault = str(params.get("fault") or "scale_zero").strip()
    variant = str(params.get("variant") or "cart").strip()
    context_c = str(params.get("context_c") or "0").strip()
    detection_timeout = int(params.get("detection_timeout") or 900)
    fault_duration = int(params.get("fault_duration") or 120)
    run_timeout = int(params.get("run_timeout") or 1800)
    judge = _as_bool(params.get("judge"), default=False)

    model = job.get("model") or {}
    api_base = _base_url(str(params.get("model_url") or model.get("url") or ""))
    model_name = str(params.get("model_name") or model.get("name") or "").strip()
    api_key = str(params.get("api_key") or os.environ.get("LFEDGE_MODEL_API_KEY") or "").strip()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    env["MLFLOW_MODE"] = "external"
    # The EvalHub sidecar overrides MLFLOW_TRACKING_URI with its own proxy
    # (http://localhost:8080). LFEDGE_MLFLOW_URI carries the real tracking
    # server so harness runs are visible in the standalone MLflow UI.
    lfedge_mlflow = (os.environ.get("LFEDGE_MLFLOW_URI") or "").strip()
    if lfedge_mlflow:
        env["MLFLOW_TRACKING_URI"] = lfedge_mlflow
        # Strip EvalHub sidecar MLflow vars that confuse the MLflow 3.x client
        # (token path / workspace headers are for the RHOAI-managed MLflow, not ours).
        for k in ("MLFLOW_TRACKING_TOKEN_PATH", "MLFLOW_WORKSPACE",
                   "MLFLOW_TRACKING_SERVER_CERT_PATH"):
            env.pop(k, None)
    env["HARNESS_PYTHON"] = sys.executable
    env["K8S_FAULT_NAMESPACE"] = str(
        params.get("namespace") or os.environ.get("K8S_FAULT_NAMESPACE") or "otel-demo"
    )
    env["OTEL_DEMO_NAMESPACE"] = env["K8S_FAULT_NAMESPACE"]
    if params.get("clickhouse_http"):
        env["CLICKHOUSE_HTTP"] = str(params["clickhouse_http"])
    if api_base:
        env[spec["base_env"]] = api_base
    if api_key:
        env[spec["key_env"]] = api_key
    if model_name:
        env[spec["model_env"]] = model_name
    env.setdefault(spec["model_env"], spec["default_model"])
    env.setdefault("CONTEXT_REPO_PATH", str(ROOT / "context" / "opentelemetry-demo"))

    print(f"[lfedge_aiops] checking cluster baseline for {env['K8S_FAULT_NAMESPACE']}/{variant}", flush=True)
    gate = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "pre_run_health_gate.py"),
            "--target",
            variant,
            "--namespace",
            env["K8S_FAULT_NAMESPACE"],
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if gate.stdout:
        print(gate.stdout, flush=True)
    if gate.returncode != 0:
        raise RuntimeError(f"cluster baseline gate failed: {gate.stderr[-1000:]}")

    out_path = workdir / "harness_run.json"
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "code" / "harness" / "run_harness.py"),
        "--scenario",
        scenario,
        "--classifier",
        str(ROOT / spec["script"]),
        "--flag",
        fault,
        "--variant",
        variant,
        "--context-c",
        context_c,
        "--detection-timeout",
        str(detection_timeout),
        "--fault-duration",
        str(fault_duration),
        "--output",
        str(out_path),
    ]
    if not judge:
        cmd.append("--no-judge")

    print(f"[lfedge_aiops] running harness: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=run_timeout,
    )
    if proc.stdout:
        print(proc.stdout[-8000:], flush=True)
    if proc.stderr:
        print(proc.stderr[-8000:], file=sys.stderr, flush=True)

    if not out_path.exists():
        raise RuntimeError(
            f"harness produced no output (exit={proc.returncode}); stderr={proc.stderr[-1000:]}"
        )
    data = json.loads(out_path.read_text(encoding="utf-8"))
    # run_harness.py writes a single run dict for one agent, and {"runs": [...]}
    # only when multiple agents/scenarios were executed. Accept both shapes.
    runs = data.get("runs")
    if runs is None and isinstance(data, dict) and "run_id" in data:
        runs = [data]
    if not runs:
        raise RuntimeError(
            f"harness output contained no runs (keys={sorted(data.keys()) if isinstance(data, dict) else type(data)})"
        )
    return runs[0]


def _metrics(run: dict[str, Any]) -> dict[str, float]:
    def num(key: str, default: float = 0.0) -> float:
        v = run.get(key)
        return default if v is None else float(v)

    def flag(key: str) -> float:
        return 1.0 if bool(run.get(key)) else 0.0

    return {
        "mttd_seconds": num("mttd_seconds"),
        "mttr_seconds": num("mttr_seconds"),
        "mttr_verified_seconds": num("mttr_verified_seconds"),
        "detected": flag("detected"),
        "rca_correct": flag("rca_correct"),
        "remediation_correct": flag("remediation_correct"),
        "remediation_executed": flag("remediation_executed"),
        "recovery_verified": flag("recovery_verified"),
        "llm_invoked": flag("llm_invoked"),
        "agent_error": flag("agent_error"),
        "run_count": 1.0,
        "metric_source": 3.0,  # 3 = real harness run
    }


def main() -> int:
    try:
        job = _load_job()
    except FileNotFoundError:
        print(
            f"[lfedge_aiops] job spec not found at {JOB_PATH}; this provider is "
            "meant to run inside an EvalHub Kubernetes job",
            file=sys.stderr,
            flush=True,
        )
        return 2
    job_id = job["id"]
    provider_id = job["provider_id"]
    benchmark_id = job["benchmark_id"]
    benchmark_index = int(job.get("benchmark_index", 0))
    callback = job.get("callback_url") or DEFAULT_CALLBACK
    params = job.get("parameters") or {}

    started = _utc_now()
    _post_event(
        callback,
        job_id,
        {
            "benchmark_status_event": {
                "provider_id": provider_id,
                "id": benchmark_id,
                "benchmark_index": benchmark_index,
                "status": "running",
                "started_at": started,
            }
        },
    )

    try:
        with tempfile.TemporaryDirectory(prefix="lfedge-aiops-") as td:
            run = _run_cell(params, job, Path(td))
        metrics = _metrics(run)
        _post_event(
            callback,
            job_id,
            {
                "benchmark_status_event": {
                    "provider_id": provider_id,
                    "id": benchmark_id,
                    "benchmark_index": benchmark_index,
                    "status": "completed",
                    "started_at": started,
                    "completed_at": _utc_now(),
                    "metrics": metrics,
                }
            },
        )
        print(f"[lfedge_aiops] completed metrics={json.dumps(metrics)}", flush=True)
        return 0
    except Exception as e:
        print(f"[lfedge_aiops] failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        try:
            _post_event(
                callback,
                job_id,
                {
                    "benchmark_status_event": {
                        "provider_id": provider_id,
                        "id": benchmark_id,
                        "benchmark_index": benchmark_index,
                        "status": "failed",
                        "started_at": started,
                        "completed_at": _utc_now(),
                        "error_message": {"message": f"{type(e).__name__}: {e}"[:2000], "message_code": "provider_error"},
                    }
                },
            )
        except Exception as post_err:
            print(f"[lfedge_aiops] could not post failure: {post_err}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
