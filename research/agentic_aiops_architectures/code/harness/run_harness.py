#!/usr/bin/env python3
"""
MTTD/MTTR Harness for Agentic AIOps comparison.

Orchestrates chaos injection (via flagd), invokes LLM agents (Ollama, MaaS),
records events, computes MTTD and MTTR, logs to MLflow.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.agents.mlflow_agent_logging import log_mlflow_dataset_inputs_for_harness_summary
from code.agents.mlflow_config import DEFAULT_MLFLOW_TRACKING_URI


def _print_mlflow_tracking_banner(uri: str, insecure: bool) -> None:
    """Print the resolved MLflow tracking URI prominently at harness startup."""
    if not (uri or "").strip():
        return
    w = max(76, min(120, len(uri) + 12))
    bar = "=" * w
    sub = (
        "MLFLOW_TRACKING_INSECURE_TLS=true (self-signed HTTPS OK)"
        if insecure
        else "TLS: use default certificate verification"
    )
    print("\n\n" + bar, flush=True)
    print("MLFLOW  TRACKING  SERVER  (  O P E N S H I F T  )".center(w), flush=True)
    print(bar, flush=True)
    print("", flush=True)
    # URI in “large” block: extra vertical space + full-width emphasis
    for _ in range(2):
        print("", flush=True)
    print(uri.center(w), flush=True)
    print("", flush=True)
    print(sub.center(w), flush=True)
    print("", flush=True)
    print(bar + "\n", flush=True)


def _python_for_subprocess() -> str:
    """Return Python executable for agent subprocess.

    Cursor/VS Code may set sys.executable to the editor/AppImage — that binary must not run agent code.
    Override with env HARNESS_PYTHON or PYTHON_BIN (absolute path recommended in IDE terminals).
    """
    for key in ("HARNESS_PYTHON", "PYTHON_BIN"):
        override = (os.environ.get(key) or "").strip()
        if override:
            return override
    exe = (sys.executable or "").lower()
    bad_substrings = ("cursor", "appimage", "electron", "code-insiders", "vscode")
    if exe and not any(b in exe for b in bad_substrings) and "python" in exe:
        return sys.executable
    # Prefer real CPython on PATH (Fedora/RHEL often has python3.13)
    for cand in ("python3.13", "python3.12", "python3"):
        w = shutil.which(cand)
        if w:
            return w
    return "python3"

# Config file (optional): config/harness.yaml
HARNESS_CONFIG = ROOT / "config" / "harness.yaml"


def _load_harness_config() -> dict:
    """Load config/harness.yaml if present."""
    if not HARNESS_CONFIG.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(HARNESS_CONFIG.read_text()) or {}
    except Exception:
        return {}


def _load_mlflow_config() -> tuple[str, bool]:
    """MLflow tracking URI and insecure_tls: env > config > default."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    insecure = os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "").lower() in ("1", "true", "yes")
    cfg = _load_harness_config()
    if isinstance(cfg.get("mlflow"), dict):
        mlflow_cfg = cfg["mlflow"]
        if not uri and mlflow_cfg.get("tracking_uri"):
            uri = mlflow_cfg["tracking_uri"]
        if not os.environ.get("MLFLOW_TRACKING_INSECURE_TLS"):
            insecure = mlflow_cfg.get("insecure_tls", False)
    return (uri or "").strip() or DEFAULT_MLFLOW_TRACKING_URI, insecure


def _resolve_agent_timeout_seconds() -> int:
    """AGENT_TIMEOUT env > harness.yaml harness.agent_timeout_seconds > default (1200s for slow LLM+tools)."""
    if os.environ.get("AGENT_TIMEOUT"):
        return max(30, int(os.environ["AGENT_TIMEOUT"]))
    cfg = _load_harness_config()
    h = cfg.get("harness")
    if isinstance(h, dict) and h.get("agent_timeout_seconds") is not None:
        return max(30, int(h["agent_timeout_seconds"]))
    return 1200


def _resolve_max_tool_iterations_for_subprocess() -> str:
    """MAX_TOOL_ITERATIONS env > harness.yaml harness.max_tool_iterations > default 8 (faster than 15)."""
    if os.environ.get("MAX_TOOL_ITERATIONS"):
        return str(max(1, int(os.environ["MAX_TOOL_ITERATIONS"])))
    cfg = _load_harness_config()
    h = cfg.get("harness")
    if isinstance(h, dict) and h.get("max_tool_iterations") is not None:
        return str(max(1, int(h["max_tool_iterations"])))
    return "8"


def _strip_mlflow_status_from_agent_stdout(raw: str) -> str:
    """Drop MLflow CLI status lines (run URL / experiment URL) that leak into agent stdout when MLFLOW_RUN_ID is set."""
    if not raw:
        return raw
    lines: list[str] = []
    for ln in raw.splitlines():
        t = ln.strip()
        if "View run " in ln or "View experiment " in ln:
            continue
        if t.startswith("🏃") or t.startswith("🧪"):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def _parse_agent_json_stdout(raw: str) -> dict | None:
    """Parse agent --json output; tolerate trailing noise or markdown fences."""
    if not raw or not raw.strip():
        return None
    s = raw.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    s2 = s
    if "```" in s2:
        s2 = re.sub(r"^```(?:json)?\s*", "", s2, flags=re.IGNORECASE)
        s2 = re.sub(r"\s*```\s*$", "", s2)
        try:
            return json.loads(s2.strip())
        except json.JSONDecodeError:
            pass
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _load_flagd_config() -> tuple[bool, str]:
    """Flagd use_openshift_cm and configmap_namespace: env > config > default."""
    use_cm = os.environ.get("FLAGD_USE_OPENSHIFT_CM", "").lower() in ("1", "true", "yes")
    ns = os.environ.get("FLAGD_CONFIGMAP_NAMESPACE", "otel-demo")
    cfg = _load_harness_config()
    if isinstance(cfg.get("flagd"), dict):
        flagd_cfg = cfg["flagd"]
        if not os.environ.get("FLAGD_USE_OPENSHIFT_CM"):
            use_cm = flagd_cfg.get("use_openshift_cm", False)
        if not os.environ.get("FLAGD_CONFIGMAP_NAMESPACE"):
            ns = flagd_cfg.get("configmap_namespace", "otel-demo")
    if use_cm and "FLAGD_CONFIGMAP_NAMESPACE" not in os.environ:
        os.environ["FLAGD_CONFIGMAP_NAMESPACE"] = ns
    return use_cm, ns


# Env-based config
FLAGD_READ_URL = os.environ.get("FLAGD_READ_URL", "http://localhost:8080/feature/api/read")
FLAGD_WRITE_URL = os.environ.get("FLAGD_WRITE_URL", "http://localhost:8080/feature/api/write")
FLAGD_USE_OPENSHIFT_CM, _ = _load_flagd_config()
CLICKHOUSE_HTTP = os.environ.get("CLICKHOUSE_HTTP", "http://localhost:8123")
MLFLOW_TRACKING_URI, MLFLOW_INSECURE_TLS = _load_mlflow_config()
if MLFLOW_INSECURE_TLS and "MLFLOW_TRACKING_INSECURE_TLS" not in os.environ:
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
EVENT_LEDGER_PATH = os.environ.get("EVENT_LEDGER_PATH", str(ROOT / "data" / "event_ledger.jsonl"))


# MLflow experiment name and description (UI: experiment notes / description)
MLFLOW_EXPERIMENT_NAME = "agentic_aiops_mttd_mttr"
HARNESS_EXPERIMENT_DESCRIPTION = """\
**Agentic AIOps — MTTD / MTTR harness**

End-to-end comparison of LLM agents for **failure detection** and **remediation suggestions** on the OpenTelemetry demo stack.

**Procedure**
1. Inject a fault via **flagd** (HTTP API or OpenShift ConfigMap patch).
2. Poll **ClickHouse** telemetry (logs, traces, metrics) from `since` = fault injection time.
3. Run agentic tool-calling loops (**ollama_qwen2** / Ollama, **deepseek_agent** / **qwen3_agent** / **llama_scout_agent** / MaaS).
4. Record **MTTD** (time to first positive detection) and **MTTR** (time from alert to fault cleared).
5. Log prompts, tool calls, and run summary to this experiment.

**Metrics**
- `mttd_seconds`, `mttr_seconds`, `detected` (per run)
- Artifacts: `harness_run.json`, `artifacts/prompts.json`, `artifacts/tool_calls.json`, etc.

**Agents**: ollama_qwen (Ollama), maas_deepseek (DeepSeek R1), maas_qwen3 (Qwen3), maas_llama-scout (Llama Scout).

*(The section **This harness invocation** is appended at run start with the active CLI config: how many agents, which scripts, timeouts, fault flag.)*
"""


def _experiment_description_for_mlflow() -> str:
    """Experiment-level note: config `mlflow.experiment_description` overrides default."""
    cfg = _load_harness_config()
    ml = cfg.get("mlflow")
    if isinstance(ml, dict):
        custom = ml.get("experiment_description") or ml.get("experiment_note")
        if isinstance(custom, str) and custom.strip():
            return custom.strip()
    return HARNESS_EXPERIMENT_DESCRIPTION.strip()


def _format_harness_invocation_note(
    *,
    agents_to_run: list[tuple[str, Path]],
    flag: str,
    variant: str,
    poll_interval: int,
    detection_timeout: int,
    fault_duration: int,
    skip_fault_injection: bool,
    selection_mode: str,
) -> str:
    """Markdown block: concrete harness config for this process (logged to MLflow experiment note)."""
    n = len(agents_to_run)
    lines = [
        "**This harness invocation**",
        f"- **Agents scheduled:** {n} (`{selection_mode}`)",
        f"- **Approaches & scripts:**",
    ]
    for approach, script in agents_to_run:
        try:
            rel = script.resolve().relative_to(ROOT)
            sp = str(rel)
        except ValueError:
            sp = str(script)
        lines.append(f"  - `{approach}` → `{sp}`")
    lines.extend(
        [
            f"- **Fault:** `{flag}` = `{variant}`",
            f"- **skip_fault_injection:** {skip_fault_injection}",
            f"- **poll_interval_s:** {poll_interval}",
            f"- **detection_timeout_s:** {detection_timeout} (wall-clock)",
            f"- **fault_duration_s:** {fault_duration}",
            f"- **agent subprocess timeout (effective):** {_resolve_agent_timeout_seconds()}s (env `AGENT_TIMEOUT` or `harness.agent_timeout_seconds`)",
            f"- **MAX_TOOL_ITERATIONS (effective):** {_resolve_max_tool_iterations_for_subprocess()}",
            f"- **Registered agents** in `AGENTS` (default pool): {len(AGENTS)}",
        ]
    )
    return "\n".join(lines)


def _set_mlflow_experiment_description(*, invocation_note: str | None = None) -> None:
    """Write experiment description to MLflow (`mlflow.note.content`). Call after set_experiment."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if exp is None:
            return
        client = MlflowClient()
        desc = _experiment_description_for_mlflow()
        if invocation_note and invocation_note.strip():
            desc = f"{desc}\n\n{invocation_note.strip()}"
        # MLflow UI shows this as the experiment description / note
        client.set_experiment_tag(exp.experiment_id, "mlflow.note.content", desc)
    except Exception as e:
        print(f"[harness] MLflow experiment description (skipped): {e}", flush=True)


# All agents: (approach_name, script_path). Approach name states the LLM.
AGENTS = [
    ("ollama_qwen", ROOT / "code" / "agents" / "ollama_qwen2" / "agent.py"),
    ("maas_deepseek", ROOT / "code" / "agents" / "deepseek_agent" / "agent.py"),
    ("maas_qwen3", ROOT / "code" / "agents" / "qwen3_agent" / "agent.py"),
    ("maas_llama-scout", ROOT / "code" / "agents" / "llama_scout_agent" / "agent.py"),
]


@dataclass
class HarnessRun:
    run_id: str
    fault_flag: str
    fault_variant: str
    fault_injection_time: str
    first_alert_time: str | None
    remediation_suggested_time: str | None
    fault_recovery_time: str | None
    mttd_seconds: float | None
    mttr_seconds: float | None
    detected: bool
    suggested_remediations: list[str]
    suggested_root_cause: str | None
    approach: str = "ollama_qwen2.5"
    llm_invoked: bool = False
    agent_error: str | None = None  # e.g. timeout, connection failure
    agent_output: str | None = None  # raw JSON the agent printed to stdout


def set_flag(flag_name: str, variant: str) -> bool:
    """Set feature flag via flagd API or OpenShift ConfigMap (when FLAGD_USE_OPENSHIFT_CM=1)."""
    if FLAGD_USE_OPENSHIFT_CM:
        return _set_flag_openshift(flag_name, variant)
    return _set_flag_http(flag_name, variant)


def _set_flag_openshift(flag_name: str, variant: str) -> bool:
    """Set flag via ConfigMap patch + rollout restart (workaround for flagd-ui write 404)."""
    try:
        script = ROOT / "scripts" / "set-flag-openshift.py"
        r = subprocess.run(
            [_python_for_subprocess(), str(script), flag_name, variant],
            capture_output=True,
            text=True,
            timeout=150,
            cwd=str(ROOT),
        )
        if r.returncode != 0:
            print(f"[harness] set_flag (openshift) failed: {r.stderr}")
            return False
        time.sleep(5)  # Allow flagd to stabilize after restart
        return True
    except Exception as e:
        print(f"[harness] set_flag (openshift) failed: {e}")
        return False


def _set_flag_http(flag_name: str, variant: str) -> bool:
    """Set feature flag via flagd-ui HTTP API."""
    try:
        import requests
        verify = os.environ.get("FLAGD_VERIFY_SSL", "true").lower() not in ("0", "false", "no")
        r = requests.get(FLAGD_READ_URL, timeout=10, verify=verify)
        r.raise_for_status()
        flags = (r.json() or {}).get("flags", {})
        if flag_name not in flags:
            print(f"[harness] Flag {flag_name} not found; available: {list(flags.keys())[:10]}...")
            return False
        doc = {"$schema": "https://flagd.dev/schema/v0/flags.json", "flags": flags}
        doc["flags"][flag_name]["defaultVariant"] = variant
        w = requests.post(FLAGD_WRITE_URL, json={"data": doc}, timeout=10, verify=verify)
        w.raise_for_status()
        time.sleep(3)
        return True
    except Exception as e:
        print(f"[harness] set_flag failed: {e}")
        return False


def invoke_classifier(
    since_ts: str,
    script: Path | None = None,
    *,
    subprocess_timeout: int | None = None,
) -> dict:
    """Invoke agent script and return result dict.

    subprocess_timeout: cap for this single subprocess.run (seconds). When None, uses
    full harness agent timeout. Callers should pass min(agent_timeout, detection_window_remaining).
    """
    script = script or AGENTS[0][1]
    env = os.environ.copy()
    env["CLICKHOUSE_HTTP"] = CLICKHOUSE_HTTP
    env["PYTHONUNBUFFERED"] = "1"
    env["MAX_TOOL_ITERATIONS"] = _resolve_max_tool_iterations_for_subprocess()
    # Agent subprocess must use the same MLflow backend as the harness (avoid wrong local URI / missing TLS).
    env["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    if MLFLOW_INSECURE_TLS:
        env["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    rid = os.environ.get("MLFLOW_RUN_ID")
    if rid:
        env["MLFLOW_RUN_ID"] = rid
    py = _python_for_subprocess()
    args = [py, "-u", str(script), "--since", since_ts, "--json"]
    err_thresh = os.environ.get("CLASSIFIER_ERROR_THRESHOLD")
    if err_thresh:
        args.extend(["--error-threshold", str(err_thresh)])
    configured = _resolve_agent_timeout_seconds()
    if subprocess_timeout is not None:
        agent_timeout = max(5, min(int(subprocess_timeout), configured))
    else:
        agent_timeout = configured
    try:
        r = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=agent_timeout,
            env=env,
            cwd=str(ROOT),
        )
        stdout_clean = _strip_mlflow_status_from_agent_stdout(r.stdout or "")
        parsed = _parse_agent_json_stdout(stdout_clean)
        if parsed is not None:
            parsed["agent_output_raw"] = stdout_clean.strip()
            if "llm_invoked" not in parsed:
                parsed["llm_invoked"] = False
            if "suggested_remediations" not in parsed:
                parsed["suggested_remediations"] = []
            if r.returncode not in (0, 1) and not parsed.get("agent_fatal"):
                parsed["agent_note"] = f"nonzero_exit_{r.returncode}"
            return parsed
        err_parts = []
        if not (r.stdout and r.stdout.strip()):
            err_parts.append("empty agent stdout")
        if r.stderr:
            err_parts.append(r.stderr[:1200])
        err = "; ".join(err_parts) if err_parts else f"exit {r.returncode}"
    except subprocess.TimeoutExpired as e:
        stdout = getattr(e, "stdout", None) or ""
        stderr = getattr(e, "stderr", None) or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stdout_clean = _strip_mlflow_status_from_agent_stdout(stdout)
        parsed = _parse_agent_json_stdout(stdout_clean)
        if parsed is not None:
            parsed["agent_output_raw"] = stdout_clean.strip()
            parsed.setdefault("llm_invoked", parsed.get("llm_invoked", False))
            parsed["agent_error"] = (
                f"subprocess timeout after {agent_timeout}s (JSON recovered from partial stdout)"
            )
            return parsed
        err = f"timeout after {agent_timeout}s"
        if stderr:
            err += f"; stderr: {stderr[:800]}"
    except Exception as e:
        err = str(e)
    return {"detected": False, "agent_error": err, "suggested_remediations": [], "llm_invoked": False}


def append_ledger(entry: dict) -> None:
    """Append event to ledger file."""
    Path(EVENT_LEDGER_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(EVENT_LEDGER_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _log_run_to_mlflow(run: HarnessRun, log_mlflow: bool, mlflow_run=None, skip_fault_injection: bool = False) -> None:
    """Log HarnessRun to MLflow. Uses existing run if mlflow_run is active."""
    if not log_mlflow:
        return
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        mttd_str = f"{run.mttd_seconds:.1f}s" if run.mttd_seconds is not None else "N/A"
        mttr_str = f"{run.mttr_seconds:.1f}s" if run.mttr_seconds is not None else "N/A"
        fault_mode = "skip_fault_injection (analyzing existing telemetry)" if skip_fault_injection else f"{run.fault_flag}={run.fault_variant} (injected via flagd)"
        desc = f"""**MTTD/MTTR Agentic AIOps Experiment**
- **Approach**: {run.approach}
- **Fault mode**: {fault_mode}
- **Detection**: {'Yes' if run.detected else 'No'}
- **MTTD** (fault → detection): {mttd_str}
- **MTTR** (fault → remediation suggested): {mttr_str}
- **LLM invoked**: {run.llm_invoked}"""
        if run.suggested_root_cause:
            desc += f"\n- **Root cause**: {run.suggested_root_cause}"
        if run.suggested_remediations:
            desc += f"\n- **Remediations**: {len(run.suggested_remediations)} step(s)"
        if run.agent_error:
            desc += f"\n- **Agent error**: {run.agent_error[:300]}..."

        # Dataset metadata (telemetry source for this run)
        dataset_meta = {
            "source": "clickhouse",
            "tables": ["otel.otel_logs", "otel.otel_traces", "otel.otel_metrics_gauge", "otel.otel_metrics_sum"],
            "since": run.fault_injection_time,
            "fault_flag": run.fault_flag,
            "fault_variant": run.fault_variant,
            "approach": run.approach,
            "clickhouse_http": CLICKHOUSE_HTTP,
        }
        def _do_log():
            mlflow.set_tag("mlflow.note.content", desc)
            mlflow.log_params({
                "fault_flag": run.fault_flag,
                "fault_variant": run.fault_variant,
                "fault_injected": str(not skip_fault_injection).lower(),
                "approach": run.approach,
                "llm_invoked": str(run.llm_invoked).lower(),
            })
            mlflow.log_metrics({
                "mttd_seconds": run.mttd_seconds if run.mttd_seconds is not None else -1,
                "mttr_seconds": run.mttr_seconds if run.mttr_seconds is not None else -1,
                "detected": 1.0 if run.detected else 0.0,
            })
            mlflow.log_dict(asdict(run), "harness_run.json")
            mlflow.log_dict(dataset_meta, "artifacts/dataset_metadata.json")
            if run.agent_error:
                mlflow.log_dict({"agent_error": run.agent_error}, "artifacts/agent_error.json")
            if run.agent_output:
                mlflow.log_text(run.agent_output, "artifacts/agent_output.json")

            ar = mlflow.active_run()
            if ar and ar.info.run_id:
                from mlflow.tracking import MlflowClient

                _hc = MlflowClient(MLFLOW_TRACKING_URI)
                log_mlflow_dataset_inputs_for_harness_summary(
                    _hc,
                    ar.info.run_id,
                    {
                        "harness_run_id": run.run_id,
                        "approach": run.approach,
                        "fault_flag": run.fault_flag,
                        "fault_variant": run.fault_variant,
                        "detected": run.detected,
                        "llm_invoked": run.llm_invoked,
                        "mttd_seconds": run.mttd_seconds,
                        "mttr_seconds": run.mttr_seconds,
                        "suggested_root_cause": run.suggested_root_cause or "",
                        "suggested_remediations": run.suggested_remediations,
                        "fault_injection_time": run.fault_injection_time,
                        "first_alert_time": run.first_alert_time or "",
                        "agent_error": (run.agent_error or "")[:4000],
                    },
                )

        if mlflow_run and mlflow.active_run() and mlflow.active_run().info.run_id == mlflow_run.info.run_id:
            _do_log()
            mlflow.end_run()
        else:
            with mlflow.start_run(run_name=f"{run.run_id}_{run.approach}_{run.fault_flag}_{run.fault_variant}"):
                _do_log()
    except ImportError:
        print("[harness] mlflow not installed; skip logging")
    except Exception as e:
        print(f"[harness] MLflow log error: {e}")


def run_single_experiment(
    fault_flag: str,
    fault_variant: str,
    poll_interval: int,
    detection_timeout: int,
    fault_duration: int,
    log_mlflow: bool,
    classifier_script: Path | None = None,
    approach: str = "ollama_qwen2.5",
    skip_fault_injection: bool = False,
) -> HarnessRun:
    """Run one fault injection experiment; return HarnessRun."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    # For local demo: use a past time so seeded data is in the detection window
    if skip_fault_injection:
        from datetime import timedelta
        fault_injection_time = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    else:
        fault_injection_time = datetime.now(timezone.utc).isoformat()

    # Start MLflow run early so agent's run_remediation can log during execution
    mlflow_run = None
    if log_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            mlflow_run = mlflow.start_run(run_name=f"{run_id}_{approach}_{fault_flag}_{fault_variant}")
            os.environ["MLFLOW_RUN_ID"] = mlflow_run.info.run_id
            os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
        except ImportError:
            pass
        except Exception as e:
            print(f"[harness] MLflow start_run error: {e}")

    append_ledger({
        "event": "fault_injection",
        "run_id": run_id,
        "flag": fault_flag,
        "variant": fault_variant,
        "approach": approach,
        "timestamp": fault_injection_time,
    })

    if not skip_fault_injection and not set_flag(fault_flag, fault_variant):
        run = HarnessRun(
            run_id=run_id,
            fault_flag=fault_flag,
            fault_variant=fault_variant,
            fault_injection_time=fault_injection_time,
            first_alert_time=None,
            remediation_suggested_time=None,
            fault_recovery_time=datetime.now(timezone.utc).isoformat(),
            mttd_seconds=None,
            mttr_seconds=None,
            detected=False,
            suggested_remediations=[],
            suggested_root_cause=None,
            approach=approach,
        )
        _log_run_to_mlflow(run, log_mlflow, mlflow_run=mlflow_run, skip_fault_injection=skip_fault_injection)
        if mlflow_run:
            try:
                del os.environ["MLFLOW_RUN_ID"]
            except KeyError:
                pass
        return run

    # Allow telemetry to flow to ClickHouse before first poll
    time.sleep(2 if skip_fault_injection else 15)

    first_alert_time = None
    agent_result = {}
    last_result = {}  # capture last poll result for agent_error when not detected
    ever_llm_invoked = False  # any poll succeeded talking to the LLM (last poll may time out)
    # Wall-clock deadline: each invoke_classifier can take up to AGENT_TIMEOUT, so
    # counting only poll_interval (old behavior) allowed unbounded total wait.
    deadline = time.monotonic() + float(detection_timeout)
    agent_limit = float(_resolve_agent_timeout_seconds())

    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        # Avoid a final poll with e.g. 60s left while the agent needs many minutes (guaranteed timeout noise).
        min_useful_remaining = max(60.0, min(agent_limit * 0.35, 300.0))
        if remaining < min_useful_remaining:
            print(
                f"[harness] skip further polls (~{int(remaining)}s left in window; "
                f"need ~{int(min_useful_remaining)}s for a useful agent run)",
                flush=True,
            )
            break
        # Subprocess cannot outlive the detection window (else pointless second poll with 0s left)
        sub_timeout = int(min(agent_limit, max(5, remaining)))
        print(
            f"[harness] invoking agent (~{max(0, int(remaining))}s left in window, "
            f"subprocess timeout {sub_timeout}s)",
            flush=True,
        )
        result = invoke_classifier(
            fault_injection_time,
            script=classifier_script,
            subprocess_timeout=sub_timeout,
        )
        last_result = result
        if result.get("llm_invoked"):
            ever_llm_invoked = True
        if result.get("detected"):
            first_alert_time = datetime.now(timezone.utc).isoformat()
            agent_result = result
            append_ledger({
                "event": "first_alert",
                "run_id": run_id,
                "timestamp": first_alert_time,
                "signals": result.get("signals", {}),
            })
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_interval)

    # Keep fault active until ~fault_duration seconds after injection (then recover)
    if first_alert_time:
        t_inj = datetime.fromisoformat(fault_injection_time.replace("Z", "+00:00"))
        t_alert = datetime.fromisoformat(first_alert_time.replace("Z", "+00:00"))
        since_injection = (t_alert - t_inj).total_seconds()
        time.sleep(max(0, float(fault_duration) - since_injection))

    fault_recovery_time = datetime.now(timezone.utc).isoformat()
    if not skip_fault_injection:
        set_flag(fault_flag, "off")

    append_ledger({
        "event": "fault_recovery",
        "run_id": run_id,
        "timestamp": fault_recovery_time,
    })

    # MTTD = fault_injection → first detection
    # MTTR = fault_injection → remediation suggested (time until agent proposes fix)
    mttd_seconds = None
    mttr_seconds = None
    remediation_time = None
    if first_alert_time:
        t0 = datetime.fromisoformat(fault_injection_time.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(first_alert_time.replace("Z", "+00:00"))
        mttd_seconds = (t1 - t0).total_seconds()
        src_rem = agent_result if agent_result else last_result
        if src_rem.get("suggested_remediations"):
            remediation_time = datetime.now(timezone.utc).isoformat()
            t_rem = datetime.fromisoformat(remediation_time.replace("Z", "+00:00"))
            mttr_seconds = (t_rem - t0).total_seconds()
    src = agent_result if agent_result else last_result
    run = HarnessRun(
        run_id=run_id,
        fault_flag=fault_flag,
        fault_variant=fault_variant,
        fault_injection_time=fault_injection_time,
        first_alert_time=first_alert_time,
        remediation_suggested_time=remediation_time,
        fault_recovery_time=fault_recovery_time,
        mttd_seconds=mttd_seconds,
        mttr_seconds=mttr_seconds,
        detected=bool(first_alert_time),
        suggested_remediations=src.get("suggested_remediations", []),
        suggested_root_cause=src.get("suggested_root_cause"),
        approach=approach,
        llm_invoked=bool(ever_llm_invoked or src.get("llm_invoked", False)),
        agent_error=src.get("agent_error"),
        agent_output=src.get("agent_output_raw"),
    )

    _log_run_to_mlflow(run, log_mlflow, mlflow_run=mlflow_run, skip_fault_injection=skip_fault_injection)
    if mlflow_run:
        try:
            del os.environ["MLFLOW_RUN_ID"]
        except KeyError:
            pass
    return run


def main() -> int:
    ap = argparse.ArgumentParser(description="MTTD/MTTR Harness")
    ap.add_argument("--flag", default="cartFailure", help="Fault flag to inject")
    ap.add_argument("--variant", default="on", help="Flag variant (on, off, 10%%, etc.)")
    ap.add_argument("--poll-interval", type=int, default=15, help="Agent poll interval (seconds)")
    ap.add_argument(
        "--detection-timeout",
        type=int,
        default=1500,
        help="Wall-clock seconds to wait for detection (must exceed agent subprocess timeout; default 1500)",
    )
    ap.add_argument("--fault-duration", type=int, default=90, help="Total fault duration before recovery")
    ap.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging (default: enabled — runs are logged to the experiment). Use only for offline/debug without a tracking server.",
    )
    ap.add_argument("--output", "-o", help="Write run JSON to file")
    ap.add_argument("--all-agents", action="store_true", default=True,
                    help="Run all LLM agents (ollama_qwen2.5, maas_qwen3-14b) — default")
    ap.add_argument("--single-agent", action="store_true",
                    help="Run only the first agent (use with --classifier to override)")
    ap.add_argument("--classifier", help="Path to classifier script (overrides --all-agents)")
    ap.add_argument("--skip-fault-injection", action="store_true",
                    help="Skip flagd/OpenShift set_flag; use seeded ClickHouse data (for local demo)")
    args = ap.parse_args()

    if not args.no_mlflow:
        _print_mlflow_tracking_banner(MLFLOW_TRACKING_URI, MLFLOW_INSECURE_TLS)

    # Resolve agents to run: --classifier overrides and runs single; else --all-agents runs all
    if args.classifier:
        script = Path(args.classifier)
        name = script.parent.name if script.parent else "custom"
        agents_to_run = [(name, script)]
    elif args.single_agent:
        agents_to_run = [AGENTS[0]]  # default to first agent (ollama_qwen2.5)
    else:
        agents_to_run = AGENTS

    if not args.no_mlflow:
        try:
            import mlflow

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            if args.classifier:
                mode = "--classifier (single custom script)"
            elif args.single_agent:
                mode = "--single-agent"
            else:
                mode = "--all-agents (default)"
            inv = _format_harness_invocation_note(
                agents_to_run=agents_to_run,
                flag=args.flag,
                variant=args.variant,
                poll_interval=args.poll_interval,
                detection_timeout=args.detection_timeout,
                fault_duration=args.fault_duration,
                skip_fault_injection=args.skip_fault_injection,
                selection_mode=mode,
            )
            _set_mlflow_experiment_description(invocation_note=inv)
        except ImportError:
            pass
        except Exception as e:
            print(f"[harness] MLflow experiment description at startup: {e}", flush=True)

    runs: list[HarnessRun] = []
    for approach, classifier_script in agents_to_run:
        print(f"\n[harness] Running agent: {approach}")
        run = run_single_experiment(
            fault_flag=args.flag,
            fault_variant=args.variant,
            poll_interval=args.poll_interval,
            detection_timeout=args.detection_timeout,
            fault_duration=args.fault_duration,
            log_mlflow=not args.no_mlflow,
            classifier_script=classifier_script,
            approach=approach,
            skip_fault_injection=args.skip_fault_injection,
        )
        runs.append(run)
        print(json.dumps(asdict(run), indent=2))

    # Output last run (or aggregate if multiple)
    out = asdict(runs[-1]) if runs else {}
    if len(runs) > 1:
        out = {"runs": [asdict(r) for r in runs], "summary": {"count": len(runs), "agents": [r.approach for r in runs]}}

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWritten to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
