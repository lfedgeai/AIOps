#!/usr/bin/env python3
"""
MTTD/MTTR Harness for Agentic AIOps comparison.

Orchestrates chaos injection (via flagd), invokes the baseline classifier
(or another approach), records events, computes MTTD and MTTR, logs to MLflow.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Baseline: rule-based single-shot classifier (no LLM, no tools)
CLASSIFIER_SCRIPT = ROOT / "code" / "agents" / "baseline_classifier" / "classifier.py"

# Env-based config
FLAGD_READ_URL = os.environ.get("FLAGD_READ_URL", "http://localhost:8080/feature/api/read")
FLAGD_WRITE_URL = os.environ.get("FLAGD_WRITE_URL", "http://localhost:8080/feature/api/write")
CLICKHOUSE_HTTP = os.environ.get("CLICKHOUSE_HTTP", "http://localhost:8123")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EVENT_LEDGER_PATH = os.environ.get("EVENT_LEDGER_PATH", str(ROOT / "data" / "event_ledger.jsonl"))


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


def set_flag(flag_name: str, variant: str) -> bool:
    """Set feature flag via flagd-ui HTTP API."""
    return _set_flag_http(flag_name, variant)


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


def invoke_classifier(since_ts: str, script: Path | None = None) -> dict:
    """Invoke classifier (baseline or other) and return result dict."""
    script = script or CLASSIFIER_SCRIPT
    env = os.environ.copy()
    env["CLICKHOUSE_HTTP"] = CLICKHOUSE_HTTP
    try:
        r = subprocess.run(
            [sys.executable, str(script), "--since", since_ts, "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            cwd=str(ROOT),
        )
        if r.returncode in (0, 1) and r.stdout:
            return json.loads(r.stdout)
    except Exception as e:
        return {"detected": False, "error": str(e), "suggested_remediations": []}
    return {"detected": False, "suggested_remediations": []}


def append_ledger(entry: dict) -> None:
    """Append event to ledger file."""
    Path(EVENT_LEDGER_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(EVENT_LEDGER_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_single_experiment(
    fault_flag: str,
    fault_variant: str,
    poll_interval: int,
    detection_timeout: int,
    fault_duration: int,
    log_mlflow: bool,
    classifier_script: Path | None = None,
) -> HarnessRun:
    """Run one fault injection experiment; return HarnessRun."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    fault_injection_time = datetime.now(timezone.utc).isoformat()

    append_ledger({
        "event": "fault_injection",
        "run_id": run_id,
        "flag": fault_flag,
        "variant": fault_variant,
        "timestamp": fault_injection_time,
    })

    if not set_flag(fault_flag, fault_variant):
        return HarnessRun(
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
        )

    first_alert_time = None
    agent_result = {}
    elapsed = 0

    while elapsed < detection_timeout:
        result = invoke_classifier(fault_injection_time, script=classifier_script)
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
        time.sleep(poll_interval)
        elapsed += poll_interval

    # Let fault run a bit after detection (simulate remediation window)
    if first_alert_time:
        time.sleep(max(0, fault_duration - elapsed))

    fault_recovery_time = datetime.now(timezone.utc).isoformat()
    set_flag(fault_flag, "off")

    append_ledger({
        "event": "fault_recovery",
        "run_id": run_id,
        "timestamp": fault_recovery_time,
    })

    # Compute MTTD and MTTR
    mttd_seconds = None
    mttr_seconds = None
    if first_alert_time:
        t0 = datetime.fromisoformat(fault_injection_time.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(first_alert_time.replace("Z", "+00:00"))
        t2 = datetime.fromisoformat(fault_recovery_time.replace("Z", "+00:00"))
        mttd_seconds = (t1 - t0).total_seconds()
        mttr_seconds = (t2 - t1).total_seconds()

    remediation_time = first_alert_time if agent_result else None
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
        suggested_remediations=agent_result.get("suggested_remediations", []),
        suggested_root_cause=agent_result.get("suggested_root_cause"),
    )

    if log_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("agentic_aiops_mttd_mttr")
            with mlflow.start_run(run_name=f"{run_id}_{fault_flag}_{fault_variant}"):
                mlflow.log_params({
                    "fault_flag": fault_flag,
                    "fault_variant": fault_variant,
                    "approach": "baseline_classifier",
                })
                mlflow.log_metrics({
                    "mttd_seconds": mttd_seconds if mttd_seconds is not None else -1,
                    "mttr_seconds": mttr_seconds if mttr_seconds is not None else -1,
                    "detected": 1.0 if run.detected else 0.0,
                })
                mlflow.log_dict(asdict(run), "harness_run.json")
        except ImportError:
            print("[harness] mlflow not installed; skip logging")
        except Exception as e:
            print(f"[harness] MLflow log error: {e}")

    return run


def main() -> int:
    ap = argparse.ArgumentParser(description="MTTD/MTTR Harness")
    ap.add_argument("--flag", default="cartFailure", help="Fault flag to inject")
    ap.add_argument("--variant", default="on", help="Flag variant (on, off, 10%, etc.)")
    ap.add_argument("--poll-interval", type=int, default=15, help="Agent poll interval (seconds)")
    ap.add_argument("--detection-timeout", type=int, default=120, help="Max seconds to wait for detection")
    ap.add_argument("--fault-duration", type=int, default=90, help="Total fault duration before recovery")
    ap.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    ap.add_argument("--output", "-o", help="Write run JSON to file")
    ap.add_argument("--classifier", help="Path to classifier script (default: baseline_classifier)")
    args = ap.parse_args()

    classifier_script = Path(args.classifier) if args.classifier else None
    run = run_single_experiment(
        fault_flag=args.flag,
        fault_variant=args.variant,
        poll_interval=args.poll_interval,
        detection_timeout=args.detection_timeout,
        fault_duration=args.fault_duration,
        log_mlflow=not args.no_mlflow,
        classifier_script=classifier_script,
    )

    out = asdict(run)
    print(json.dumps(out, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
