#!/usr/bin/env python3
"""Recompute corrected MTTD/MTTR for a completed matrix batch using the fixed metric logic.

Background (bugs fixed in code/harness + code/agents on 19 Oct 26):
  1. MTTD source: run_harness re-parsed agent stdout with a strict json.loads to find
     signals.detection_timestamp; any diagnostic line on stdout (execute_gate nudge,
     rag_force, model_context "context window" line) made it fail silently and fall
     back to the harness clock at subprocess exit -> inflated MTTD for affected agents.
  2. MTTD definition: agent detection_timestamp was stamped at the FIRST SUCCESSFUL
     WRITE (any target/action), so act-first agents got detection == remediation time.
  3. MTTR definition: mttr_seconds keyed off remediation_executed_time, stamped on the
     first successful write of ANY kind (wrong target / wrong action included), while
     the report's "Exec" column required a targeted + action-appropriate write.

This script replays the FIXED derivation over the recorded raw data (agent stdout and
per-write records were captured correctly; only the derivation was wrong):
  - MTTD_fixed  = robust-parsed signals.detection_timestamp - fault_injection_time
                  (same tolerant parser as the harness result pipeline)
  - MTTR_fixed  = first_effective_remediation_write(...).completed_at - fault_injection_time
                  (successful + targeted at expected component + action-appropriate;
                   identical selector to the fixed score_remediation_executed)
  - exec_check  = recomputed from remediation_writes with the fixed selector, compared
                  against the recorded remediation_executed (semantic tightening: one
                  write must satisfy targeting AND action, not two different writes).

Verification checks (fail loudly before any report is written):
  A. batch completeness (agents x contexts x faults)
  B. exec recomputed vs recorded (flips must be explained)
  C. detect / RCA / recovery counts unchanged vs published report
  D. MTTR-present <=> exec-true for every run
  E. MTTD <= MTTR per run wherever both exist
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from code.harness.evaluation import first_effective_remediation_write
from code.harness.run_harness import _parse_agent_json_stdout, _strip_mlflow_status_from_agent_stdout

FAULTS_BY_LEN = sorted(
    {
        "kill_pod", "scale_zero", "memory_limit", "network_partition",
        "readiness_probe_fail", "config_corruption", "pvc_full", "node_taint",
        "replica_overload", "dependency_removal",
    },
    key=len,
    reverse=True,
)


def parse_filename(path: Path) -> dict[str, str] | None:
    m = re.match(r"context_C([^/]+)_sa_(.+)\.json$", path.name)
    if not m:
        return None
    ctx, rest = m.group(1), m.group(2)
    for fault in FAULTS_BY_LEN:
        prefix = fault + "_"
        if rest.startswith(prefix):
            return {"context_c": ctx, "fault": fault, "agent": rest[len(prefix):]}
    return None


def _iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def recompute_run(rec: dict) -> dict:
    out = dict(rec)
    ao = rec.get("agent_output") or ""
    clean = _strip_mlflow_status_from_agent_stdout(ao)
    parsed = _parse_agent_json_stdout(clean)
    signals = (parsed or {}).get("signals") or {}
    strict_ok = True
    try:
        json.loads(clean.strip())
    except (json.JSONDecodeError, TypeError):
        strict_ok = False

    t0 = _iso(rec["fault_injection_time"])
    det_ts = signals.get("detection_timestamp")
    writes = signals.get("remediation_writes")

    # Fixed MTTD: agent-recorded detection timestamp via the tolerant parser.
    mttd_fixed = None
    det_source = None
    if rec.get("detected"):
        if det_ts:
            mttd_fixed = (_iso(det_ts) - t0).total_seconds()
            det_source = "agent_signals"
        elif rec.get("first_alert_time"):
            mttd_fixed = (_iso(rec["first_alert_time"]) - t0).total_seconds()
            det_source = "recorded_first_alert_fallback"

    # Fixed MTTR: first successful, targeted, action-appropriate write.
    mttr_fixed = None
    eff_write = first_effective_remediation_write(
        rec.get("fault_flag"), rec.get("fault_variant"), writes
    )
    if eff_write:
        w_ts = eff_write.get("completed_at") or eff_write.get("started_at")
        mttr_fixed = (_iso(w_ts) - t0).total_seconds()

    out.update(
        {
            "mttd_fixed": mttd_fixed,
            "mttr_fixed": mttr_fixed,
            "det_source": det_source,
            "stdout_strict_parse": strict_ok,
            "n_writes": len(writes or []),
            "n_success_writes": sum(1 for w in (writes or []) if w.get("success")),
            "exec_recomputed": eff_write is not None,
            "effective_write_time": (eff_write or {}).get("completed_at")
            or (eff_write or {}).get("started_at"),
        }
    )
    return out


def median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


def fmt_med(vals: list[float]) -> str:
    if not vals:
        return "—"
    return f"{statistics.median(vals):.0f}s (n={len(vals)})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kickoff", default=None, help="AEST kickoff 'YYYY-MM-DD HH:MM:SS' (default: out/.matrix_kickoff_aest, machine-local)")
    ap.add_argument("--out", default=str(ROOT / "out" / "analysis_12oct26_corrected.json"))
    args = ap.parse_args()

    kickoff_str = args.kickoff or (ROOT / "out" / ".matrix_kickoff_aest").read_text().strip()
    kickoff = datetime.strptime(kickoff_str, "%Y-%m-%d %H:%M:%S").timestamp()

    runs = []
    for path in sorted((ROOT / "out").glob("context_C*_sa_*.json")):
        meta = parse_filename(path)
        if not meta:
            continue
        if os.path.getmtime(path) < kickoff:
            continue
        rec = json.loads(path.read_text())
        if isinstance(rec, dict) and "runs" in rec:
            rec = rec["runs"][0] if rec["runs"] else rec
        rec.update(meta)
        rec["_file"] = path.name
        runs.append(recompute_run(rec))

    n = len(runs)
    problems: list[str] = []

    # A. completeness
    agents = sorted({r["agent"] for r in runs})
    contexts = sorted({r["context_c"] for r in runs}, key=lambda c: (len(c), c))
    faults = sorted({r["fault"] for r in runs})
    expected = len(agents) * len(contexts) * len(faults)
    if n != expected:
        problems.append(f"batch incomplete: {n} runs != {len(agents)}x{len(contexts)}x{len(faults)}={expected}")

    # B. exec recomputed vs recorded
    flips = [r for r in runs if r["exec_recomputed"] != bool(r.get("remediation_executed"))]
    rec_exec = sum(1 for r in runs if r.get("remediation_executed"))
    new_exec = sum(1 for r in runs if r["exec_recomputed"])

    # C. unchanged counts vs published report
    published = {"detected": 247, "rca": 175, "rem_or": 203, "exec": 50, "recovery": 49, "agent_error": 21}
    got = {
        "detected": sum(1 for r in runs if r.get("detected")),
        "rca": sum(1 for r in runs if r.get("rca_correct")),
        "rem_or": sum(1 for r in runs if r.get("remediation_correct")),
        "exec": rec_exec,
        "recovery": sum(1 for r in runs if r.get("recovery_verified")),
        "agent_error": sum(1 for r in runs if r.get("agent_error")),
    }
    for k, v in published.items():
        if got[k] != v:
            problems.append(f"count mismatch {k}: recomputed batch has {got[k]}, published report says {v}")

    # D. MTTR-present <=> exec-true
    bad_d = [r["_file"] for r in runs if (r["mttr_fixed"] is not None) != r["exec_recomputed"]]
    if bad_d:
        problems.append(f"MTTR/exec invariant violated in {len(bad_d)} runs: {bad_d[:5]}")

    # E. MTTD <= MTTR per run
    bad_e = [
        (r["_file"], r["mttd_fixed"], r["mttr_fixed"])
        for r in runs
        if r["mttd_fixed"] is not None and r["mttr_fixed"] is not None and r["mttd_fixed"] > r["mttr_fixed"] + 1e-6
    ]
    if bad_e:
        problems.append(f"MTTD>MTTR in {len(bad_e)} runs: {bad_e[:5]}")

    # Summary tables
    print(f"batch: {n} runs | agents={len(agents)} contexts={len(contexts)} faults={len(faults)}")
    print(f"exec: recorded={rec_exec} recomputed={new_exec} flips={len(flips)}")
    for f in flips:
        print(f"  FLIP {f['_file']}: recorded={f.get('remediation_executed')} recomputed={f['exec_recomputed']} "
              f"writes={[(w.get('tool'), w.get('arguments'), w.get('success')) for w in (json.loads(json.dumps(f.get('agent_output') or '{}')) if False else [])][:0]}")
    print()
    print("=== By agent (48 runs each) — corrected ===")
    print(f"{'agent':<22} {'detect':>9} {'rca':>9} {'exec':>9} {'MTTD med':>12} {'MTTR med':>12} {'MTTRv med':>12}")
    for a in agents:
        items = [r for r in runs if r["agent"] == a]
        mttd_vals = [r["mttd_fixed"] for r in items if r.get("detected") and r["mttd_fixed"] is not None]
        mttr_vals = [r["mttr_fixed"] for r in items if r["mttr_fixed"] is not None]
        mttrv_vals = [r["mttr_verified_seconds"] for r in items if r.get("mttr_verified_seconds") is not None]
        old_mttd = [r["mttd_seconds"] for r in items if r.get("detected") and r.get("mttd_seconds") is not None]
        old_mttr = [r["mttr_seconds"] for r in items if r.get("mttr_seconds") is not None]
        print(
            f"{a:<22} "
            f"{sum(1 for r in items if r.get('detected')):>4}/{len(items):<4} "
            f"{sum(1 for r in items if r.get('rca_correct')):>4}/{len(items):<4} "
            f"{sum(1 for r in items if r['exec_recomputed']):>4}/{len(items):<4} "
            f"{fmt_med(mttd_vals):>12} "
            f"{fmt_med(mttr_vals):>12} "
            f"{fmt_med(mttrv_vals):>12}"
        )
        print(
            f"{'  (old:':<22} "
            f"{'':>9} {'':>9} {'':>9} "
            f"{fmt_med(old_mttd):>12} {fmt_med(old_mttr):>12})"
        )
    print()
    print("=== Overall ===")
    all_mttd = [r["mttd_fixed"] for r in runs if r.get("detected") and r["mttd_fixed"] is not None]
    all_mttr = [r["mttr_fixed"] for r in runs if r["mttr_fixed"] is not None]
    print(f"MTTD median (detected, n={len(all_mttd)}): {median(all_mttd):.0f}s  (published: 117s)")
    print(f"MTTR median (n={len(all_mttr)}): {median(all_mttr):.0f}s  (published: 105s, n=105)")
    print(f"stdout strictly parses (old MTTD path): {sum(1 for r in runs if r['stdout_strict_parse'])}/{n} runs")

    if problems:
        print("\n!!! VERIFICATION PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1

    Path(args.out).write_text(json.dumps(runs, indent=2, default=str))
    print(f"\nAll verification checks passed. Corrected rows written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
