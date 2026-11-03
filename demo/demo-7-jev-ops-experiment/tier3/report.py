"""Scorecard in the LF Edge group's terms, export in their per-run field names, and
a reasoning sheet for hand scoring (design D12). Every figure is computed from run
records. Counts are shown with their denominators, never as bare percentages.

Field names follow HarnessRun in lfedgeai/AIOps
research/agentic_aiops_architectures/code/harness/run_harness.py (read at a408708):
fault_flag is the fault type, fault_variant the target deployment.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from tier3.evaluate import PASS, UNKNOWN

NOT_MEASURED = "not measured (recommendation only)"


def load_records(path: str | Path, include_replay: bool = False) -> list[dict]:
    """Every run record under `path` (a run directory or a parent of several). Replay
    records (`mode: replay`) are left out unless asked for: live scorecards never mix
    them in by accident."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    out = []
    for f in files:
        if f.name == "summary.json" or f.name.startswith("."):
            continue
        rec = json.loads(f.read_text())
        if isinstance(rec, dict) and "experiment_id" in rec and "polls" in rec:
            if rec.get("mode") == "replay" and not include_replay:
                continue
            out.append(rec)
    return out


def _detected(rec: dict) -> bool:
    """The engine reported an incident at some poll (the group's `detected`)."""
    return any(p["decision"]["status"] == "ok" and p["decision"]["incident_detected"]
               for p in rec.get("polls", []))


def _count(rows: list[dict], dim: str) -> tuple[int, int]:
    decided = [r for r in rows if r["evaluation"][dim] != UNKNOWN]
    return sum(1 for r in decided if r["evaluation"][dim] == PASS), len(decided)


def _sum_or_none(vals):
    vals = list(vals)
    return sum(vals) if vals and all(isinstance(v, (int, float)) for v in vals) else None


def _block(recs: list[dict]) -> dict:
    ok = [r for r in recs if r["status"] == "ok"]
    faults = [r for r in ok if r["ground_truth"]["incident"]]
    controls = [r for r in ok if not r["ground_truth"]["incident"]]
    mttd = sorted(r["timing"]["mttd_seconds"] for r in faults if r["timing"].get("mttd_seconds") is not None)
    fp = [r["false_positive"] for r in controls if r.get("false_positive")]
    fp_runs = sum(1 for f in fp if f["false_positive_run"])
    det_faults = sum(1 for r in faults if _detected(r))
    missed = len(faults) - det_faults
    polls = [p for r in recs for p in r.get("polls", [])]
    return {
        "runs": len(recs), "runs_ok": len(ok),
        "runs_error": sum(1 for r in recs if r["status"] == "error"),
        "runs_contaminated": sum(1 for r in recs if r["status"] == "contaminated"),
        "detect": (det_faults, len(faults)),
        "rca": _count(faults, "localization"),
        "diagnosis": _count(faults, "diagnosis"),
        "remediation_correct": _count(faults, "action"),
        "control_correct": _count(controls, "detection"),
        "mttd": {"n": len(mttd), "median": statistics.median(mttd) if mttd else None,
                 "min": mttd[0] if mttd else None, "max": mttd[-1] if mttd else None},
        "false_positive_runs": (fp_runs, len(fp)),
        "false_positive_polls": (sum(f["false_positive_polls"] for f in fp),
                                 sum(f["answered_polls"] for f in fp)),
        "detection_precision": (det_faults, det_faults + fp_runs),
        # Run-level binary detection: a fault run is positive, a no-incident run negative.
        # F1 = 2TP / (2TP + FP + FN), kept as a fraction like every other count here.
        "detection_recall": (det_faults, len(faults)),
        "detection_f1": (2 * det_faults, 2 * det_faults + fp_runs + missed),
        "detection_confusion": {"tp": det_faults, "fn": missed, "fp": fp_runs,
                                "tn": len(fp) - fp_runs},
        "agent_errors": (sum(1 for p in polls if p["decision"]["status"] == "error"), len(polls)),
        "invalid_replies": (sum(1 for p in polls if p["decision"]["status"] == "invalid"), len(polls)),
        "engine_calls": sum(r["usage"]["engine_calls"] for r in recs),
        "input_tokens": _sum_or_none(r["usage"]["input_tokens"] for r in recs),
        "output_tokens": _sum_or_none(r["usage"]["output_tokens"] for r in recs),
        "cost_usd": _sum_or_none(r["usage"]["cost_usd"] for r in recs),
    }


def scorecard_data(records: list[dict]) -> dict:
    by: dict[str, list[dict]] = {}
    for r in records:
        by.setdefault(r["experiment_id"], []).append(r)
    return {"overall": _block(records) if records else {},
            "by_scenario": {k: _block(v) for k, v in sorted(by.items())}}


def _frac(pair) -> str:
    a, b = pair
    return f"{a}/{b}" if b else "n/a"


def _dec(pair) -> str:
    a, b = pair
    return f" ({a / b:.2f})" if b else ""


def _num(x, unit="") -> str:
    return "n/a" if x is None else (f"{x:g}{unit}" if isinstance(x, (int, float)) else str(x))


def format_scorecard(records: list[dict]) -> str:
    data = scorecard_data(records)
    if not records:
        return "no run records found"
    cols = ["scenario", "runs", "detect", "RCA", "diagnosis", "remediation correct",
            "MTTD median (min-max) s", "FP runs", "FP polls", "agent errors"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]

    def row(name, b):
        m = b["mttd"]
        mttd = "n/a" if m["median"] is None else f"{m['median']:g} ({m['min']:g}-{m['max']:g}, n={m['n']})"
        return (f"| {name} | {b['runs_ok']}/{b['runs']} | {_frac(b['detect'])} | {_frac(b['rca'])} | "
                f"{_frac(b['diagnosis'])} | {_frac(b['remediation_correct'])} | {mttd} | "
                f"{_frac(b['false_positive_runs'])} | {_frac(b['false_positive_polls'])} | "
                f"{_frac(b['agent_errors'])} |")
    for name, b in data["by_scenario"].items():
        lines.append(row(name, b))
    o = data["overall"]
    lines.append(row("**all**", o))
    lines += [
        "",
        f"Detection precision (detected fault runs / (detected fault runs + false-positive control runs)): "
        f"{_frac(o['detection_precision'])}",
        f"Detection recall (detected fault runs / fault runs): {_frac(o['detection_recall'])}",
        f"Detection F1 (2TP / (2TP + FP + FN)): {_frac(o['detection_f1'])}{_dec(o['detection_f1'])}",
        "Detection confusion (runs): TP {tp}, FN {fn}, FP {fp}, TN {tn}.".format(**o["detection_confusion"]),
        f"Invalid replies: {_frac(o['invalid_replies'])} polls. Runs with errors: {o['runs_error']}, "
        f"contaminated: {o['runs_contaminated']}.",
        engines_line(records),
        f"Engine calls: {o['engine_calls']}. Tokens in/out: {_num(o['input_tokens'])}/"
        f"{_num(o['output_tokens'])}. Cost: {_num(o['cost_usd'], ' USD')}.",
        f"Remediation executed: {NOT_MEASURED}.",
        f"MTTR: {NOT_MEASURED}.",
        f"Recovery verified: {NOT_MEASURED}.",
        "",
        "detect, RCA, diagnosis and remediation correct count fault runs only; FP counts control runs. "
        "RCA is the localization verdict on the scored poll (first detecting poll, else the last).",
    ]
    return "\n".join(lines)


def _bool(verdict: str):
    return None if verdict == UNKNOWN else verdict == PASS


def engine_name(r: dict) -> str:
    """Records from before engines were recorded are all Jev's."""
    return (r.get("engine") or {}).get("name") or "jev"


def engines_line(records: list[dict]) -> str:
    seen = sorted({(engine_name(r), (r.get("engine") or {}).get("transport") or "jev-api")
                   for r in records})
    return "Engine: " + ", ".join(f"{n} ({t})" for n, t in seen) + "."


def export_lfedge(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        j, gt, f = r.get("jev") or {}, r["ground_truth"], r.get("fault") or {}
        model = None
        for p in r.get("polls", []):
            model = model or (p["decision"].get("telemetry") or {}).get("model")
        out.append({
            "run_id": f"{r['run_id']}/{r['experiment_id']}",
            "fault_flag": f.get("type") or "none",
            "fault_variant": f.get("target") or "none",
            "fault_injected": r.get("fault") is not None,
            "fault_injection_time": r["timing"].get("t_inject"),
            "first_alert_time": r["timing"].get("t_detect"),
            "mttd_seconds": r["timing"].get("mttd_seconds"),
            "detected": _detected(r),
            "suggested_root_cause": j.get("affected_component"),
            "expected_root_cause": gt["component"],
            "rca_correct": _bool(r["evaluation"]["localization"]),
            "suggested_remediations": [j["recommended_action"]] if j.get("recommended_action") else [],
            "remediation_correct": _bool(r["evaluation"]["action"]),
            "remediation_declared": bool(j.get("recommended_action")),
            "remediation_executed": None,
            "mttr_seconds": None,
            "mttr_verified_seconds": None,
            "recovery_verified": None,
            "approach": f"{engine_name(r)}:{model}" if model else engine_name(r),
            "llm_invoked": r["usage"]["answered_calls"] > 0,
            "agent_error": r["message"] or None if r["status"] == "error" else None,
            # ours, alongside theirs
            "jev_experiment_id": r["experiment_id"],
            "jev_status": r["status"],
            "jev_diagnosis": j.get("diagnosis"),
            "jev_diagnosis_correct": _bool(r["evaluation"]["diagnosis"]),
            "jev_detection_verdict": r["evaluation"]["detection"],
            "jev_engine_calls": r["usage"]["engine_calls"],
            "jev_false_positive": r.get("false_positive"),
        })
    return out


def _obs_summary(req: dict | None) -> str:
    if not req:
        return "no observation"
    bits = []
    for d in req.get("deployments", []):
        if d["available"] is not None and d["baseline_available"] is not None \
                and d["available"] < d["baseline_available"]:
            bits.append(f"{d['deployment']} {d['available']}/{d['baseline_available']} available")
    if req.get("pods"):
        bits.append(f"{len(req['pods'])} unhealthy or new pods")
    reasons = sorted({e["reason"] for e in req.get("events", []) if e.get("reason")})
    if reasons:
        bits.append("events: " + ", ".join(reasons[:4]))
    worst = [s for s in req.get("services", []) if (s.get("error_rate_change") or 0) > 0.02][:3]
    if worst:
        bits.append("errors up: " + ", ".join(s["service"] for s in worst))
    return "; ".join(bits) or "no change visible"


def export_reasoning(records: list[dict]) -> str:
    lines = ["# Tier 3 reasoning sheet",
             "",
             "Hand scoring only. These scores are never folded into the four verdicts.",
             "",
             "| run | experiment | poll | observation summary | reasoning | ground truth | hand score | notes |",
             "|---|---|---|---|---|---|---|---|"]
    for r in records:
        j = r.get("jev") or {}
        reason = (j.get("reasoning") or "no reasoning returned").replace("|", "\\|").replace("\n", " ")
        gt = r["ground_truth"]
        lines.append(f"| {r['run_id']} | {r['experiment_id']} | {r.get('scored_poll', 'n/a')} | "
                     f"{_obs_summary(r.get('request'))} | {reason} | "
                     f"{gt['component']} / {gt['diagnosis']} | | |")
    return "\n".join(lines) + "\n"
