"""Tier 4 scorecard and LF Edge export (design D10). Tier 3's columns plus measured
remediation, recovery and MTTR. Every figure is computed from the run records."""
from __future__ import annotations

import statistics

from tier3.report import _block, _frac, export_lfedge as tier3_export, format_scorecard as tier3_scorecard
from tier3.report import load_records  # noqa: F401 - re-exported for the CLI
from tier4.recovery import NOT_RECOVERED, RECOVERED_BY_ACTION, RECOVERED_WITHOUT_ACTION


def _spread(vals: list[float]) -> dict:
    vals = sorted(vals)
    return {"n": len(vals), "median": statistics.median(vals) if vals else None,
            "min": vals[0] if vals else None, "max": vals[-1] if vals else None}


def _rem_block(recs: list[dict]) -> dict:
    ok = [r for r in recs if r["status"] == "ok" and r.get("remediation")]
    faults = [r for r in ok if r["ground_truth"]["incident"]]
    rem = [r["remediation"] for r in faults]
    alarms = [r["remediation"] for r in ok if not r["ground_truth"]["incident"]]
    t = [x["timing"] for x in rem]
    return {
        "fault_runs": len(rem),
        "executed": sum(1 for x in rem if x["remediation_executed"]),
        "not_executable": sum(1 for x in rem if x["detected"] and not x["remediation_executed"]),
        "by_action": sum(1 for x in rem if x["outcome"] == RECOVERED_BY_ACTION),
        "without_action": sum(1 for x in rem if x["outcome"] == RECOVERED_WITHOUT_ACTION),
        "not_recovered": sum(1 for x in rem if x["outcome"] == NOT_RECOVERED),
        "relapsed": sum(1 for x in rem if x.get("relapsed")),
        "mttr": _spread([x["mttr_seconds"] for x in t if x["mttr_seconds"] is not None]),
        "mttr_verified": _spread([x["mttr_verified_seconds"] for x in t
                                  if x["mttr_verified_seconds"] is not None]),
        "self_recovery": _spread([x["self_recovery_seconds"] for x in t
                                  if x["self_recovery_seconds"] is not None]),
        "control_runs": len(alarms),
        "false_alarm_actions": sum(1 for x in alarms if x["action_on_false_alarm"]),
        "disturbed": sum(1 for x in alarms if x["disturbed_healthy_system"]),
    }


def scorecard_data(records: list[dict]) -> dict:
    by: dict[str, list[dict]] = {}
    for r in records:
        by.setdefault(r["experiment_id"], []).append(r)
    return {"overall": {**(_block(records) if records else {}), "remediation": _rem_block(records)},
            "by_scenario": {k: {**_block(v), "remediation": _rem_block(v)} for k, v in sorted(by.items())}}


def _med(s: dict) -> str:
    return "n/a" if s["median"] is None else f"{s['median']:g} ({s['min']:g}-{s['max']:g}, n={s['n']})"


def format_scorecard(records: list[dict]) -> str:
    if not records:
        return "no run records found"
    # Tier 3's table, without its "not measured" lines: Tier 4 measures them below.
    head = [x for x in tier3_scorecard(records).splitlines()
            if not x.startswith(("Remediation executed:", "MTTR:", "Recovery verified:"))]
    data = scorecard_data(records)
    cols = ["scenario", "fault runs", "executed", "not executable", "recovered by action",
            "recovered without action", "not recovered", "relapsed", "MTTR median (min-max) s",
            "verified MTTR median (min-max) s", "self-recovery median s", "false-alarm actions",
            "disturbed healthy"]
    lines = ["", "## Remediation", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]

    def row(name, x):
        f = x["fault_runs"]
        return (f"| {name} | {f} | {x['executed']}/{f} | {x['not_executable']}/{f} | {x['by_action']}/{f} | "
                f"{x['without_action']}/{f} | {x['not_recovered']}/{f} | {x['relapsed']}/{f} | {_med(x['mttr'])} | "
                f"{_med(x['mttr_verified'])} | {_med(x['self_recovery'])} | "
                f"{_frac((x['false_alarm_actions'], x['control_runs']))} | "
                f"{_frac((x['disturbed'], x['control_runs']))} |")
    for name, b in data["by_scenario"].items():
        lines.append(row(name, b["remediation"]))
    lines.append(row("**all**", data["overall"]["remediation"]))
    lines += [
        "",
        "Single-shot recommendation MTTR: one answer, carried out once, no retry. MTTR is fault to remediation written (the LF Edge `mttr_seconds`); verified MTTR is fault to "
        "the start of the final run of passing service-level checks (at least 60 s, lasting to the end "
        "of the 360 s window), for recovered-by-action runs only (`mttr_verified_seconds`). Relapsed: "
        "a run of passing checks met the hold and later failed. Fault-run columns count incident scenarios; false-alarm columns "
        "count the no-incident scenarios and the control.",
    ]
    return "\n".join(head + lines)


def export_lfedge(records: list[dict]) -> list[dict]:
    out = []
    for r, row in zip(records, tier3_export(records)):
        rem = r.get("remediation")
        if rem:
            t = rem["timing"]
            row.update({
                "remediation_executed": rem["remediation_executed"],
                "mttr_seconds": t["mttr_seconds"],
                "mttr_verified_seconds": t["mttr_verified_seconds"],
                "recovery_verified": rem["recovery_verified"],
                "jev_recovery_outcome": rem["outcome"],
                "jev_action_not_executable": rem["action"].get("not_executable"),
                "jev_self_recovery_seconds": t["self_recovery_seconds"],
                "jev_action_on_false_alarm": rem["action_on_false_alarm"],
            })
        out.append(row)
    return out
