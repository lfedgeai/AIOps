"""Tier 3 experiment lifecycle.

health -> workload -> no leftovers -> baseline (stats, cluster state, placement) ->
arm dead-man timer -> save + inject + read back -> detect (poll or single) ->
store every decision -> evaluate the scored one -> reset (always) -> disarm ->
verify cluster recovery -> persist.

All collaborators are injected so the lifecycle is testable offline.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from prompts.loader import engine_prompt
from tier2.runner import new_run_id, persist  # noqa: F401 - re-exported for the CLI
from tier3.engine import STATUS_ERROR as DECISION_ERROR
from tier3.engine import STATUS_OK as DECISION_OK
from tier3.engine import Decision
from tier3.evaluate import DIMENSIONS, UNKNOWN, evaluate, false_positive_polls, seconds_between
from tier3.layouts import DEFAULT_LAYOUT, apply_layout, get_layout
from tier3.manifest import manifestation, recovery_problems
from tier3.observation import build_observation
from tier3.scenarios import Scenario

STATUS_OK, STATUS_ERROR, STATUS_CONTAMINATED = "ok", "error", "contaminated"
MODES = ("poll", "single")


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Config:
    mode: str = "poll"
    telemetry_lag_s: float = 15.0  # as the LF Edge harness: wait before the first poll
    baseline_wait_s: float = 0.0  # time to let steady load build before the baseline window
    manifest_poll_s: float = 15.0  # single mode only; poll mode checks at every poll
    manifest_timeout_s: float = 150.0
    recover_wait_s: float = 0.0
    recover_timeout_s: float = 300.0
    recover_poll_s: float = 30.0
    deadman_margin_s: int = 120
    stop_at_first_detection: bool = False  # Tier 4: no-incident scenarios stop at detection too
    extra_hold_s: float = 0.0  # Tier 4: action and recovery time added to the dead-man hold
    scrub_terms: tuple[str, ...] = ()
    layout: str = DEFAULT_LAYOUT  # observation layout applied after build_observation (tier3/layouts.py)
    versions: dict = field(default_factory=dict)
    git_commit: str | None = None


@dataclass
class Deps:
    engine: object  # AIOpsEngine
    injector: object  # tier3.faults.FaultInjector
    workload: object  # tier2.workload.Workload
    health: Callable[[], None]  # raises on an unhealthy testbed
    stats: Callable[[float, int], dict]  # (at, window_s) -> {service: ServiceStats}
    kube_state: Callable[..., object]  # (start_s=None, end_s=None) -> KubeState
    collect: Callable[..., tuple]  # (window_end, window_s, baseline_end, baseline_s) -> (Telemetry, KubeState)
    deadman: object | None = None  # tier3.deadman.Deadman, or None to skip (tests)
    between_runs: Callable[[], None] | None = None  # batch upkeep after each run (rebalance)
    # Tier 4 hooks; None keeps Tier 3 recommendation-only. after_scored(rec, decision, ctx)
    # runs after evaluation; on_reset() runs first in `finally` and returns what it restored.
    after_scored: Callable[[dict, Decision, dict], None] | None = None
    on_reset: Callable[[], list[str]] | None = None
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep


def engine_info(engine) -> dict:
    """Which engine answered. An engine without .info is recorded by its class name."""
    info = getattr(engine, "info", None)
    return dict(info) if isinstance(info, dict) else {"name": type(engine).__name__}


def _decision_dict(d: Decision) -> dict:
    return {"status": d.status, "incident_detected": d.incident_detected,
            "affected_component": d.affected_component, "diagnosis": d.diagnosis,
            "recommended_action": d.recommended_action, "confidence": d.confidence,
            "reasoning": d.reasoning, "message": d.message, "raw_response": d.raw_response,
            "telemetry": d.telemetry}


def _usage(polls: list[dict]) -> dict:
    """Calls, tokens and cost over every poll. Cost only when every answered call reported one.
    A run that never called the engine (it stopped before polling) used nothing: 0, not None."""
    tele = [p["decision"]["telemetry"] for p in polls]
    answered = [t for p, t in zip(polls, tele) if p["decision"]["status"] != DECISION_ERROR]

    def total(key):
        if not polls:
            return 0
        vals = [t.get(key) for t in answered]
        return sum(vals) if vals and all(isinstance(v, (int, float)) for v in vals) else None
    return {"engine_calls": len(polls), "answered_calls": len(answered),
            "input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"),
            "cost_usd": total("cost_usd")}


def run_experiment(scn: Scenario, deps: Deps, cfg: Config, run_id: str,
                   results_dir: Path | None = None) -> dict:
    if cfg.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    t: dict = {"started_at": _iso(deps.clock()), "t_inject": None, "t_manifest": None,
               "t_detect": None}
    rec: dict = {
        "run_id": run_id, "experiment_id": scn.experiment_id, "mode": cfg.mode,
        "fault": {"type": scn.fault.type, "target": scn.fault.target} if scn.fault else None,
        "ground_truth": {"incident": scn.expected.incident,
                         "component": scn.expected.affected_component,
                         "diagnosis": scn.expected.diagnosis,
                         "acceptable_actions": list(scn.expected.acceptable_actions)},
        "status": STATUS_OK, "message": "", "versions": cfg.versions, "git_commit": cfg.git_commit,
        "engine": engine_info(deps.engine),
        "prompt": engine_prompt(deps.engine),
        "observation": get_layout(cfg.layout).identity,
        "polls": [],
    }
    injected, timer = False, None
    base_stats: dict = {}
    base_kube = None
    polls: list[dict] = rec["polls"]
    try:
        deps.health()
        if scn.workload_users:
            deps.workload.set_users(scn.workload_users)
        deps.workload.require_running()
        leftover = deps.injector.leftovers()
        if leftover:
            raise RuntimeError(f"harness objects left from an earlier run: {leftover}; "
                               "run python -m tier3 reset-faults")
        deps.sleep(cfg.baseline_wait_s)

        t0 = deps.clock()
        base_stats = deps.stats(t0, scn.baseline_window_s)
        base_kube = deps.kube_state()
        rec["placement"] = {p["name"]: p["node"] for p in base_kube.pods}
        rec["baseline_deployments"] = base_kube.deployments

        if scn.fault is not None:
            if deps.deadman is not None:
                hold = max(scn.duration_s, scn.detection_window_s) + cfg.telemetry_lag_s + cfg.extra_hold_s
                timer = deps.deadman.arm(int(hold + cfg.deadman_margin_s))
                rec["deadman_unit"] = timer
            injected = True  # set before inject: a partial inject must still be reset
            t_inject = deps.clock()
            t["t_inject"] = _iso(t_inject)
            rec["saved_state"] = deps.injector.inject(scn.fault)
        else:
            t_inject = t0  # reference point for the polling schedule only

        def poll() -> tuple[Decision, dict]:
            now = deps.clock()
            telemetry, kstate = deps.collect(window_end=now, window_s=scn.observation_window_s,
                                             baseline_end=t0, baseline_s=scn.baseline_window_s)
            why = manifestation(telemetry.services, base_stats, kstate, base_kube) \
                if scn.fault is not None else []
            if why and t["t_manifest"] is None:
                t["t_manifest"] = _iso(now)
            obs, trims = build_observation(telemetry, kstate, base_kube,
                                           (*cfg.scrub_terms, scn.experiment_id))
            obs, layout_notes = apply_layout(cfg.layout, obs)
            trims = {**trims, **layout_notes}
            try:
                d: Decision = deps.engine.decide(obs)
            except Exception as e:  # noqa: BLE001 - recorded on the poll; polling goes on
                d = Decision(status=DECISION_ERROR, message=f"{type(e).__name__}: {e}")
            t_dec = deps.clock()
            p = {"n": len(polls) + 1, "t_observation": _iso(now), "t_decision": _iso(t_dec),
                 "seconds_since_inject": round(t_dec - t_inject, 3) if scn.fault else None,
                 "request": obs,  # exactly what the engine received (no ground truth)
                 "request_bytes": len(json.dumps(obs).encode()), "trim_notes": trims,
                 "manifest": why, "decision": _decision_dict(d)}
            polls.append(p)
            return d, p

        scored: tuple[Decision, dict] | None = None
        if cfg.mode == "single":
            if scn.fault is not None:
                waited = 0.0
                while waited <= cfg.manifest_timeout_s and t["t_manifest"] is None:
                    now = deps.clock()
                    if manifestation(deps.stats(now, min(scn.baseline_window_s, 180)), base_stats,
                                     deps.kube_state(), base_kube):
                        t["t_manifest"] = _iso(now)
                        break
                    deps.sleep(cfg.manifest_poll_s)
                    waited += cfg.manifest_poll_s
            remaining = t_inject + scn.duration_s - deps.clock()
            if remaining > 0:
                deps.sleep(remaining)
            scored = poll()
        else:
            deps.sleep(cfg.telemetry_lag_s)
            deadline = t_inject + scn.detection_window_s
            while True:
                start = deps.clock()
                d, p = poll()
                if d.status == DECISION_OK and d.incident_detected and scored is None:
                    scored = (d, p)
                    t["t_detect"] = p["t_decision"]
                    if scn.expected.incident or cfg.stop_at_first_detection:
                        break  # fault runs stop at the first detection
                nxt = start + scn.poll_interval_s
                if nxt > deadline:
                    break
                wait = nxt - deps.clock()
                if wait > 0:
                    deps.sleep(wait)
            if scored is None:
                scored = (d, p)  # nothing detected: the last poll is scored

        decision, sp = scored
        rec["scored_poll"] = sp["n"]
        rec["jev"] = sp["decision"]
        rec["request"] = sp["request"]
        # Evaluation happens only after every decision is stored on the record.
        rec["evaluation"] = evaluate(decision, scn.expected)
        rec["false_positive"] = false_positive_polls(polls, scn.expected)
        if all(p["decision"]["status"] == DECISION_ERROR for p in polls):
            rec["status"], rec["message"] = STATUS_ERROR, polls[-1]["decision"]["message"]
        if deps.after_scored is not None:
            deps.after_scored(rec, decision, {"t0": t0, "t_inject": t_inject if scn.fault else None,
                                              "t_detect": t["t_detect"], "base_kube": base_kube,
                                              "base_stats": base_stats, "scenario": scn})
    except Exception as e:  # noqa: BLE001 - recorded, and the reset below still runs
        rec["status"], rec["message"] = STATUS_ERROR, f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc(limit=4)
    finally:
        reset_ok = True
        extra_restored = False
        if deps.on_reset is not None:
            try:
                rec["restored_action"] = deps.on_reset()
                extra_restored = bool(rec["restored_action"])
            except Exception as e:  # noqa: BLE001
                reset_ok = False
                rec["status"] = STATUS_CONTAMINATED
                rec["message"] = f"action reset failed: {type(e).__name__}: {e}"
        if injected:
            try:
                rec["restored"] = deps.injector.reset()
            except Exception as e:  # noqa: BLE001
                reset_ok = False
                rec["status"] = STATUS_CONTAMINATED
                rec["message"] = f"reset failed: {type(e).__name__}: {e}"
        if timer is not None and reset_ok:  # a failed reset leaves the timer to try again
            try:
                deps.deadman.disarm(timer)
            except Exception as e:  # noqa: BLE001
                rec["deadman_disarm_error"] = f"{type(e).__name__}: {e}"
        if (injected or extra_restored) and reset_ok and base_kube is not None:
            problems = _recover(scn, deps, cfg, base_kube, base_stats)
            if problems:
                rec["status"] = STATUS_CONTAMINATED
                rec["message"] = "cluster did not return to baseline after reset: " + "; ".join(problems)
                rec["recovery_problems"] = problems
        t["ended_at"] = _iso(deps.clock())

    rec["timing"] = {**t,
                     "mttd_seconds": seconds_between(t["t_inject"], t["t_detect"]),
                     "detect_after_manifest_seconds": seconds_between(t["t_manifest"], t["t_detect"]),
                     "total_seconds": seconds_between(t["started_at"], t["ended_at"])}
    rec["usage"] = _usage(polls)
    if "evaluation" not in rec:
        rec["evaluation"] = {d: UNKNOWN for d in DIMENSIONS}
        rec["false_positive"] = None
    if results_dir is not None:
        persist(rec, results_dir)
    return rec


def _recover(scn: Scenario, deps: Deps, cfg: Config, base_kube, base_stats) -> list[str]:
    deps.sleep(cfg.recover_wait_s)
    waited = 0.0
    while True:
        problems = recovery_problems(deps.kube_state(), base_kube, deps.injector.leftovers(),
                                     deps.stats(deps.clock(), min(scn.baseline_window_s, 120)),
                                     base_stats)
        if not problems or waited >= cfg.recover_timeout_s:
            return problems
        deps.sleep(cfg.recover_poll_s)
        waited += cfg.recover_poll_s


def run_batch(scenarios: list[Scenario], deps: Deps, cfg: Config, results_dir: Path,
              repeats: int = 1, run_id: str | None = None) -> dict:
    """Run every scenario `repeats` times. A contaminated run stops the batch, because
    later experiments would start from a cluster that is not at baseline."""
    from tier3.report import scorecard_data

    run_id = run_id or new_run_id()
    records, stopped = [], None
    for _ in range(repeats):
        for scn in scenarios:
            r = run_experiment(scn, deps, cfg, run_id, results_dir)
            records.append(r)
            if r["status"] == STATUS_CONTAMINATED:
                stopped = f"{scn.experiment_id}: {r['message']}"
                break
            if deps.between_runs:
                deps.between_runs()
        if stopped:
            break
    summary = {"run_id": run_id, "mode": cfg.mode, "repeats": repeats, "stopped_early": stopped,
               "prompt": engine_prompt(deps.engine),
               "observation": get_layout(cfg.layout).identity,
               "scorecard": scorecard_data(records),
               "runs": [{"experiment_id": r["experiment_id"], "status": r["status"],
                         "evaluation": r["evaluation"],
                         "mttd_seconds": r["timing"]["mttd_seconds"]} for r in records]}
    d = Path(results_dir) / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
