"""Tier 2 experiment lifecycle.

verify health -> baseline -> verify workload -> inject -> wait for manifestation ->
collect -> build observation -> engine decides -> store raw response -> evaluate ->
reset (always) -> verify return to baseline -> persist.

All collaborators are injected so the lifecycle is testable offline.
"""
from __future__ import annotations

import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from prompts.loader import engine_prompt
from tier2.engine import Decision
from tier2.evaluate import evaluate, is_false_positive, timing
from tier2.manifest import any_deviation
from tier2.observation import ServiceStats, build_observation
from tier2.scenarios import Scenario

STATUS_OK, STATUS_ERROR, STATUS_CONTAMINATED = "ok", "error", "contaminated"


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Config:
    baseline_wait_s: float = 0.0  # time to let steady load build before the baseline window
    manifest_poll_s: float = 15.0
    manifest_timeout_s: float = 150.0
    recover_wait_s: float = 0.0  # time after reset before checking return to baseline
    recover_timeout_s: float = 300.0
    recover_poll_s: float = 30.0
    scrub_terms: tuple[str, ...] = ()
    versions: dict = field(default_factory=dict)
    git_commit: str | None = None


@dataclass
class Deps:
    engine: object  # AIOpsEngine
    injector: object  # FaultInjector
    workload: object  # Workload
    health: Callable[[], None]  # raises on an unhealthy testbed
    stats: Callable[[float, int], dict[str, ServiceStats]]  # (at, window_s) -> stats
    collect: Callable[..., object]  # (window_end, window_s, baseline_end, baseline_s) -> Telemetry
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep


def run_experiment(scn: Scenario, deps: Deps, cfg: Config, run_id: str,
                   results_dir: Path | None = None) -> dict:
    t = {"started_at": _iso(deps.clock())}
    rec: dict = {
        "run_id": run_id, "experiment_id": scn.experiment_id,
        "fault": {"target": scn.target_service, "type": scn.fault_type},
        "ground_truth": {"incident": scn.expected.incident,
                         "service": scn.expected.affected_service,
                         "diagnosis": scn.expected.diagnosis,
                         "acceptable_actions": list(scn.expected.acceptable_actions)},
        "status": STATUS_OK, "message": "", "versions": cfg.versions, "git_commit": cfg.git_commit,
        "prompt": engine_prompt(deps.engine),
    }
    injected = False
    base_stats: dict[str, ServiceStats] = {}
    try:
        deps.health()
        if scn.workload_users:
            deps.workload.set_users(scn.workload_users)
        deps.workload.require_running()
        leftover = deps.injector.active()
        if leftover:
            raise RuntimeError(f"faults already active before the run: {leftover}")
        deps.sleep(cfg.baseline_wait_s)

        t_inject = deps.clock()
        base_at = t_inject
        base_stats = deps.stats(base_at, scn.baseline_window_s)
        if scn.fault is not None:
            deps.injector.inject(scn.fault)
            injected = True
            t["t_inject"] = _iso(t_inject)

        t_manifest = None
        if scn.fault is not None:
            waited = 0.0
            while waited <= cfg.manifest_timeout_s:
                cur = deps.stats(deps.clock(), min(scn.baseline_window_s, 180))
                if any_deviation(cur, base_stats):
                    t_manifest = deps.clock()
                    break
                deps.sleep(cfg.manifest_poll_s)
                waited += cfg.manifest_poll_s
        t["t_manifest"] = _iso(t_manifest) if t_manifest else None

        remaining = t_inject + scn.duration_s - deps.clock()
        if remaining > 0:
            deps.sleep(remaining)

        window_end = deps.clock()
        telemetry = deps.collect(window_end=window_end, window_s=scn.observation_window_s,
                                 baseline_end=base_at, baseline_s=scn.baseline_window_s)
        observation, trims = build_observation(
            telemetry, (*cfg.scrub_terms, scn.experiment_id))
        rec["trim_notes"] = trims
        t["t_observation"] = _iso(deps.clock())

        decision: Decision = deps.engine.decide(observation)
        t["t_decision"] = _iso(deps.clock())
        rec["jev"] = {"incident_detected": decision.incident_detected,
                      "affected_service": decision.affected_service,
                      "diagnosis": decision.diagnosis,
                      "recommended_action": decision.recommended_action,
                      "confidence": decision.confidence, "reasoning": decision.reasoning,
                      "status": decision.status, "message": decision.message,
                      "raw_response": decision.raw_response, "telemetry": decision.telemetry}
        rec["request"] = observation  # exactly what the engine received (no ground truth)
        # Evaluation happens only after the decision is stored on the record.
        rec["evaluation"] = evaluate(decision, scn.expected)
        rec["false_positive"] = is_false_positive(decision, scn.expected)
        if decision.status == "error":
            rec["status"], rec["message"] = STATUS_ERROR, decision.message
    except Exception as e:  # noqa: BLE001 - recorded, and the reset below still runs
        rec["status"], rec["message"] = STATUS_ERROR, f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc(limit=4)
    finally:
        if injected:
            try:
                deps.injector.reset()
            except Exception as e:  # noqa: BLE001
                rec["status"] = STATUS_CONTAMINATED
                rec["message"] = f"reset failed: {type(e).__name__}: {e}"
        if injected and rec["status"] != STATUS_CONTAMINATED:
            if not _recovered(scn, deps, cfg, base_stats):
                rec["status"] = STATUS_CONTAMINATED
                rec["message"] = "system did not return to baseline after reset"
        t["ended_at"] = _iso(deps.clock())
    rec["timing"] = timing(started_at=t["started_at"], t_inject=t.get("t_inject"),
                           t_manifest=t.get("t_manifest"), t_observation=t.get("t_observation"),
                           t_decision=t.get("t_decision"), ended_at=t["ended_at"])
    if "evaluation" not in rec:
        rec["evaluation"] = {d: "UNKNOWN" for d in ("detection", "localization", "diagnosis", "action")}
        rec["false_positive"] = None
    if results_dir is not None:
        persist(rec, results_dir)
    return rec


def _recovered(scn: Scenario, deps: Deps, cfg: Config, base_stats) -> bool:
    deps.sleep(cfg.recover_wait_s)
    waited = 0.0
    while True:
        cur = deps.stats(deps.clock(), min(scn.baseline_window_s, 120))
        if not any_deviation(cur, base_stats):
            return True
        if waited >= cfg.recover_timeout_s:
            return False
        deps.sleep(cfg.recover_poll_s)
        waited += cfg.recover_poll_s


def persist(rec: dict, results_dir: Path) -> Path:
    d = Path(results_dir) / rec["run_id"]
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob(f"{rec['experiment_id']}*.json")))
    p = d / (f"{rec['experiment_id']}.json" if n == 0 else f"{rec['experiment_id']}-{n + 1}.json")
    p.write_text(json.dumps(rec, indent=2, default=str))
    return p


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def run_batch(scenarios: list[Scenario], deps: Deps, cfg: Config, results_dir: Path,
              repeats: int = 1, run_id: str | None = None) -> dict:
    """Run every scenario `repeats` times. A contaminated run stops the batch, because
    later experiments would start from a system that is not at baseline."""
    from tier2.evaluate import aggregate

    run_id = run_id or new_run_id()
    records, stopped = [], None
    for _ in range(repeats):
        for scn in scenarios:
            r = run_experiment(scn, deps, cfg, run_id, results_dir)
            records.append(r)
            if r["status"] == STATUS_CONTAMINATED:
                stopped = f"{scn.experiment_id}: {r['message']}"
                break
        if stopped:
            break
    summary = {"run_id": run_id, "repeats": repeats, "stopped_early": stopped,
               "prompt": engine_prompt(deps.engine),
               "aggregate": aggregate(records),
               "runs": [{"experiment_id": r["experiment_id"], "status": r["status"],
                         "evaluation": r["evaluation"]} for r in records]}
    d = Path(results_dir) / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
