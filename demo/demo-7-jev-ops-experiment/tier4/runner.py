"""Tier 4 lifecycle (design D1): Tier 3's run up to the first detecting poll, Tier 3's
verdicts on that poll, then one allowlisted action and a recovery watch, then reset of
the action and the fault in `finally`. Only this module and tier4/cli.py import the
executor.

The engine sees nothing new: the action and recovery run after the last poll, and their
results go to the record only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from prompts.loader import engine_prompt
from tier3.engine import STATUS_OK as DECISION_OK
from tier3.evaluate import seconds_between
from tier3.runner import STATUS_CONTAMINATED, Config, Deps, new_run_id
from tier3.runner import run_experiment as tier3_run_experiment
from tier3.scenarios import Scenario
from tier4.executor import ROLLOUT_TIMEOUT_S, Executor
from tier4.recovery import RECOVERED_BY_ACTION, RECOVERED_WITHOUT_ACTION, RecoveryConfig
from tier4.recovery import check, disturbed, outcome, watch

DEFAULT_RESULTS = Path("results") / "tier4"


@dataclass
class Tier4Deps:
    base: Deps  # Tier 3 dependencies; its hooks are set per run here
    executor: Callable[[], Executor]  # a fresh executor per run (one action per run)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)


def tier3_config(cfg: Config, rcfg: RecoveryConfig) -> Config:
    """Stop polling at the first detection on every scenario (we act on it), and keep the
    dead-man timer armed through the action and the recovery window."""
    extra = ROLLOUT_TIMEOUT_S + rcfg.window_s + rcfg.hold_s + rcfg.check_s
    return replace(cfg, stop_at_first_detection=True, extra_hold_s=cfg.extra_hold_s + extra)


def _iso(ts: float | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds")


def _ts(iso: str | None) -> float | None:
    return None if iso is None else datetime.fromisoformat(iso).timestamp()


def run_experiment(scn: Scenario, deps: Tier4Deps, cfg: Config, run_id: str,
                   results_dir: Path | None = None) -> dict:
    b, rcfg = deps.base, deps.recovery
    ex = deps.executor()
    own_timer: list[str] = []  # a dead-man timer armed here when Tier 3 armed none (K000)

    def after_scored(rec: dict, decision, ctx: dict) -> None:
        base_kube, base_stats = ctx["base_kube"], ctx["base_stats"]
        detected = decision.status == DECISION_OK and bool(decision.incident_detected)
        rem: dict = {"detected": detected, "checks": []}
        rec["remediation"] = rem
        prior = None
        if detected:
            prior = check(b.stats, b.kube_state, base_stats, base_kube, b.clock(), rcfg)
            if rec.get("deadman_unit") is None and b.deadman is not None:
                own_timer.append(b.deadman.arm(int(ROLLOUT_TIMEOUT_S + rcfg.window_s + rcfg.hold_s
                                                   + rcfg.check_s + cfg.deadman_margin_s)))
                rem["deadman_unit"] = own_timer[-1]
            act = ex.execute(decision.recommended_action, decision.affected_component,
                             {n: d["desired"] for n, d in base_kube.deployments.items()})
        else:
            act = {"action": decision.recommended_action, "component": decision.affected_component,
                   "executed": False, "not_executable": "no incident reported",
                   "t_action_start": None, "t_action_done": None}
        rem["action"] = act
        start = _ts(act["t_action_done"]) if act["executed"] else b.clock()
        w = watch(start, b.stats, b.kube_state, base_stats, base_kube, b.clock, b.sleep, rcfg,
                  prior=prior)
        rem["checks"] = [{**c, "t": _iso(c["t"])} for c in w["checks"]]
        rem["outcome"] = outcome(act["executed"], w["t_recovered"] is not None, w["before_action"])
        t_inject = ctx["t_inject"]
        t_rec, t_ver = _iso(w["t_recovered"]), _iso(w["t_recovery_verified"])
        by_action = rem["outcome"] == RECOVERED_BY_ACTION
        rem["remediation_executed"] = act["executed"]
        rem["recovery_verified"] = by_action
        rem["self_recovered"] = rem["outcome"] == RECOVERED_WITHOUT_ACTION
        rem["relapsed"] = w["relapsed"]
        rem["timing"] = {
            "t_action_start": act["t_action_start"], "t_action_done": act["t_action_done"],
            "t_recovered": t_rec, "t_recovery_verified": t_ver,
            "mttr_seconds": seconds_between(_iso(t_inject), act["t_action_done"]) if act["executed"] else None,
            "mttr_verified_seconds": seconds_between(_iso(t_inject), t_rec) if by_action else None,
            "self_recovery_seconds": seconds_between(_iso(t_inject), t_rec) if rem["self_recovered"] else None,
            "time_to_remediate_seconds": seconds_between(ctx["t_detect"], act["t_action_done"])
            if act["executed"] else None,
        }
        false_alarm = not scn.expected.incident and act["executed"]
        rem["action_on_false_alarm"] = false_alarm
        after = [c for c in w["checks"] if c is not prior]
        rem["disturbed_healthy_system"] = false_alarm and disturbed(after)

    def on_reset() -> list[str]:
        done = ex.restore()
        for unit in own_timer:
            b.deadman.disarm(unit)
        return done

    run_deps = replace(b, after_scored=after_scored, on_reset=on_reset)
    return tier3_run_experiment(scn, run_deps, tier3_config(cfg, rcfg), run_id, results_dir)


def run_batch(scenarios: list[Scenario], deps: Tier4Deps, cfg: Config, results_dir: Path,
              repeats: int = 1, run_id: str | None = None) -> dict:
    """As Tier 3's batch: a contaminated run stops it; pods are rebalanced between runs."""
    from tier4.report import scorecard_data

    run_id = run_id or new_run_id()
    records, stopped = [], None
    for _ in range(repeats):
        for scn in scenarios:
            r = run_experiment(scn, deps, cfg, run_id, results_dir)
            records.append(r)
            if r["status"] == STATUS_CONTAMINATED:
                stopped = f"{scn.experiment_id}: {r['message']}"
                break
            if deps.base.between_runs:
                deps.base.between_runs()
        if stopped:
            break
    from tier3.layouts import get_layout
    summary = {"run_id": run_id, "mode": cfg.mode, "repeats": repeats, "stopped_early": stopped,
               "observation": get_layout(cfg.layout).identity,
               "prompt": engine_prompt(deps.base.engine),
               "scorecard": scorecard_data(records),
               "runs": [{"experiment_id": r["experiment_id"], "status": r["status"],
                         "evaluation": r["evaluation"],
                         "outcome": (r.get("remediation") or {}).get("outcome")} for r in records]}
    d = Path(results_dir) / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
