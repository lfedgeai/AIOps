"""Tier 4 command line: list, reset-faults, run, batch, scorecard, export-lfedge.
The cluster, health check and scenarios are Tier 3's."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tier3 import testbed
from tier3.cli import (ENGINES, _write, add_layout_arg, add_prompt_arg, build_deps, make_config, print_record,
                       resolve_prompt)
from tier3.scenarios import ScenarioError, format_list, get_scenario, load_all
from tier4.runner import DEFAULT_RESULTS


def build_tier4_deps(engine: str = "jev", prompt=None):
    from tier3.collectors.kube import Kube
    from tier4.executor import Executor, kubectl_undo
    from tier4.runner import Tier4Deps

    base = build_deps(engine, prompt)
    kube = Kube.from_kubeconfig(testbed.kubeconfig(), testbed.namespace())
    undo = kubectl_undo(testbed.kubeconfig(), testbed.namespace())
    return Tier4Deps(base=base, executor=lambda: Executor(kube, undo=undo))


class ScriptedEngine:
    """For `probe` only: always reports an incident with a fixed action and component, so
    an action can be tried on a fault without calling Jev (tasks 1.3-1.6)."""

    def __init__(self, action: str, component: str):
        self.action, self.component = action, component
        self.info = {"name": "probe", "transport": "scripted"}

    def decide(self, observation):
        from tier3.engine import Decision
        return Decision(status="ok", incident_detected=True, affected_component=self.component,
                        diagnosis="unknown", recommended_action=self.action, confidence=1.0,
                        telemetry={"model": "probe"})


def print_remediation(rec: dict) -> None:
    rem = rec.get("remediation")
    if not rem:
        return
    a, t = rem["action"], rem["timing"]
    what = a.get("operation") or f"{a.get('action')} on {a.get('component')}: {a.get('not_executable')}"
    print(f"  action        {what}")
    print(f"  outcome       {rem['outcome']}  mttr={t['mttr_seconds']} verified={t['mttr_verified_seconds']} "
          f"self={t['self_recovery_seconds']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tier4", description="Tier 4: closed-loop remediation on kind")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list experiment IDs (Tier 3's catalogue)")
    sub.add_parser("reset-faults", help="undo a recorded action, then the fault; remove jev-fault objects")
    r = sub.add_parser("run", help="run one experiment")
    r.add_argument("experiment_id")
    b = sub.add_parser("batch", help="run several experiments (default: the whole catalogue)")
    b.add_argument("experiment_ids", nargs="*")
    b.add_argument("--repeats", type=int, default=1)
    pr = sub.add_parser("probe", help="try one action on one fault, without Jev (testbed probes)")
    pr.add_argument("experiment_id")
    pr.add_argument("action")
    pr.add_argument("component")
    pr.add_argument("--results-dir", default=str(DEFAULT_RESULTS / "probe"))
    for p in (r, b):
        p.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
        p.add_argument("--engine", choices=ENGINES, default="jev")
        add_prompt_arg(p)
        add_layout_arg(p)
    for name, hlp in (("scorecard", "print the scorecard for a results directory"),
                      ("export-lfedge", "write JSONL in the LF Edge harness's field names")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("results")
        p.add_argument("--include-replay", action="store_true",
                       help="also read replay records (left out by default; replay has no remediation)")
        if name != "scorecard":
            p.add_argument("-o", "--output")
    args = ap.parse_args(argv)
    resolve_prompt(ap, args)

    if args.cmd == "list":
        print(format_list())
        return 0
    if args.cmd in ("scorecard", "export-lfedge"):
        from tier4.report import export_lfedge, format_scorecard, load_records
        recs = load_records(args.results, include_replay=args.include_replay)
        if args.cmd == "scorecard":
            print(format_scorecard(recs))
        else:
            _write("".join(json.dumps(x) + "\n" for x in export_lfedge(recs)), args.output)
        return 0
    if args.cmd == "reset-faults":
        from tier3.collectors.kube import Kube
        from tier3.faults import reset_faults
        changed = reset_faults(Kube.from_kubeconfig(testbed.kubeconfig(), testbed.namespace()))
        print("reset: " + ("; ".join(changed) if changed else "nothing to change"))
        return 0

    from tier4.runner import new_run_id, run_batch, run_experiment
    if args.cmd == "probe":
        from dataclasses import replace
        deps = build_tier4_deps()
        deps.base = replace(deps.base, engine=ScriptedEngine(args.action, args.component))
        rec = run_experiment(get_scenario(args.experiment_id), deps, make_config("poll"),
                             new_run_id(), Path(args.results_dir))
        print_record(rec)
        print_remediation(rec)
        return 0 if rec["status"] == "ok" else 1
    try:
        if args.cmd == "run":
            scns = [get_scenario(args.experiment_id)]
        elif args.experiment_ids:
            scns = [get_scenario(e) for e in args.experiment_ids]
        else:
            scns = list(load_all().values())
    except ScenarioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    deps, cfg = build_tier4_deps(args.engine, args.prompt_variant), make_config("poll", args.observation)
    if args.cmd == "run":
        rec = run_experiment(scns[0], deps, cfg, new_run_id(), Path(args.results_dir))
        print_record(rec)
        print_remediation(rec)
        print(f"result: {Path(args.results_dir) / rec['run_id']}")
        return 0 if rec["status"] == "ok" else 1
    summary = run_batch(scns, deps, cfg, Path(args.results_dir), repeats=args.repeats)
    from tier4.report import format_scorecard, load_records
    print(format_scorecard(load_records(Path(args.results_dir) / summary["run_id"])), flush=True)
    if summary["stopped_early"]:
        print(f"stopped early: {summary['stopped_early']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
