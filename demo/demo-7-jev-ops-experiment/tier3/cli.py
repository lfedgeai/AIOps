"""Tier 3 command line: list, health, reset-faults, run, batch, scorecard,
export-lfedge, export-reasoning."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tier3 import testbed
from tier3.scenarios import ScenarioError, format_list, get_scenario, load_all
from tier3.vocab import FAULT_TYPES

DEFAULT_RESULTS = Path("results") / "tier3"
ENGINES = ("jev", "claude-opus")
REPO = Path(__file__).resolve().parents[1]


def _clients():
    from tier2.collectors.jaeger import Jaeger
    from tier2.collectors.opensearch import OpenSearch
    from tier2.collectors.prometheus import Prometheus
    from tier2.workload import Workload
    from tier3.collectors.kube import Kube
    from tier3.faults import FaultInjector

    kube = Kube.from_kubeconfig(testbed.kubeconfig(), testbed.namespace())
    return dict(kube=kube, injector=FaultInjector(kube), prom=Prometheus(testbed.prometheus_url()),
                jaeger=Jaeger(testbed.jaeger_url()), logs=OpenSearch(testbed.opensearch_url()),
                workload=Workload(testbed.base_url()))


def build_engine(name: str = "jev", prompt=None):
    """The engine behind every poll, asking the given prompt variant (the baseline if None).
    Raises SystemExit on a configuration error, before any run."""
    if name == "jev":
        from tier3.jev_adapter import ConfigError, JevAdapter
        try:
            return JevAdapter(prompt=prompt)
        except ConfigError as e:
            raise SystemExit(f"error: {e}")
    from tier3.claude_engine import ConfigError, build_claude_engine
    try:
        return build_claude_engine(name, prompt=prompt)
    except ConfigError as e:
        raise SystemExit(f"error: {e}")


def add_prompt_arg(p) -> None:
    from prompts.loader import DEFAULT_VARIANT, list_variants
    p.add_argument("--prompt", default=DEFAULT_VARIANT,
                   help=f"prompt variant (prompts/tier3/*.yaml): {', '.join(list_variants('tier3'))} "
                        f"(default {DEFAULT_VARIANT})")


def add_layout_arg(p) -> None:
    from tier3.layouts import DEFAULT_LAYOUT, list_layouts
    p.add_argument("--observation", choices=list_layouts(), default=DEFAULT_LAYOUT,
                   help=f"observation layout (tier3/layouts.py), default {DEFAULT_LAYOUT}")


def resolve_prompt(ap, args) -> None:
    """Load --prompt into args.prompt_variant, or exit before any run with the variants listed."""
    from prompts.loader import VariantError
    from tier3.questions import load_prompt
    args.prompt_variant = None
    if getattr(args, "prompt", None) is not None:
        try:
            args.prompt_variant = load_prompt(args.prompt)
        except VariantError as e:
            ap.error(str(e))


def build_deps(engine: str = "jev", prompt=None):
    from tier3.collect import collect, kube_state
    from tier3.deadman import Deadman
    from tier3.runner import Deps

    engine = build_engine(engine, prompt)
    c = _clients()

    def health():
        failed = [x for x in testbed.run_checks(c["kube"], c["prom"], c["jaeger"], c["workload"],
                                                c["injector"]) if not x.ok]
        if failed:
            raise RuntimeError("testbed unhealthy: " + "; ".join(f"{x.name} ({x.detail})" for x in failed))

    def between_runs():
        # node_taint leaves one app node empty; the next run's health check would refuse it.
        if not testbed.placement_ok(c["kube"]):
            print("placement unbalanced after the run; running tier3/testbed/rebalance.sh", flush=True)
            subprocess.run([str(REPO / "tier3" / "testbed" / "rebalance.sh")], check=True)

    return Deps(engine=engine, injector=c["injector"], workload=c["workload"], health=health,
                between_runs=between_runs,
                stats=c["prom"].service_stats,
                kube_state=lambda start_s=None, end_s=None: kube_state(c["kube"], start_s, end_s),
                collect=lambda **kw: collect(c["prom"], c["jaeger"], c["logs"], c["kube"], **kw),
                deadman=Deadman())


def make_config(mode: str, layout: str | None = None):
    from tier2.testbed import git_commit
    from tier3.layouts import DEFAULT_LAYOUT
    from tier3.runner import Config
    return Config(mode=mode, baseline_wait_s=180, recover_wait_s=60, layout=layout or DEFAULT_LAYOUT,
                  scrub_terms=("flagd", *FAULT_TYPES),
                  versions=testbed.versions(), git_commit=git_commit())


def print_record(rec: dict) -> None:
    print(f"{rec['experiment_id']}  status={rec['status']}  {rec.get('message', '')}".rstrip())
    for dim, v in rec["evaluation"].items():
        print(f"  {dim:<13} {v}")
    t, u = rec["timing"], rec["usage"]
    print(f"  polls         {len(rec['polls'])} (scored: {rec.get('scored_poll', 'n/a')}), "
          f"engine calls {u['engine_calls']}")
    print(f"  timing        mttd={t['mttd_seconds']} manifest={t['t_manifest']} total={t['total_seconds']}")
    if rec.get("false_positive"):
        fp = rec["false_positive"]
        print(f"  false alarms  {fp['false_positive_polls']}/{fp['answered_polls']} polls")


def _write(text: str, out: str | None) -> None:
    if out:
        Path(out).write_text(text)
        print(f"wrote {out}")
    else:
        sys.stdout.write(text)


def _note_replay(path, included: bool) -> None:
    from tier3.replay import load_sources
    from tier3.report import load_records
    if included:
        return
    n = len(load_records(path, include_replay=True)) - len(load_sources(path))
    if n:
        print(f"note: {n} replay record(s) left out; add --include-replay to read them", file=sys.stderr)


def _replay(args) -> int:
    from tier3.layouts import LayoutError
    from tier3.replay import check_source_layout, load_sources, plan, replay_batch
    from tier3.report import format_scorecard
    p = plan(load_sources(args.records))
    try:
        for path, src in p["replay"]:
            check_source_layout(src, path)
    except LayoutError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"replay {len(p['replay'])} record(s), {p['engine_calls']} engine call(s), "
          f"engine {args.engine}, prompt {args.prompt}, observation {args.observation}; "
          f"skipped {len(p['skipped'])}", flush=True)
    for s in p["skipped"]:
        print(f"  skipped {s['path']}: {s['reason']} ({s['status']})")
    if args.dry_run or not p["replay"]:
        return 0 if p["replay"] or args.dry_run else 1
    engine = build_engine(args.engine, args.prompt_variant)
    summary = replay_batch(args.records, engine, args.results_dir, layout=args.observation)
    d = Path(args.results_dir) / "replay" / summary["run_id"]
    from tier3.report import load_records
    print(format_scorecard(load_records(d, include_replay=True)))
    print(f"censored runs: {summary['censored']}/{summary['replayed']}")
    print(f"result: {d}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tier3", description="Tier 3: Kubernetes faults on kind")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list experiment IDs")
    sub.add_parser("health", help="check the cluster and the demo")
    sub.add_parser("reset-faults", help="restore from the fault ledger, remove every jev-fault object")
    r = sub.add_parser("run", help="run one experiment")
    r.add_argument("experiment_id")
    b = sub.add_parser("batch", help="run several experiments (default: the whole catalogue)")
    b.add_argument("experiment_ids", nargs="*")
    b.add_argument("--repeats", type=int, default=1)
    for p in (r, b):
        p.add_argument("--mode", choices=("poll", "single"), default="poll")
        p.add_argument("--engine", choices=ENGINES, default="jev")
        add_prompt_arg(p)
        add_layout_arg(p)
        p.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    for name, hlp in (("scorecard", "print the scorecard for a results directory"),
                      ("export-lfedge", "write JSONL in the LF Edge harness's field names"),
                      ("export-reasoning", "write a Markdown sheet for hand scoring")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("results")
        p.add_argument("--include-replay", action="store_true",
                       help="also read replay records (left out by default)")
        if name != "scorecard":
            p.add_argument("-o", "--output")
    cp = sub.add_parser("compare", help="prompt variants side by side (live and replay records)")
    cp.add_argument("results", nargs="*", help="record directories (replay output, live runs, ...)")
    cp.add_argument("--dev", nargs="+", metavar="DIR", help="source records the variants were written on")
    cp.add_argument("--confirm", nargs="+", metavar="DIR", help="held-out source records (shown first)")
    cp.add_argument("--unpaired", action="store_true",
                    help="compare columns that did not see the same observations (warns)")
    cp.add_argument("-o", "--output")
    rp = sub.add_parser("replay", help="re-ask stored Tier 3/4 observations under a prompt variant "
                                       "(no cluster; calls the engine once per stored poll)")
    rp.add_argument("records", help="a run record, a run directory, or a parent of several")
    rp.add_argument("--engine", choices=ENGINES, default="jev")
    add_prompt_arg(rp)
    add_layout_arg(rp)
    rp.add_argument("--results-dir", default=str(DEFAULT_RESULTS),
                    help="replay records go to <results-dir>/replay/<run-id>/")
    rp.add_argument("--dry-run", action="store_true",
                    help="print what would be replayed and how many engine calls, then stop")
    args = ap.parse_args(argv)
    resolve_prompt(ap, args)

    if args.cmd == "list":
        print(format_list())
        return 0
    if args.cmd in ("scorecard", "export-lfedge", "export-reasoning"):
        from tier3.report import export_lfedge, export_reasoning, format_scorecard, load_records
        recs = load_records(args.results, include_replay=args.include_replay)
        _note_replay(args.results, args.include_replay)
        if args.cmd == "scorecard":
            print(format_scorecard(recs))
        elif args.cmd == "export-lfedge":
            _write("".join(json.dumps(x) + "\n" for x in export_lfedge(recs)), args.output)
        else:
            _write(export_reasoning(recs), args.output)
        return 0
    if args.cmd == "replay":
        return _replay(args)
    if args.cmd == "compare":
        from prompts.compare import HashConflict
        from tier3.compare import format_compare
        if not (args.results or args.dev or args.confirm):
            ap.error("compare needs record directories, --dev or --confirm")
        try:
            text = format_compare(args.results, args.dev, args.confirm, args.unpaired)
        except HashConflict as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        _write(text, args.output)
        return 0
    if args.cmd == "health":
        c = _clients()
        checks = testbed.run_checks(c["kube"], c["prom"], c["jaeger"], c["workload"], c["injector"])
        for x in checks:
            print(f"{'ok  ' if x.ok else 'FAIL'} {x.name:<12} {x.detail}")
        return 0 if all(x.ok for x in checks) else 1
    if args.cmd == "reset-faults":
        from tier3.collectors.kube import Kube
        from tier3.faults import reset_faults
        changed = reset_faults(Kube.from_kubeconfig(testbed.kubeconfig(), testbed.namespace()))
        print("reset: " + ("; ".join(changed) if changed else "nothing to change"))
        return 0

    from tier3.runner import new_run_id, run_batch, run_experiment
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
    deps, cfg = build_deps(args.engine, args.prompt_variant), make_config(args.mode, args.observation)
    if args.cmd == "run":
        rec = run_experiment(scns[0], deps, cfg, new_run_id(), Path(args.results_dir))
        print_record(rec)
        print(f"result: {Path(args.results_dir) / rec['run_id']}")
        return 0 if rec["status"] == "ok" else 1
    summary = run_batch(scns, deps, cfg, Path(args.results_dir), repeats=args.repeats)
    from tier3.report import format_scorecard, load_records
    print(format_scorecard(load_records(Path(args.results_dir) / summary["run_id"])))
    if summary["stopped_early"]:
        print(f"stopped early: {summary['stopped_early']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
