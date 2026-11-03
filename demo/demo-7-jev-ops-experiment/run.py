#!/usr/bin/env python3
"""Run the Jev ops-decision experiment.

    python run.py                                  one run, random scenario
    python run.py --scenario degraded              one run, named scenario
    python run.py --scenario ambiguous --seed 1234 reproducible input
    python run.py --runs 70                        batch, scenarios in rotation
    python run.py --runs 70 --no-logs              same, but metrics only (no logs)
    python run.py --runs 70 --engine claude-sonnet same batch, answered by a baseline LLM

Engines: jev (default), claude-haiku, claude-sonnet, claude-opus, ollama:<model>. Every engine
gets the same observation and the same questions.

Read-only: it generates data, asks the engine, and records the answer. It never acts on it.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluation.evaluator import FIELDS, aggregate, format_summary, make_record
from generator.server_state import generate
from jev.client import ConfigError, load_prompt
from llm.engines import ENGINE_CHOICES, build_engine, engine_info, engine_slug, parse_engine
from prompts.loader import DEFAULT_VARIANT, VariantError, engine_prompt, list_variants
from scenarios.definitions import get_scenario, scenario_names


def plan_runs(scenario_arg: str | None, runs: int, base_seed: int) -> list[tuple[str, int]]:
    """Return (scenario, seed) for each run. Fully determined by the arguments."""
    names = scenario_names()
    rng = random.Random(base_seed)
    plan = []
    for i in range(runs):
        if scenario_arg and scenario_arg != "random":
            name = scenario_arg
        elif scenario_arg == "random" or runs == 1:
            name = rng.choice(names)
        else:
            name = names[i % len(names)]  # balanced rotation for batches
        plan.append((name, base_seed + i))
    return plan


MAX_LOG_LINES = 15


def format_logs(logs, included: bool) -> list[str]:
    if not included:
        return ["Logs", "----", "not sent to Jev (--no-logs)"]
    logs = logs or []
    counts = {lvl: sum(1 for x in logs if x["level"] == lvl) for lvl in ("FATAL", "ERROR", "WARN", "INFO")}
    head = ", ".join(f"{n} {lvl}" for lvl, n in counts.items() if n) or "none"
    lines = [f"Logs (last 15 min): {head}", "----"]
    for x in logs[-MAX_LOG_LINES:]:
        lines.append(f"{x['ts'][11:19]} {x['level']:<5} {x['service']:<8} {x['message']}")
    if len(logs) > MAX_LOG_LINES:
        lines.append(f"... {len(logs) - MAX_LOG_LINES} older lines not shown (all were sent to Jev)")
    return lines


def _n(x, spec=",.0f", unit=""):
    return "n/a" if x is None else f"{x:{spec}}{unit}"


def format_run_telemetry(d: dict) -> list[str]:
    cost = "n/a" if d.get("cost_usd") is None else f"${d['cost_usd']:.6f} ({d['cost_source']})"
    return [
        f"Tokens:       {_n(d.get('input_tokens'))} in / {_n(d.get('output_tokens'))} out",
        f"Cost:         {cost}",
        f"Round trip:   {_n(d.get('latency_ms'), '.0f', ' ms')}  (server elapsed {_n(d.get('server_elapsed_ms'), '.0f', ' ms')})",
        f"Total:        {_n(d.get('total_ms'), '.0f', ' ms')}  in {d.get('attempts', 0)} attempt(s), HTTP {d.get('http_status') or 'n/a'}",
        f"Request:      {_n(d.get('request_bytes'))} bytes",
    ]


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5, cwd=Path(__file__).parent).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _engine_heading(record: dict) -> list[str]:
    e = record.get("engine") or {"name": "jev"}
    title = "Jev" if e["name"] == "jev" else f"{e['name']} ({e.get('transport')})"
    return [title, "-" * len(title)]


def format_run(record: dict) -> str:
    o, e, d = record["observations"], record["expected"], record["decision"]
    svc = ", ".join(f"{k}={v}" for k, v in o["services"].items())
    lines = [
        f"Scenario: {record['scenario']}",
        f"Seed: {record['seed']}",
        "",
        "Server state", "------------",
        f"CPU:          {o['cpu_pct']}%",
        f"Memory:       {o['memory_pct']}%",
        f"Disk:         {o['disk_pct']}%",
        f"Load:         {o['load_1m']}",
        f"Net errors:   {o['network_errors_per_min']}/min",
        f"App errors:   {o['app_error_rate_pct']}%",
        f"Latency:      {o['api_latency_ms']}ms",
        f"Services:     {svc}",
        f"Restarts/h:   {o['restarts_last_hour']}",
        f"Events:       {'; '.join(o['recent_events']) or 'none'}",
        "",
        *format_logs(o.get("logs"), record.get("include_logs", True)),
        "",
        "Expected", "--------",
        f"Severity:     {e['severity']}",
        f"Action:       {e['action']}",
        f"Human review: {e['human_review']}",
        f"Cause:        {e.get('probable_cause', 'n/a')}",
        "",
        *_engine_heading(record),
    ]
    if d["status"] == "ok":
        conf = d["confidence"]
        p = d.get("human_review_probability")
        lines += [
            f"Severity:     {d['severity']}",
            f"Action:       {d['action']}",
            f"Human review: {d['human_review']}" + ("" if p is None else f"  (P(yes)={p:.2f})"),
            f"Cause:        {d.get('probable_cause')}",
            f"Confidence:   {'n/a' if conf is None else f'{conf:.2f}'}",
        ]
    else:
        lines.append(f"Status:       {d['status'].upper()}: {d['message']}")
    if d["status"] == "error":
        result = "ERROR (no answer, excluded from accuracy)"
    elif record["match"]["overall"]:
        result = "PASS"
    else:
        wrong = [f for f in FIELDS if not record["match"][f]]
        result = "FAIL (" + ", ".join(wrong) + ")"
    lines += ["", "Telemetry", "---------", *format_run_telemetry(d), "", f"Result: {result}"]
    if d["status"] == "ok" and "probable_cause" in record["match"]:
        verdict = "correct" if record["match"]["probable_cause"] else "incorrect"
        lines.append(f"Cause:  {verdict} (scored separately)")
    return "\n".join(lines)


def run_experiment(client, scenario_arg, runs, base_seed, out_dir, verbose=None, out=print,
                   include_logs=True):
    info = engine_info(client)
    prompt = engine_prompt(client)
    verbose = (runs == 1) if verbose is None else verbose
    records = []
    started_at = datetime.now(timezone.utc)
    t_start = time.perf_counter()
    for i, (name, seed) in enumerate(plan_runs(scenario_arg, runs, base_seed), 1):
        scenario = get_scenario(name)
        obs = generate(scenario, seed).to_dict()
        if not include_logs:  # ablation: same metrics, logs withheld from Jev
            obs.pop("logs")
        if getattr(client, "accepts_seed", False):  # local models: reproducible per run
            decision = client.decide(obs, seed=seed)
        else:
            decision = client.decide(obs)
        rec = make_record(i, name, seed, obs, scenario.expected, decision)
        rec["engine"] = {**info, "model_version": decision.to_dict().get("model_version")}
        rec["prompt"] = prompt
        rec["include_logs"] = include_logs
        rec["log_lines_sent"] = len(obs.get("logs", []))
        rec["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        records.append(rec)
        if verbose:
            out(format_run(rec))
        else:
            tag = "ERROR" if rec["status"] == "error" else ("PASS" if rec["match"]["overall"] else "FAIL")
            toks = decision.to_dict().get("input_tokens")
            out(f"[{i}/{runs}] {name:<15} seed={seed:<8} {tag:<5} "
                f"{_n(decision.to_dict().get('latency_ms'), '.0f', ' ms'):>8}  {_n(toks)} tok")
    wall_s = time.perf_counter() - t_start
    summary = aggregate(records)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "" if info["name"] == "jev" else f"-{engine_slug(info['name'])}"
    if prompt and prompt["name"] != DEFAULT_VARIANT:
        suffix += f"-prompt-{prompt['name']}"
    base = out_dir / f"{stamp}-{runs}runs{'' if include_logs else '-nologs'}{suffix}"
    with open(f"{base}.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    meta = {"base_seed": base_seed, "scenario_arg": scenario_arg, "runs": runs,
            "include_logs": include_logs,
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wall_time_s": round(wall_s, 3),
            "runs_per_min": round(runs / wall_s * 60, 1) if wall_s > 0 else None,
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "engine": info,
            "prompt": prompt,
            "jev_url": getattr(getattr(client, "config", None), "url", None),
            "jev_model_requested": getattr(getattr(client, "config", None), "model", None),
            "models": sorted({r["decision"].get("model_version") for r in records
                              if r["decision"].get("model_version")})}
    with open(f"{base}-summary.json", "w") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, indent=2)
    if runs > 1:
        out("")
        out(format_summary(summary))
    out(f"\nWall time: {wall_s:.1f} s ({meta['runs_per_min']} runs/min)")
    out(f"\nResults: {base}.jsonl")
    return records, summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", help=f"one of: {', '.join(scenario_names())}, or 'random'")
    p.add_argument("--seed", type=int, help="base seed; a random one is chosen and printed if omitted")
    p.add_argument("--runs", type=int, default=1, help="number of runs (default 1)")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--verbose", action="store_true", help="print full detail for every run in a batch")
    p.add_argument("--no-logs", action="store_true",
                   help="withhold application logs from the engine (metrics only), to compare against runs with logs")
    p.add_argument("--prompt", default=DEFAULT_VARIANT,
                   help=f"prompt variant (prompts/tier1/*.yaml): {', '.join(list_variants('tier1'))} "
                        f"(default {DEFAULT_VARIANT})")
    p.add_argument("--engine", default="jev",
                   help=f"who answers: {', '.join(ENGINE_CHOICES)} (default jev)")
    a = p.parse_args(argv)
    if a.runs < 1:
        p.error("--runs must be at least 1")
    try:
        parse_engine(a.engine)
    except ValueError as e:
        p.error(str(e))
    try:
        a.prompt_variant = load_prompt(a.prompt)
    except VariantError as e:
        p.error(str(e))
    if a.scenario and a.scenario != "random" and a.scenario not in scenario_names():
        p.error(f"unknown scenario {a.scenario!r}; valid: {', '.join(scenario_names())}, random")
    return a


def main(argv=None, client=None) -> int:
    a = parse_args(argv)
    if client is None:
        try:
            client = build_engine(a.engine, prompt=a.prompt_variant)
        except ConfigError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    seed = a.seed if a.seed is not None else random.SystemRandom().randrange(1_000_000)
    print(f"Base seed: {seed}")
    print(f"Prompt: {a.prompt}\n")
    run_experiment(client, a.scenario, a.runs, seed, a.output_dir, verbose=a.verbose or None,
                   include_logs=not a.no_logs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
