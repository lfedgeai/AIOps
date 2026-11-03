#!/usr/bin/env python3
"""Turn a results JSONL file into a Markdown report.

    python report.py                 latest file in results/
    python report.py results/X.jsonl a specific file
    python report.py --compare A.jsonl B.jsonl ...   engines side by side
    python report.py --compare A.jsonl B.jsonl --by-prompt   one column per engine and prompt variant
    python report.py --metrics A.jsonl [B.jsonl ...] F1, p95, detection rate, cost... with a glossary

Everything is computed from the records. Nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from evaluation.evaluator import ALL_FIELDS, aggregate, format_summary
from evaluation.metrics_report import render_metrics
from prompts.compare import HashConflict
from prompts.loader import ASSUMED_NOTE, record_prompt


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest(results_dir: Path) -> Path:
    files = sorted(results_dir.glob("*runs*.jsonl"))
    if not files:
        raise SystemExit(f"no *runs.jsonl files in {results_dir}. Run run.py first.")
    return files[-1]


def _n(x) -> str:
    return "n/a" if x is None else f"{x:,.0f}"


def engine_of(records: list[dict]) -> dict:
    """Engine identity of a result file. Records written before engines existed are Jev."""
    for r in records:
        if r.get("engine"):
            return r["engine"]
    return {"name": "jev", "transport": "jev-api"}


def choices(records: list[dict], field: str) -> str:
    c = Counter(r["decision"].get(field) or r["status"] for r in records)
    return ", ".join(f"{k} x{v}" for k, v in c.most_common())


def render(records: list[dict], source: str) -> str:
    s = aggregate(records)
    eng = engine_of(records)
    who = "Jev" if eng["name"] == "jev" else eng["name"]
    logs = "yes" if all(r.get("include_logs", True) for r in records) else "no (metrics only)"
    out = ["# Jev ops experiment report", "", f"Source: `{source}`", "",
           f"Engine: {who} (transport {eng.get('transport')})", "",
           f"Application logs sent to {who}: {logs}", ""]
    if eng.get("notice"):
        out += [f"> **Note:** {eng['notice']}", ""]
    out += ["```text", format_summary(s), "```", "",
            f"## What {who} chose, by scenario", "",
            f"Expected value first, then everything {who} returned for that scenario.", ""]
    for name in dict.fromkeys(r["scenario"] for r in records):
        rs = [r for r in records if r["scenario"] == name]
        exp = rs[0]["expected"]
        out += [f"### {name} ({len(rs)} runs)", "",
                f"| Field | Expected | {who} returned |", "|---|---|---|"]
        out += [f"| {f} | {exp.get(f, 'n/a')} | {choices(rs, f)} |" for f in ALL_FIELDS]
        out.append("")

    out += ["## Telemetry by scenario", "",
            "| Scenario | Log lines | Tokens in | Tokens out | Round trip (ms) | Cost/run (USD) |",
            "|---|---|---|---|---|---|"]
    for name, t in s["telemetry"]["by_scenario"].items():
        cost = "n/a" if t["mean_cost_usd"] is None else f"{t['mean_cost_usd']:.6f}"
        out.append(f"| {name} | {_n(t['mean_log_lines'])} | {_n(t['mean_input_tokens'])} | "
                   f"{_n(t['mean_output_tokens'])} | {_n(t['mean_latency_ms'])} | {cost} |")
    out += ["", "Means per run. Compare with a `--no-logs` report to see what the logs cost in tokens.", ""]

    bad = [r for r in records if r["status"] != "ok" or not r["match"]["overall"]]
    out += [f"## Misses ({len(bad)})", "", "A miss is a wrong operational decision. Cause is shown but does not decide PASS/FAIL.", ""]
    if bad:
        out += ["| Run | Scenario | Seed | Status | Severity (exp/got) | Action (exp/got) | Human (exp/got) | Cause (exp/got) | Conf |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in bad:
            d, e = r["decision"], r["expected"]
            conf = d.get("confidence")
            out.append(
                f"| {r['run']} | {r['scenario']} | {r['seed']} | {r['status']} | "
                f"{e['severity']}/{d.get('severity') or '-'} | {e['action']}/{d.get('action') or '-'} | "
                f"{e['human_review']}/{d.get('human_review') or '-'} | "
                f"{e.get('probable_cause', 'n/a')}/{d.get('probable_cause') or '-'} | "
                f"{'n/a' if conf is None else f'{conf:.2f}'} |")
        flag = "" if eng["name"] == "jev" else f" --engine {eng['name']}"
        out += ["", f"Replay one: `python run.py --scenario <scenario> --seed <seed>{flag}`"]
    else:
        out.append("None.")
    return "\n".join(out) + "\n"


def _pct(c: int, t: int) -> str:
    return "n/a" if not t else f"{c}/{t} ({c / t:.0%})"


def prompt_of(records: list[dict]) -> str:
    """The prompt variant of a set of records, as `name` (plus ` v<n>` after version 1).
    Records without a prompt field are baseline; a mix is shown as such."""
    names = sorted({(record_prompt(r)[0]["name"], record_prompt(r)[0]["version"]) for r in records},
                   key=lambda x: (x[0], x[1] or 0))
    if len(names) != 1:
        return "mixed prompts"
    n, v = names[0]
    policy = any(record_prompt(r)[0].get("style") == "policy" for r in records)
    return (n if v in (1, None) else f"{n} v{v}") + (" [policy stated]" if policy else "")


def _label(records: list[dict]) -> str:
    eng = engine_of(records)
    nologs = "" if all(r.get("include_logs", True) for r in records) else " (no logs)"
    return f"{eng['name']} / {prompt_of(records)}{nologs}"


def check_prompt_hashes(runs: list[tuple[str, list[dict]]]) -> None:
    """Stop when one prompt name and version carries two hashes (an edited variant)."""
    seen: dict[tuple, tuple[str, str]] = {}
    for src, rs in runs:
        for r in rs:
            ident, _ = record_prompt(r)
            if not ident["sha256"]:
                continue
            k = (ident["name"], ident["version"])
            first = seen.setdefault(k, (ident["sha256"], src))
            if first[0] != ident["sha256"]:
                raise HashConflict(f"prompt {k[0]!r} v{k[1]} has two different hashes: "
                                   f"{first[0][:12]}... in {first[1]} and {ident['sha256'][:12]}... in {src}. "
                                   "A variant file was edited without bumping its version.")


def by_prompt(runs: list[tuple[str, list[dict]]]) -> list[tuple[str, list[dict]]]:
    """Regroup the records of several files by (engine, prompt, logs) instead of by file."""
    groups: dict[tuple, list[dict]] = {}
    for _, rs in runs:
        for r in rs:
            ident, _ = record_prompt(r)
            eng = (r.get("engine") or {}).get("name") or "jev"
            groups.setdefault((eng, ident["name"], ident["version"], r.get("include_logs", True)), []).append(r)
    return [(f"{e} / {n}" + ("" if v in (1, None) else f" v{v}") + ("" if logs else " (no logs)"), rs)
            for (e, n, v, logs), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] != "baseline",
                                                                            kv[0][1], kv[0][2] or 0, not kv[0][3]))]


def _ms(x) -> str:
    return "n/a" if x is None else f"{x:,.0f}"


def column_labels(runs: list[tuple[str, list[dict]]]) -> list[str]:
    labels = [_label(rs) for _, rs in runs]
    seen: dict[str, int] = {}
    for i, lab in enumerate(labels):  # same engine twice: number them
        seen[lab] = seen.get(lab, 0) + 1
        if labels.count(lab) > 1:
            labels[i] = f"{lab} #{seen[lab]}"
    return labels


def input_warnings(runs: list[tuple[str, list[dict]]]) -> list[str]:
    warnings = []
    pairs = [{(r["scenario"], r["seed"]) for r in rs} for _, rs in runs]
    if any(p != pairs[0] for p in pairs[1:]):
        warnings.append("The files do not cover the same scenario and seed pairs, so the engines "
                        "did not see identical observations. Use the same `--seed` and `--runs`.")
    logs = {all(r.get("include_logs", True) for r in rs) for _, rs in runs}
    if len(logs) > 1:
        warnings.append("Some files were run with logs and some without (`--no-logs`).")
    if any(prompt_of(rs) == "mixed prompts" for _, rs in runs):
        warnings.append("A column mixes prompt variants. Use `--by-prompt` to split it.")
    out = [line for w in warnings for line in (f"> **Warning:** {w}", "")]
    assumed = sum(1 for _, rs in runs for r in rs if record_prompt(r)[1])
    if assumed:
        out += [f"> Note: {assumed} record(s) had {ASSUMED_NOTE}.", ""]
    return out


def render_compare(runs: list[tuple[str, list[dict]]]) -> str:
    """Several result files side by side. Every figure comes from aggregate()."""
    labels = column_labels(runs)
    sums = [aggregate(rs) for _, rs in runs]
    engines = [engine_of(rs) for _, rs in runs]
    head = "| | " + " | ".join(labels) + " |"
    sep = "|" + "---|" * (len(labels) + 1)

    out = ["# Engine comparison", ""]
    out += input_warnings(runs)

    out += ["## Sources", "", "| Engine | File | Transport | Model that answered | Effort |",
            "|---|---|---|---|---|"]
    for lab, (src, rs), eng in zip(labels, runs, engines):
        models = sorted({r["decision"].get("model_version") for r in rs} - {None}) or ["n/a"]
        out.append(f"| {lab} | `{src}` | {eng.get('transport')} | {', '.join(models)} | "
                   f"{eng.get('effort') or 'n/a'} |")
    for lab, eng in zip(labels, engines):
        if eng.get("transport") == "claude-cli":
            out += ["", f"> **{lab}** ran through the `claude -p` CLI, not the bare API "
                    "(no `ANTHROPIC_API_KEY`). Its cost is what the CLI reports; on a "
                    "subscription plan that is a notional figure, not a charge."]
    out.append("")

    out += ["## Operational decision (PASS needs severity, action and human review)", "", head, sep]
    rows = [("Accuracy", lambda s: _pct(s["correct"], s["scored"])),
            ("Runs", lambda s: str(s["runs"])),
            ("Invalid (counted wrong)", lambda s: str(s["invalid"])),
            ("Errored (excluded)", lambda s: str(s["errored"]))]
    out += [f"| {name} | " + " | ".join(f(s) for s in sums) + " |" for name, f in rows]
    out += ["", "## Accuracy by decision", "", head, sep]
    for f in ALL_FIELDS:
        note = " (diagnosis, not in PASS/FAIL)" if f == "probable_cause" else ""
        out.append(f"| {f}{note} | " + " | ".join(
            _pct(s["by_field"][f]["correct"], s["by_field"][f]["total"]) for s in sums) + " |")
    out += ["", "## Accuracy by scenario", "", head, sep]
    names = list(dict.fromkeys(r["scenario"] for _, rs in runs for r in rs))
    for n in names:
        cells = []
        for s in sums:
            b = s["by_scenario"].get(n)
            cells.append("n/a" if not b else _pct(b["correct"], b["total"]))
        out.append(f"| {n} | " + " | ".join(cells) + " |")

    out += ["", "## Telemetry (means per run unless noted)", "", head, sep]
    tel = [s["telemetry"] for s in sums]

    def cost(t):
        c = t["cost_usd"]
        if c["total"] is None:
            return "n/a"
        return f"${c['total']:.4f} total ({', '.join(c['sources'])})"

    rows = [("Tokens in", lambda t: _ms(t["input_tokens"]["mean"])),
            ("Tokens out", lambda t: _ms(t["output_tokens"]["mean"])),
            ("Round trip p50 (ms)", lambda t: _ms(t["round_trip_ms"]["p50"])),
            ("Round trip p95 (ms)", lambda t: _ms(t["round_trip_ms"]["p95"])),
            ("Cost", cost)]
    out += [f"| {name} | " + " | ".join(f(t) for t in tel) + " |" for name, f in rows]
    out += ["", "Confidence is not compared: only Jev returns probabilities. Baseline LLMs "
            "report none, and none is invented.", ""]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="results JSONL (default: latest in results/)")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--compare", nargs="+", metavar="JSONL", help="two or more results files to compare")
    p.add_argument("--metrics", nargs="+", metavar="JSONL",
                   help="metrics report (F1, p95, detection rate, cost...) for one or more files")
    p.add_argument("--by-prompt", action="store_true",
                   help="with --compare or --metrics: one column per engine and prompt variant, not per file")
    a = p.parse_args(argv)
    if a.by_prompt and not (a.compare or a.metrics):
        p.error("--by-prompt needs --compare or --metrics")
    if a.metrics:
        paths = [Path(x) for x in a.metrics]
        runs = [(x.name, load(x)) for x in paths]
        try:
            check_prompt_hashes(runs)
        except HashConflict as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if a.by_prompt:
            runs = by_prompt(runs)
        text = render_metrics(runs, column_labels(runs), input_warnings(runs) if len(runs) > 1 else [])
        report = paths[0].with_name(datetime.now().strftime("%Y%m%d-%H%M%S") + "-metrics-report.md")
        report.write_text(text)
        print(text)
        print(f"Report saved: {report}")
        return 0
    if a.compare:
        if len(a.compare) < 2 and not a.by_prompt:
            p.error("--compare needs at least two files (or --by-prompt)")
        paths = [Path(x) for x in a.compare]
        runs = [(x.name, load(x)) for x in paths]
        try:
            check_prompt_hashes(runs)
        except HashConflict as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        text = render_compare(by_prompt(runs) if a.by_prompt else runs)
        report = paths[0].with_name(datetime.now().strftime("%Y%m%d-%H%M%S") + "-compare-report.md")
        report.write_text(text)
        print(text)
        print(f"Report saved: {report}")
        return 0
    path = Path(a.file) if a.file else latest(Path(a.results_dir))
    text = render(load(path), path.name)
    report = path.with_name(path.name.replace(".jsonl", "-report.md"))
    report.write_text(text)
    print(text)
    print(f"Report saved: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
