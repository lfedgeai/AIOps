#!/usr/bin/env python3
"""Summarize C-matrix JSON outputs into a markdown report."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def load_runs(out_dir: Path) -> list[dict]:
    runs: list[dict] = []
    patterns = ("context_C*_s*.json", "context_L*_s*.json")
    seen: set[str] = set()
    for pat in patterns:
        for path in sorted(out_dir.glob(pat)):
            if path.name in seen:
                continue
            seen.add(path.name)
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if "runs" in data:
                for r in data["runs"]:
                    r["_file"] = path.name
                    runs.append(r)
            else:
                data["_file"] = path.name
                runs.append(data)
    return runs


def _parse_context_c(filename: str, record: dict) -> str:
    if record.get("context_c"):
        return str(record["context_c"])
    m = re.search(r"context_C([^_]+)_", filename)
    if m:
        return m.group(1)
    m = re.search(r"context_L(\d+)_", filename)
    if m:
        return m.group(1)
    return "0"


def rate(items: list[dict], key: str) -> tuple[str, float | None]:
    vals = [x.get(key) for x in items if x.get(key) is not None]
    if not vals:
        return "n/a", None
    ok = sum(1 for v in vals if v)
    pct = 100 * ok / len(vals)
    return f"{ok}/{len(vals)} ({pct:.0f}%)", pct


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--report", default="docs/CONTEXT_ENGINEERING_EVAL_REPORT.md")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.out_dir
    runs = load_runs(out_dir)
    report_path = root / args.report

    for r in runs:
        r["context_c"] = _parse_context_c(r.get("_file", ""), r)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if not runs:
        report_path.write_text(
            f"# Context Engineering Evaluation Report\n\n"
            f"*Generated {now}*\n\n"
            "No result files found in `out/context_C*_*.json`.\n"
            "If the matrix is still running, re-run:\n\n"
            "`python scripts/analyze_context_matrix.py`\n"
        )
        print(f"Wrote empty report to {report_path}")
        return 1

    by_ctx: dict[str, list[dict]] = defaultdict(list)
    by_agent: dict[str, list[dict]] = defaultdict(list)
    by_fault: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_ctx[str(r.get("context_c", "0"))].append(r)
        by_agent[str(r.get("agent", r.get("classifier", "?")))].append(r)
        by_fault[str(r.get("fault_flag", "?"))].append(r)

    lines = [
        "# Context Engineering Evaluation Report",
        "",
        f"*Generated {now} — {len(runs)} runs*",
        "",
        "## Executive summary",
        "",
        "- **Matrix**: agents × 3 faults × single CONTEXT_C corpora (C1–C4; C0 optional).",
        "- **Corpora**: C1=source, C2=docs, C3=architecture, C4=dependencies.",
        "- **compare_telemetry**: always-on for telemetry profiles (not a corpus).",
        "",
        "### RCA by context corpus",
        "",
        "| Context | RCA accuracy | MTTD median |",
        "|---------|--------------|-------------|",
    ]
    for ctx in sorted(by_ctx.keys(), key=lambda x: (x == "0", x)):
        items = by_ctx[ctx]
        rca_s, _ = rate(items, "rca_correct")
        mttd_vals = [x.get("mttd_seconds") for x in items if x.get("mttd_seconds") is not None]
        mttd_med = f"{sorted(mttd_vals)[len(mttd_vals)//2]:.0f}s" if mttd_vals else "n/a"
        lines.append(f"| C{ctx} | {rca_s} | {mttd_med} |")

    lines.extend([
        "",
        "### RCA by fault",
        "",
        "| Fault | RCA accuracy |",
        "|-------|--------------|",
    ])
    for fault in sorted(by_fault.keys()):
        lines.append(f"| {fault} | {rate(by_fault[fault], 'rca_correct')[0]} |")

    lines.extend([
        "",
        "## Per-run detail",
        "",
        "| File | C | Agent | Scenario | Fault | RCA | Remediation |",
        "|------|---|-------|----------|-------|-----|-------------|",
    ])
    for r in sorted(
        runs,
        key=lambda x: (x.get("context_c", ""), x.get("fault_flag", ""), x.get("_file", "")),
    ):
        lines.append(
            f"| {r.get('_file', '-')} | C{r.get('context_c', '?')} | "
            f"{r.get('agent', r.get('classifier', '-'))} | {r.get('scenario', '-')} | "
            f"{r.get('fault_flag', '-')} | {r.get('rca_correct', '-')} | "
            f"{r.get('remediation_correct', '-')} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "MLflow: `http://localhost:5050` — filter by tag `context_c`.",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote report to {report_path} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
