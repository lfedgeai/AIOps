#!/usr/bin/env python3
"""Compare two Option-2 harness kickoff windows (detect/RCA/exec/errors).

Example:
  python3.13 scripts/compare_option2_kickoffs.py \\
    --baseline-kickoff "2026-09-22 00:02:00" \\
    --baseline-end "2026-09-23 06:00:00" \\
    --new-kickoff "$(cat out/.matrix_kickoff_aest)"
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_option2_run import load_harness_runs, rate  # noqa: E402


def _ts(s: str) -> float:
    return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S").timestamp()


def _by_agent(runs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        out[str(r.get("agent", "?"))].append(r)
    return out


def _pct(ok: int, n: int) -> float:
    return 100.0 * ok / n if n else 0.0


def _row(label: str, a: list[dict], b: list[dict]) -> str:
    metrics = ("detected", "rca_correct", "remediation_executed", "recovery_verified")
    parts = [f"{label:22}"]
    for key in metrics:
        ao, an, _ = rate(a, key)
        bo, bn, _ = rate(b, key)
        ap, bp = _pct(ao, an), _pct(bo, bn)
        parts.append(f"{ap:5.0f}%→{bp:5.0f}% ({bp - ap:+.0f})")
    ae_a = sum(1 for r in a if r.get("agent_error"))
    ae_b = sum(1 for r in b if r.get("agent_error"))
    parts.append(f"err {ae_a}/{len(a)}→{ae_b}/{len(b)}")
    return " | ".join(parts)


def _filter_agents(runs: list[dict], agents: set[str] | None) -> list[dict]:
    if not agents:
        return runs
    return [r for r in runs if str(r.get("agent")) in agents]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-kickoff", required=True)
    ap.add_argument("--baseline-end", default="", help="Default: new kickoff time")
    ap.add_argument("--new-kickoff", required=True)
    ap.add_argument("--new-end", default="", help="Default: now (omit upper bound)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out")
    ap.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Directory of frozen baseline harness JSON (e.g. out/archive_sep22_option2)",
    )
    ap.add_argument(
        "--agents",
        default="maas_deepseek,maas_qwen3,maas_qwen38-27b,maas_qwen36-27b,maas_granite,maas_glm-53-flash",
        help="Comma-separated agent ids to include (default: Sep22 7-agent lineup)",
    )
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()

    agent_set = {a.strip() for a in args.agents.split(",") if a.strip()} or None

    b0 = _ts(args.baseline_kickoff)
    n0 = _ts(args.new_kickoff)
    b1 = _ts(args.baseline_end) if args.baseline_end else n0
    n1 = _ts(args.new_end) if args.new_end else None

    if args.baseline_dir:
        baseline = load_harness_runs(args.baseline_dir)
    else:
        baseline = load_harness_runs(args.out_dir, mtime_after=b0, mtime_before=b1)
    new = load_harness_runs(args.out_dir, mtime_after=n0, mtime_before=n1)
    baseline = _filter_agents(baseline, agent_set)
    new = _filter_agents(new, agent_set)

    lines = [
        f"# Option 2 kickoff compare",
        "",
        f"- Baseline: `{args.baseline_kickoff}` → `{args.baseline_end or args.new_kickoff}` (n={len(baseline)})",
        f"- New: `{args.new_kickoff}` → `{args.new_end or 'now'}` (n={len(new)})",
        "",
        "Columns: detect / RCA / exec / recovery (baseline%→new% (Δ)) then agent_error counts.",
        "",
        "```",
        _row("ALL", baseline, new),
    ]
    agents = sorted(set(_by_agent(baseline)) | set(_by_agent(new)))
    ba, na = _by_agent(baseline), _by_agent(new)
    for agent in agents:
        lines.append(_row(agent, ba.get(agent, []), na.get(agent, [])))
    lines.append("```")
    lines.append("")
    if len(new) < 336:
        lines.append(f"_New window incomplete ({len(new)}/336). Re-run when matrix finishes._")
        lines.append("")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text, encoding="utf-8")
        print(f"Wrote {args.markdown}", file=sys.stderr)
    return 0 if new else 0


if __name__ == "__main__":
    raise SystemExit(main())
