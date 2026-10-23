#!/usr/bin/env python3
"""Refresh docs/Learnings_across_runs_report.md after an Option-2 matrix completes.

Appends (or replaces) a dated section with detect/RCA/exec/error deltas vs a
frozen baseline archive. Invoked from run_c_matrix_scenario_a_c_permutations.sh.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_option2_run import load_harness_runs, rate  # noqa: E402

LEARNINGS = ROOT / "docs" / "Learnings_across_runs_report.md"
SECTION_MARK = "<!-- AUTO:POST_MATRIX_LEARNINGS -->"
DEFAULT_AGENTS = (
    "rhdp_deepseek-r1-distill-qwen-14b,rhdp_qwen3-14b,rhdp_qwen38-27b,litemaas_qwen3.8-27b,"
    "rhdp_granite-3-2-8b-instruct,prelude_glm-53-flash"
)


def _ts(s: str) -> float:
    return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S").timestamp()


def _pct(ok: int, n: int) -> float:
    return 100.0 * ok / n if n else 0.0


def _metric_line(label: str, base: list, new: list) -> str:
    parts = [f"| {label} |"]
    for key in ("detected", "rca_correct", "remediation_executed", "recovery_verified"):
        bo, bn, _ = rate(base, key)
        no, nn, _ = rate(new, key)
        bp, np_ = _pct(bo, bn), _pct(no, nn)
        parts.append(f" {bp:.0f}% → **{np_:.0f}%** ({np_ - bp:+.0f}) |")
    be = sum(1 for r in base if r.get("agent_error"))
    ne = sum(1 for r in new if r.get("agent_error"))
    parts.append(f" {be}/{len(base)} → **{ne}/{len(new)}** |")
    return "".join(parts)


def build_section(
    *,
    kickoff: str,
    eval_report: str | None,
    baseline_dir: Path,
    out_dir: Path,
    agents: set[str],
    notes: str,
) -> str:
    baseline = [
        r for r in load_harness_runs(baseline_dir) if str(r.get("agent")) in agents
    ]
    new = [
        r
        for r in load_harness_runs(out_dir, mtime_after=_ts(kickoff))
        if str(r.get("agent")) in agents
    ]
    date_label = datetime.strptime(kickoff, "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y")
    eval_link = f"[{Path(eval_report).name}]({Path(eval_report).name})" if eval_report else "_(pending)_"
    lines = [
        SECTION_MARK,
        f"## Post-matrix update — {date_label}",
        "",
        f"**Kickoff:** `{kickoff}` · **New runs:** {len(new)} · **Baseline archive:** `{baseline_dir.relative_to(ROOT) if baseline_dir.is_relative_to(ROOT) else baseline_dir}` (n={len(baseline)})",
        f"**Eval report:** {eval_link}",
        "",
        notes.strip(),
        "",
        "| Scope | Detect | RCA | Exec | Recovery | agent_error |",
        "|-------|-------:|----:|-----:|---------:|------------:|",
        _metric_line("ALL", baseline, new),
    ]
    by_b: dict[str, list] = {}
    by_n: dict[str, list] = {}
    for r in baseline:
        by_b.setdefault(str(r.get("agent")), []).append(r)
    for r in new:
        by_n.setdefault(str(r.get("agent")), []).append(r)
    for agent in sorted(agents):
        lines.append(_metric_line(agent, by_b.get(agent, []), by_n.get(agent, [])))
    lines.extend(
        [
            "",
            "_Auto-appended by `scripts/update_learnings_across_runs_report.py` after Option 2 COMPLETE._",
            "",
        ]
    )
    if len(new) < 300:
        lines.insert(
            6,
            f"> **Incomplete:** only {len(new)}/336 new cells — re-run this script when the matrix finishes.",
        )
        lines.insert(7, "")
    return "\n".join(lines)


def upsert_section(doc: Path, section: str) -> None:
    text = doc.read_text(encoding="utf-8") if doc.is_file() else ""
    # Drop prior auto section(s)
    text = re.sub(
        rf"{re.escape(SECTION_MARK)}\n## Post-matrix update.*?(?=\n## |\n# |\Z)",
        "",
        text,
        flags=re.DOTALL,
    )
    text = text.rstrip() + "\n\n" + section
    # Refresh header "Generated" / in-progress line lightly
    text = re.sub(
        r"\*\*In progress:\*\*.*\n",
        "",
        text,
        count=1,
    )
    today = datetime.now().strftime("%Y-%m-%d")
    text = re.sub(
        r"(\*\*Generated:\*\*)\s*[^\n]+",
        rf"\1 {today} (post-matrix refresh)",
        text,
        count=1,
    )
    doc.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kickoff", required=True, help="New matrix kickoff AEST")
    ap.add_argument(
        "--baseline-dir",
        type=Path,
        default=ROOT / "out" / "archive_sep22_option2",
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out")
    ap.add_argument("--eval-report", default="")
    ap.add_argument("--agents", default=DEFAULT_AGENTS)
    ap.add_argument(
        "--notes",
        default=(
            "Experiment vs Sep 22–23 baseline: shared AIOps skills in system prompt, "
            "CHANGELOG removed from C2 docs corpus, AGENT_TIMEOUT=900 / detection=1200."
        ),
    )
    ap.add_argument("--learnings", type=Path, default=LEARNINGS)
    args = ap.parse_args()
    agents = {a.strip() for a in args.agents.split(",") if a.strip()}
    section = build_section(
        kickoff=args.kickoff,
        eval_report=args.eval_report or None,
        baseline_dir=args.baseline_dir,
        out_dir=args.out_dir,
        agents=agents,
        notes=args.notes,
    )
    upsert_section(args.learnings, section)
    print(f"Updated {args.learnings}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
