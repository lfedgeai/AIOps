#!/usr/bin/env python3
"""Analyze Option 2 matrix: harness JSON vs MLflow, family comparisons, top combos."""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.tools.rag_context import INDEX_VERSION as INDEX_VERSION_TAG

METRICS = (
    "detected",
    "rca_correct",
    "remediation_correct",
    "remediation_executed",
    "recovery_verified",
    "mttd_seconds",
    "mttr_seconds",
    "mttr_verified_seconds",
)

# Primary Option-2 baseline (Sep 25–27) — from EVALUATION_RESULTS_27sep26.md
SEP27_BASELINE = {
    "n": 335,
    "detected": "266/335 (79%)",
    "rca_correct": "196/335 (59%)",
    "remediation_correct": "207/335 (62%)",
    "remediation_executed": "83/335 (25%)",
    "recovery_verified": "73/335 (22%)",
    "mttd_median": "100s",
    "mttr_median": "—",  # not tabulated in that report
    "mttr_verified_median": "—",
    "agent_error": "21/335",
}

# Aug 11 run (5 agents) — from EVALUATION_RESULTS_11Aug26.md + remaining JSON
AUG11_AGENT_MAP = {
    "nemotron-nano-3": "nemotron",
    "maas_deepseek": "deepseek",
    "maas_qwen3": "qwen3",
    "maas_granite": "granite",
    "maas_llama31-70b": "llama31-70b",
}

FAULTS = ("scale_zero", "config_corruption", "kill_pod")
NEW_RUN_KICKOFF_AEST = "2026-08-11 21:58:00"
AUG11_START_MS = int(datetime(2026, 8, 10, 3, 4, tzinfo=timezone.utc).timestamp() * 1000)
AUG11_END_MS = int(datetime(2026, 8, 10, 17, 43, tzinfo=timezone.utc).timestamp() * 1000)

# MLflow note uses script dir names; harness JSON uses approach ids.
# Values are the CANONICAL labels (23 Oct 26 rename: {infra}_{exact-model}).
APPROACH_ALIASES: dict[str, str] = {
    "deepseek_agent": "rhdp_deepseek-r1-distill-qwen-14b",
    "qwen3_agent": "rhdp_qwen3-14b",
    "granite_agent": "rhdp_granite-3-2-8b-instruct",
    "llama31_70b_agent": "rhdp_llama-31-70b-cpu",
    "nemotron_agent": "prelude_nemotron-cascade-2-30b",
    # pre-rename labels (all harness JSONs / MLflow runs before 23 Oct 26)
    "maas_deepseek": "rhdp_deepseek-r1-distill-qwen-14b",
    "maas_qwen3": "rhdp_qwen3-14b",
    "maas_granite": "rhdp_granite-3-2-8b-instruct",
    "maas_llama31-70b": "rhdp_llama-31-70b-cpu",
    "maas_qwen38-27b": "rhdp_qwen38-27b",
    "maas_qwen36-27b": "litemaas_qwen3.8-27b",
    "maas_glm-53-flash": "prelude_glm-53-flash",
    "nemotron-cascade-2": "prelude_nemotron-cascade-2-30b",
    # post-rename labels (identity)
    "rhdp_deepseek-r1-distill-qwen-14b": "rhdp_deepseek-r1-distill-qwen-14b",
    "rhdp_qwen3-14b": "rhdp_qwen3-14b",
    "rhdp_granite-3-2-8b-instruct": "rhdp_granite-3-2-8b-instruct",
    "rhdp_llama-31-70b-cpu": "rhdp_llama-31-70b-cpu",
    "rhdp_qwen38-27b": "rhdp_qwen38-27b",
    "litemaas_qwen3.8-27b": "litemaas_qwen3.8-27b",
    "prelude_glm-53-flash": "prelude_glm-53-flash",
    "prelude_nemotron-cascade-2-30b": "prelude_nemotron-cascade-2-30b",
    # legacy MLflow approach ids with no current equivalent
    "maas_qwen35-9b": "maas_qwen35-9b",
    "qwen35_9b_agent": "maas_qwen35-9b",
    "nemotron-nano-3": "nemotron-nano-3",
    "qwen38_27b_agent": "rhdp_qwen38-27b",
    "qwen36_27b_agent": "litemaas_qwen3.8-27b",
    "glm_53_flash_agent": "prelude_glm-53-flash",
}


def canonical_agent(label: str) -> str:
    """Map any agent id (pre- or post-rename) to the canonical {infra}_{model} label."""
    return APPROACH_ALIASES.get(label, label)


def _ctx_file_to_tag(ctx: str) -> str:
    """Filename C1-2-4 → MLflow tag 1,2,4."""
    if not ctx or ctx == "0":
        return "0"
    return ctx.replace("-", ",")


def _ctx_tag_to_file(ctx: str) -> str:
    return (ctx or "0").replace(",", "-")


def _parse_filename(path: Path) -> dict[str, str] | None:
    m = re.match(r"context_C([^/]+)_sa_(.+)\.json$", path.name)
    if not m:
        return None
    ctx, rest = m.group(1), m.group(2)
    for fault in FAULTS:
        prefix = fault + "_"
        if rest.startswith(prefix):
            return {"context_c": ctx, "fault": fault, "agent": canonical_agent(rest[len(prefix) :])}
    return None


def load_harness_runs(
    out_dir: Path,
    *,
    mtime_after: float | None = None,
    mtime_before: float | None = None,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("context_C*_sa_*_*.json")):
        meta = _parse_filename(path)
        if not meta:
            continue
        mt = path.stat().st_mtime
        if mtime_after is not None and mt < mtime_after:
            continue
        if mtime_before is not None and mt >= mtime_before:
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "runs" in data:
            rec = data["runs"][0] if data["runs"] else data
        else:
            rec = data
        rec = dict(rec)
        rec.update(meta)
        rec["_file"] = path.name
        rec["_mtime"] = mt
        runs.append(rec)
    return runs


def _bool_metric(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return None


def rate(runs: list[dict], key: str) -> tuple[int, int, str]:
    vals = [r.get(key) for r in runs if r.get(key) is not None]
    if not vals:
        return 0, 0, "n/a"
    ok = sum(1 for v in vals if v)
    pct = 100 * ok / len(vals)
    return ok, len(vals), f"{ok}/{len(vals)} ({pct:.0f}%)"


def median_mttd(runs: list[dict]) -> str:
    vals = [r["mttd_seconds"] for r in runs if r.get("detected") and r.get("mttd_seconds") is not None]
    if not vals:
        return "—"
    return f"{statistics.median(vals):.0f}s"


def median_mttr(runs: list[dict], *, key: str = "mttr_seconds") -> str:
    """Median MTTR over cells where the field is present (typically remediated / verified)."""
    vals = [float(r[key]) for r in runs if r.get(key) is not None]
    if not vals:
        return "—"
    return f"{statistics.median(vals):.0f}s"


def median_mttr_n(runs: list[dict], *, key: str = "mttr_seconds") -> tuple[str, int]:
    vals = [float(r[key]) for r in runs if r.get(key) is not None]
    if not vals:
        return "—", 0
    return f"{statistics.median(vals):.0f}s", len(vals)


def ctx_label(ctx: Any) -> str:
    """Render a CONTEXT_C set in unambiguous comma notation: '1-4' (dash-joined) -> 'C1,4'.

    The dash form in filenames ('C2-4') reads like a range; it is the SET {2,4}.
    'C2,3,4' is the set {2,3,4}.
    """
    parts = [p for p in str(ctx).split("-") if p]
    return "C" + ",".join(parts)


def combo_key(r: dict) -> tuple[str, str, str]:
    return (str(r.get("context_c", "")), str(r.get("agent", "")), str(r.get("fault", "")))


def top_combos(runs: list[dict], *, min_n: int = 1) -> list[tuple[tuple, int, int, int]]:
    """Return (combo, detect_ok, rca_ok, exec_ok) sorted by rca then exec."""
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        by[combo_key(r)].append(r)
    rows = []
    for k, items in by.items():
        if len(items) < min_n:
            continue
        d = sum(1 for x in items if x.get("detected"))
        rca = sum(1 for x in items if x.get("rca_correct"))
        ex = sum(1 for x in items if x.get("remediation_executed"))
        rows.append((k, d, rca, ex, len(items)))
    rows.sort(key=lambda x: (-x[2], -x[3], -x[1]))
    return rows


def load_mlflow_runs(start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri("http://127.0.0.1:5050")
    client = MlflowClient()
    exp = client.get_experiment_by_name("agentic_aiops_mttd_mttr")
    if not exp:
        return []
    filter_str = f"attributes.start_time >= {start_ms} AND attributes.start_time < {end_ms}"
    out: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        page = client.search_runs(
            [exp.experiment_id],
            filter_string=filter_str,
            max_results=500,
            page_token=page_token,
            order_by=["start_time ASC"],
        )
        if not page:
            break
        for run in page:
            note = run.data.tags.get("mlflow.note.content") or ""
            approach_m = re.search(r"\*\*Approach\*\*:\s*(\S+)", note)
            raw_approach = approach_m.group(1) if approach_m else None
            if not raw_approach:
                name = run.info.run_name or ""
                parts = name.split("_")
                if len(parts) >= 2:
                    raw_approach = parts[1]
            approach = APPROACH_ALIASES.get(raw_approach or "", raw_approach)
            ctx_tag = run.data.tags.get("context_c") or "0"
            fault = "?"
            for f in FAULTS:
                if f in (run.info.run_name or ""):
                    fault = f
                    break
            rec = {
                "run_id": (run.info.run_name or "").split("_")[0],
                "run_name": run.info.run_name,
                "approach": approach,
                "context_c": _ctx_tag_to_file(ctx_tag),
                "context_c_tag": ctx_tag,
                "fault": fault,
                "start_time": run.info.start_time,
            }
            for m in METRICS:
                if m in run.data.metrics:
                    rec[m] = run.data.metrics[m]
            out.append(rec)
        page_token = page.token if hasattr(page, "token") else None
        if not page_token:
            break
    return out


def _agent_id(r: dict) -> str:
    return canonical_agent(str(r.get("approach") or r.get("agent") or ""))


# Agent labels are canonical since the 23 Oct 26 rename ({infra}_{exact-model});
# APPROACH_ALIASES maps pre-rename ids found in historical JSONs/MLflow run names.
def _agent_label(r: dict) -> str:
    return _agent_id(r)


def _filter_agent(runs: list[dict], agent_id: str) -> list[dict]:
    return [r for r in runs if _agent_id(r) == agent_id]


def mlflow_from_run_name(name: str) -> dict[str, str]:
    """Parse 20260812111134_maas_llama31-70b_scenarioa_kill_pod_cart"""
    parts = (name or "").split("_")
    out: dict[str, str] = {}
    if len(parts) >= 5:
        out["approach"] = parts[1]
        out["fault"] = parts[3] if parts[2].startswith("scenario") else parts[2]
    return out


def validate_mlflow(harness: list[dict], mlflow_runs: list[dict]) -> dict[str, Any]:
    """Match harness JSON to MLflow by run_id prefix in filename pattern."""
    by_id: dict[str, dict] = {}
    for m in mlflow_runs:
        rid = m.get("run_name", "").split("_")[0] if m.get("run_name") else ""
        if rid:
            by_id[rid] = m

    mismatches: list[str] = []
    matched = 0
    for h in harness:
        rid = str(h.get("run_id", ""))
        m = by_id.get(rid)
        if not m:
            continue
        matched += 1
        for key in ("detected", "rca_correct", "remediation_executed", "remediation_correct"):
            hv = _bool_metric(h.get(key))
            mv = _bool_metric(m.get(key))
            if hv is not None and mv is not None and bool(hv) != bool(mv):
                mismatches.append(f"{rid} {key}: harness={hv} mlflow={mv}")
    return {"matched": matched, "mismatches": mismatches, "mlflow_count": len(mlflow_runs)}


def agent_family(agent: str) -> str:
    a = agent.lower()
    if "nemotron" in a:
        return "nemotron"
    if "qwen" in a:
        return "qwen"
    if "kimi" in a:
        return "kimi"
    if "deepseek" in a:
        return "deepseek"
    if "granite" in a:
        return "granite"
    if "llama" in a:
        return "llama"
    return "other"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _pct(ok: int, n: int) -> str:
    if n <= 0:
        return "—"
    return f"{100 * ok / n:.0f}%"


def _headline_takeaways(new_runs: list[dict], by_agent: dict[str, list[dict]]) -> list[str]:
    """Data-driven bullets; avoid hard-coded Aug-era lineup claims."""
    n = len(new_runs)
    d_ok, _, _ = rate(new_runs, "detected")
    ex_ok, _, _ = rate(new_runs, "remediation_executed")
    bullets: list[str] = [
        f"**Detection vs execution:** detect **{_pct(d_ok, n)}** ({d_ok}/{n}); "
        f"remediation executed **{_pct(ex_ok, n)}** ({ex_ok}/{n}) — most models still stop at diagnosis.",
    ]

    def agent_stats(aid: str) -> tuple[int, int, int, int, int]:
        items = by_agent.get(aid, [])
        return (
            rate(items, "detected")[0],
            rate(items, "rca_correct")[0],
            rate(items, "remediation_executed")[0],
            sum(1 for r in items if r.get("agent_error")),
            len(items),
        )

    deepseek = agent_stats("rhdp_deepseek-r1-distill-qwen-14b")
    granite = agent_stats("rhdp_granite-3-2-8b-instruct")
    if deepseek[4] or granite[4]:
        bullets.append(
            f"**RCA vs execute split:** "
            f"**DeepSeek** RCA **{deepseek[1]}/{deepseek[4]}** / exec **{deepseek[2]}/{deepseek[4]}**; "
            f"**Granite** RCA **{granite[1]}/{granite[4]}** / exec **{granite[2]}/{granite[4]}**."
        )

    executors = sorted(
        (
            (aid, agent_stats(aid)[2], agent_stats(aid)[4])
            for aid in by_agent
            if agent_stats(aid)[2] > 0
        ),
        key=lambda x: -x[1],
    )[:4]
    if executors:
        bits = ", ".join(f"**{AGENT_DISPLAY_LABELS.get(aid, aid)}** {ex}/{nn}" for aid, ex, nn in executors)
        bullets.append(f"**Primary executors this run:** {bits}.")

    q36 = agent_stats("litemaas_qwen3.8-27b")
    if q36[4]:
        bullets.append(
            f"**qwen3.8-27b (maas-litemaas):** detect/RCA/exec **{q36[0]}/{q36[4]}** with **{q36[3]}/{q36[4]}** "
            "`agent_error` (when healthy, detect usually co-occurs with RCA+exec)."
        )

    llama = agent_stats("maas_llama31-70b")
    if llama[4]:
        bullets.append(
            f"**Llama:** **rhdp_llama-31-70b-cpu** detect **{llama[0]}/{llama[4]}**, "
            f"`agent_error` **{llama[3]}/{llama[4]}** — keep out of default lineup until fixed."
        )

    glm = agent_stats("prelude_glm-53-flash")
    if glm[4]:
        bullets.append(
            f"**glm-53-flash:** detect **{glm[0]}/{glm[4]}**, RCA **{glm[1]}/{glm[4]}**, "
            f"exec **{glm[2]}/{glm[4]}** (MTTD med {median_mttd(by_agent.get('prelude_glm-53-flash', []))})."
        )

    if not any("cascade" in str(a) or "nemotron" in str(a) for a in by_agent):
        bullets.append(
            f"**Nemotron / Kimi:** not in this batch "
            "(omit from headlines until preflight-green)."
        )
    return [f"{i}. {b}" for i, b in enumerate(bullets, 1)]


def build_report(
    new_runs: list[dict],
    aug11_runs: list[dict],
    mlflow_val: dict[str, Any],
    aug11_mlflow: list[dict],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M AEST")
    n = len(new_runs)
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for r in new_runs:
        by_agent[_agent_id(r)].append(r)
    n_agents = len(by_agent)
    # The two qwen slots run the same model (Qwen 3.8 27B) on different gateways — see
    # AGENT_DISPLAY_LABELS / the slot-identity finding. Count distinct models accordingly.
    n_distinct_models = n_agents - (
        1 if {"litemaas_qwen3.8-27b", "rhdp_qwen38-27b"} <= set(by_agent) else 0
    )
    n_contexts = len({str(r.get("context_c")) for r in new_runs})
    agents_per = sorted(len(v) for v in by_agent.values())
    per_agent = agents_per[0] if agents_per and agents_per[0] == agents_per[-1] else None
    lineup = ", ".join(f"`{a}`" for a in sorted(by_agent))
    lineup_labeled = ", ".join(f"`{AGENT_DISPLAY_LABELS.get(a, a)}`" for a in sorted(by_agent))

    d_ok, d_n, d_s = rate(new_runs, "detected")
    r_ok, r_n, r_s = rate(new_runs, "rca_correct")
    rem_ok, rem_n, rem_s = rate(new_runs, "remediation_correct")
    ex_ok, ex_n, ex_s = rate(new_runs, "remediation_executed")
    rec_ok, _, rec_s = rate(new_runs, "recovery_verified")
    agent_errors = sum(1 for r in new_runs if r.get("agent_error"))

    lines: list[str] = [
        f"# Evaluation Results — Option 2 matrix ({n_agents} agents)",
        "",
        f"**Generated:** {now}  ",
        f"**Design:** Option 2 — scenario **A** × all CONTEXT_C subsets ({n_contexts}) × "
        f"**{n_agents} agents** × 3 faults  ",
        f"**Runs:** **{n}/{n}** completed  ",
        "**Variant:** `cart` · **Forced RAG:** on for C≠0  ",
        "**Tracking:** local MLflow (`http://127.0.0.1:5050`), experiment `agentic_aiops_mttd_mttr`  ",
        "**Primary baseline:** [EVALUATION_RESULTS_27sep26.md](EVALUATION_RESULTS_27sep26.md) (Option 2, skills + no-CHANGELOG)  ",
        "**Run notes:** [RUN_NOTES_07Oct26.md](RUN_NOTES_07Oct26.md)  ",
        "",
        f"> **Lineup (this run):** {lineup_labeled or '(none)'}.  ",
        "> **Comparability:** Shared prompts (`code/agents/prompts.py`) and API preflight. "
        "Primary Δ table uses Sep 25–27; older Aug-11 (different design size) appears only in legacy combo sections.",
        "",
        "---",
        "",
        "## What this report measures",
        "",
        f"Each of the **{n}** cells is one harness trial: inject a Kubernetes fault on a target workload "
        "(scenario A / cart variant), give the agent a time window with telemetry/K8s tools "
        "(and optional RAG corpora), then score the outcome.",
        "",
        "| Axis | Values |",
        "|------|--------|",
        "| Scenario | A (single-service fault) |",
        f"| CONTEXT_C | {n_contexts} labels (`0` + subsets of {{1,2,3,4}}) — see corpus definitions below |",
        f"| Agent | {n_agents} slots ({n_distinct_models} distinct models — two slots run Qwen3.8-27B on different gateways) |",
        "| Fault | `scale_zero`, `config_corruption`, `kill_pod` |",
        "",
        "**Context corpora (C1–C4).** A CONTEXT_C label is the *set* of RAG corpora enabled for that",
        "cell (e.g. `C1,4` = corpora 1 and 4; `C0` = none). Each corpus is a slice of the git-SHA-pinned",
        "OpenTelemetry demo repo, chunked at 900 chars / 120 overlap and exposed to the agent as its own",
        "`rag_search_*` function tool (queried in a forced round-0 phase per enabled corpus; retrieval",
        "mechanics in the retriever note under By context):",
        "",
        "| Context | Corpus | Content (file globs in the pinned repo) |",
        "|---------|--------|------------------------------------------|",
        "| C0 | — none | No RAG tools offered; the agent relies on telemetry + K8s tools only |",
        "| C1 | `source` | Application source code: `src/**/*.{py,go,ts,js,rs,java,cs}` |",
        "| C2 | `docs` | Product documentation: `README.md`, `docs/**/*.md`, `CONTRIBUTING.md`, `.github/**/*.md` (CHANGELOG.md deliberately excluded — noise for harness-injected K8s faults) |",
        "| C3 | `architecture` | Service-topology material: README/compose plus files matching `*architecture*`/`*diagram*`/service-map, mermaid/flowchart blocks, and chunks mentioning service names — a chunk filter keeps only topology-relevant pieces |",
        "| C4 | `dependencies` | Deployment/dependency configuration: `compose*.yaml`, `kubernetes/**`, `**/Dockerfile*`, otel-collector / prometheus / jaeger configs |",
        "",
        "",
        "| Metric | Meaning |",
        "|--------|---------|",
        "| **Detect** | Agent recognized a fault within the detection window |",
        "| **RCA correct** | Agent identified the right failing component / root cause |",
        "| **Remediation correct (OR)** | Appropriate fix proposed (judged), whether or not executed |",
        "| **Remediation executed** | Agent performed a remediation action |",
        "| **Recovery verified** | Target workload healthy after the agent’s actions |",
        "| **MTTD** | Seconds from fault → detection (median over *detected* runs with `mttd_seconds`) |",
        "| **MTTR** | Seconds from fault → first successful remediation write (`mttr_seconds`) |",
        "| **MTTR verified** | Seconds from fault → recovery verified (`mttr_verified_seconds`) |",
        "| **`agent_error`** | Harness recorded an agent/API failure for that cell |",
        "",
        "**How to read rates:** `successes/trials`. "
        + (
            f"Per-agent rows are **{per_agent}** trials. "
            if per_agent
            else "Per-agent row counts vary. "
        )
        + f"Per-context rows are **{n_agents * 3}** trials ({n_agents} agents × 3 faults) when the matrix is complete. "
        "“Full win” = detect ∧ RCA ∧ exec on the same cell. "
        "MTTR medians use only cells where the field is present (n shown in model×context tables).",
        "",
        "---",
        "",
        "## Executive summary",
        "",
        f"| Metric | This run (n={n}) | Sep 25–27 (n={SEP27_BASELINE['n']}) |",
        "|--------|------------------|-------------------------------------|",
        f"| Detection | **{d_s}** | {SEP27_BASELINE['detected']} |",
        f"| RCA correct | **{r_s}** | {SEP27_BASELINE['rca_correct']} |",
        f"| Remediation correct (OR) | **{rem_s}** | {SEP27_BASELINE['remediation_correct']} |",
        f"| Remediation executed | **{ex_s}** | {SEP27_BASELINE['remediation_executed']} |",
        f"| Recovery verified | **{rec_s}** | {SEP27_BASELINE['recovery_verified']} |",
        f"| MTTD median (detected) | **{median_mttd(new_runs)}** | {SEP27_BASELINE['mttd_median']} |",
        f"| MTTR median | **{median_mttr(new_runs)}** | {SEP27_BASELINE['mttr_median']} |",
        f"| MTTR verified median | **{median_mttr(new_runs, key='mttr_verified_seconds')}** | {SEP27_BASELINE['mttr_verified_median']} |",
        f"| `agent_error` | **{agent_errors}/{n}** | {SEP27_BASELINE['agent_error']} |",
        "",
        "_Primary baseline = [EVALUATION_RESULTS_27sep26.md](EVALUATION_RESULTS_27sep26.md) (same Option 2 design). "
        "Older Aug-11 (5-agent / 240-cell) figures remain in family/combo sections where noted._",
        "",
        "### Headline takeaways",
        "",
    ]
    lines.extend(_headline_takeaways(new_runs, by_agent))
    lines.append("")

    agent_rows = []
    for agent in sorted(by_agent.keys()):
        items = by_agent[agent]
        mttr_s, mttr_n = median_mttr_n(items)
        mttr_v_s, mttr_v_n = median_mttr_n(items, key="mttr_verified_seconds")
        agent_rows.append([
            f"`{AGENT_DISPLAY_LABELS.get(agent, agent)}`",
            rate(items, "detected")[2],
            rate(items, "rca_correct")[2],
            rate(items, "remediation_executed")[2],
            median_mttd(items),
            f"{mttr_s} (n={mttr_n})" if mttr_n else "—",
            f"{mttr_v_s} (n={mttr_v_n})" if mttr_v_n else "—",
        ])
    agent_hdr = f"## By agent ({per_agent} runs each)" if per_agent else "## By agent"
    lines.append(agent_hdr)
    lines.append("")
    lines.append(
        render_table(
            ["Agent", "Detect", "RCA", "Exec", "MTTD med", "MTTR med", "MTTR verified med"],
            agent_rows,
        )
    )
    lines.append("")

    lines.extend(["## Model family comparison", ""])
    cascade = [r for r in new_runs if "cascade" in str(r.get("approach", r.get("agent", "")))]
    nano = [r for r in aug11_runs if "nemotron" in str(r.get("agent", ""))]
    if cascade or nano:
        lines.append("### Nemotron")
        lines.append("")
        nem_rows = []
        if nano:
            nem_rows.append([
                "`nemotron-nano-3` (Aug)",
                rate(nano, "detected")[2],
                rate(nano, "rca_correct")[2],
                rate(nano, "remediation_executed")[2],
                "prior batch",
            ])
        if cascade:
            nem_rows.append([
                "`nemotron-cascade-2` (this run)",
                rate(cascade, "detected")[2],
                rate(cascade, "rca_correct")[2],
                rate(cascade, "remediation_executed")[2],
                "in lineup",
            ])
        else:
            nem_rows.append(["`nemotron-cascade-2` (this run)", "n/a", "n/a", "n/a", "not in this batch"])
        lines.append(render_table(["Lineup", "Detect", "RCA", "Exec", "Notes"], nem_rows))
        lines.append("")
    else:
        lines.append(
            "Nemotron was **not** in this matrix. See prior Option 2 reports for nano/cascade history."
        )
        lines.append("")

    lines.append("### Qwen family (`qwen3-14b` vs Qwen3.8-27B × two gateways)")
    lines.append("")
    qwen_specs = [
        ("qwen3 (RHDP)", "rhdp_qwen3-14b"),
        ("qwen3.8-27b @ rhdp", "rhdp_qwen38-27b"),
        ("qwen3.8-27b @ litemaas", "litemaas_qwen3.8-27b"),
    ]
    qwen_rows = []
    for label, agent_id in qwen_specs:
        items = _filter_agent(new_runs, agent_id)
        if not items:
            continue
        mttr_s, mttr_n = median_mttr_n(items)
        mttr_v_s, mttr_v_n = median_mttr_n(items, key="mttr_verified_seconds")
        qwen_rows.append([
            label,
            rate(items, "detected")[2],
            rate(items, "rca_correct")[2],
            rate(items, "remediation_executed")[2],
            median_mttd(items),
            f"{mttr_s} (n={mttr_n})" if mttr_n else "—",
            f"{mttr_v_s} (n={mttr_v_n})" if mttr_v_n else "—",
        ])
    aug_qwen3 = _filter_agent(aug11_mlflow, "maas_qwen3")
    if aug_qwen3:
        qwen_rows.append([
            "qwen3 (Aug 11 MLflow)",
            rate(aug_qwen3, "detected")[2],
            rate(aug_qwen3, "rca_correct")[2],
            rate(aug_qwen3, "remediation_executed")[2],
            median_mttd(aug_qwen3),
            "—",
            "—",
        ])
    if qwen_rows:
        lines.append(
            render_table(
                ["Model", "Detect", "RCA", "Exec", "MTTD med", "MTTR med", "MTTR verified med"],
                qwen_rows,
            )
        )
        lines.append("")

    lines.extend(["## Top performing combinations", ""])
    lines.append("### This run — best cells (detect ∧ RCA ∧ exec)")
    lines.append("")
    best = [
        r for r in new_runs
        if r.get("detected") and r.get("rca_correct") and r.get("remediation_executed")
    ]
    fw_by: dict[str, int] = defaultdict(int)
    for r in best:
        fw_by[_agent_id(r)] += 1
    fw_bits = ", ".join(f"{AGENT_DISPLAY_LABELS.get(a, a)} {c}" for a, c in sorted(fw_by.items(), key=lambda x: -x[1]))
    lines.append(f"**Full wins:** **{len(best)}/{n}**" + (f" ({fw_bits})" if fw_bits else ""))
    lines.append("")
    if best:
        lines.append("| Context | Agent | Fault | MTTD | MTTR | MTTR verified |")
        lines.append("|---------|-------|-------|------|------|---------------|")
        for r in sorted(best, key=lambda x: x.get("mttd_seconds") or 9999)[:20]:
            mttr = f"{r['mttr_seconds']:.0f}s" if r.get("mttr_seconds") is not None else "—"
            mttr_v = (
                f"{r['mttr_verified_seconds']:.0f}s"
                if r.get("mttr_verified_seconds") is not None
                else "—"
            )
            mttd = f"{r['mttd_seconds']:.0f}s" if r.get("mttd_seconds") is not None else "—"
            lines.append(
                f"| {ctx_label(r['context_c'])} | `{_agent_label(r)}` | {r['fault']} | {mttd} | {mttr} | {mttr_v} |"
            )
    lines.append("")

    lines.append("### Top CONTEXT_C × agent (RCA rate, 3 faults per cell)")
    lines.append("")
    by_ca: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in new_runs:
        by_ca[(str(r["context_c"]), _agent_id(r))].append(r)
    top_ca = []
    for (ctx, ag), items in by_ca.items():
        if len(items) != 3:
            continue
        rca = sum(1 for x in items if x.get("rca_correct"))
        ex = sum(1 for x in items if x.get("remediation_executed"))
        det = sum(1 for x in items if x.get("detected"))
        top_ca.append((rca, ex, det, ctx, ag))
    top_ca.sort(reverse=True)
    lines.append("| Rank | Context | Agent | Detect | RCA | Exec |")
    lines.append("|------|---------|-------|--------|-----|------|")
    for i, (rca, ex, det, ctx, ag) in enumerate(top_ca[:15], 1):
        lines.append(f"| {i} | {ctx_label(ctx)} | `{AGENT_DISPLAY_LABELS.get(ag, ag)}` | {det}/3 | {rca}/3 | {ex}/3 |")
    lines.append("")

    lines.append("### Aug 11 top combos (from prior report — working agents only)")
    lines.append("")
    lines.append("| Context | Best RCA agents (Aug) | RCA | Exec highlight |")
    lines.append("|---------|----------------------|-----|----------------|")
    lines.append("| C3 | DeepSeek, Granite | 8/15 each | Qwen only executor |")
    lines.append("| C{1,4} | DeepSeek | 8/15 | — |")
    lines.append("| C0 | Granite, DeepSeek | weak RCA (3/15 C0) | — |")
    lines.append("| Full set C{1,2,3,4} | DeepSeek | 3/3 detect+RCA (Jul Option2) | 0 exec |")
    lines.append("")

    lines.append(
        "### Context × agent: this run vs Aug 11 legacy batch (comparable agents: deepseek, qwen3, granite)"
    )
    lines.append("")
    lines.append(
        "> Every (context × agent) cell for the three agents present in both batches — "
        "**not a \"top\" ranking**. Aug 11 was a different design size (see comparability note "
        "at the top of this report); it is shown for legacy continuity only. "
        "Rows are ordered by best current RCA, then current exec."
    )
    lines.append("")
    compare_agents = [
        ("rhdp_deepseek-r1-distill-qwen-14b", "deepseek"),
        ("rhdp_qwen3-14b", "qwen3"),
        ("rhdp_granite-3-2-8b-instruct", "granite"),
    ]
    ctx_order = ["0", "1", "2", "3", "4", "1-4", "2-4", "1-2-3-4", "2-3", "2-3-4"]
    comp_rows = []
    for ctx in ctx_order:
        for agent_id, label in compare_agents:
            new_items = [
                r for r in new_runs
                if str(r.get("context_c")) == ctx and _agent_id(r) == agent_id
            ]
            aug_items = [
                r for r in aug11_mlflow
                if str(r.get("context_c")) == ctx and _agent_id(r) == agent_id
            ]
            if not new_items and not aug_items:
                continue
            new_rca = sum(1 for x in new_items if x.get("rca_correct"))
            new_ex = sum(1 for x in new_items if x.get("remediation_executed"))
            aug_rca = sum(1 for x in aug_items if x.get("rca_correct"))
            aug_ex = sum(1 for x in aug_items if x.get("remediation_executed"))
            comp_rows.append([
                ctx_label(ctx),
                label,
                (new_rca, len(new_items)),
                (new_ex, len(new_items)),
                f"{aug_rca}/{len(aug_items)} RCA, {aug_ex}/{len(aug_items)} exec" if aug_items else "—",
            ])
    # Order by best current RCA, then current exec (as the heading implies), not by context.
    comp_rows.sort(key=lambda row: (-(row[2][0] / row[2][1] if row[2][1] else 0),
                                    -(row[3][0] / row[3][1] if row[3][1] else 0),
                                    row[0], row[1]))
    comp_rows = [
        [r[0], r[1],
         f"{r[2][0]}/{r[2][1]} RCA" if r[2][1] else "—",
         f"{r[3][0]}/{r[3][1]} exec" if r[3][1] else "—",
         r[4]]
        for r in comp_rows
    ]
    lines.append(render_table(["Context", "Agent", "Now RCA", "Now exec", "Aug 11 (MLflow)"], comp_rows))
    lines.append("")

    lines.extend([
        "## MLflow validation",
        "",
        f"- Harness JSON runs (this batch): **{len(new_runs)}**",
        f"- MLflow runs in window: **{mlflow_val['mlflow_count']}**",
        f"- Matched by `run_id`: **{mlflow_val['matched']}**",
        f"- Metric mismatches: **{len(mlflow_val['mismatches'])}**",
        "",
    ])
    if mlflow_val["mismatches"][:10]:
        lines.append("Sample mismatches:")
        for m in mlflow_val["mismatches"][:10]:
            lines.append(f"- {m}")
        lines.append("")

    per_ctx = n_agents * 3
    lines.append(f"## By context ({per_ctx} runs each = {n_agents} agents × 3 faults)")
    lines.append("")
    lines.append(
        f"> **Retriever note:** CONTEXT_C corpora (C1–C4) are served by a **lexical TF-IDF RAG** "
        "(sklearn `TfidfVectorizer`, unigrams+bigrams, 50k features, cosine top-k=5) over the "
        "git-SHA-pinned OpenTelemetry demo repo, chunked at 900 chars / 120 overlap "
        f"(`code/tools/rag_context.py`, `INDEX_VERSION={INDEX_VERSION_TAG}`). "
        "No embeddings or vector store are involved: retrieval matches keyword overlap only, "
        "and the agent formulates its own queries (RAG-as-tool with a forced round-0 phase per "
        "enabled corpus). Interpret by-context deltas with this in mind."
    )
    lines.append("")
    by_ctx: dict[str, list[dict]] = defaultdict(list)
    for r in new_runs:
        by_ctx[str(r["context_c"])].append(r)
    ctx_rows = []
    for ctx in sorted(by_ctx.keys(), key=lambda x: (len(x), x)):
        items = by_ctx[ctx]
        mttr_s, mttr_n = median_mttr_n(items)
        mttr_v_s, mttr_v_n = median_mttr_n(items, key="mttr_verified_seconds")
        ctx_rows.append([
            ctx_label(ctx),
            rate(items, "detected")[2],
            rate(items, "rca_correct")[2],
            rate(items, "remediation_executed")[2],
            median_mttd(items),
            f"{mttr_s} (n={mttr_n})" if mttr_n else "—",
            f"{mttr_v_s} (n={mttr_v_n})" if mttr_v_n else "—",
        ])
    lines.append(
        render_table(
            ["Context", "Detect", "RCA", "Exec", "MTTD med", "MTTR med", "MTTR verified med"],
            ctx_rows,
        )
    )
    lines.append("")

    # Full agent × context (3 faults aggregated per cell)
    lines.append("## By agent × context (3 faults each)")
    lines.append("")
    lines.append(
        "One row per (agent, CONTEXT_C). Rates are over the 3 faults; "
        "MTTR / MTTR verified medians use cells where the field is present."
    )
    lines.append("")
    by_ac: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in new_runs:
        by_ac[(_agent_id(r), str(r["context_c"]))].append(r)
    ac_rows = []
    for (ag, ctx) in sorted(by_ac.keys(), key=lambda x: (x[0], len(x[1]), x[1])):
        items = by_ac[(ag, ctx)]
        mttr_s, mttr_n = median_mttr_n(items)
        mttr_v_s, mttr_v_n = median_mttr_n(items, key="mttr_verified_seconds")
        ac_rows.append([
            f"`{AGENT_DISPLAY_LABELS.get(ag, ag)}`",
            ctx_label(ctx),
            f"{sum(1 for x in items if x.get('detected'))}/{len(items)}",
            f"{sum(1 for x in items if x.get('rca_correct'))}/{len(items)}",
            f"{sum(1 for x in items if x.get('remediation_executed'))}/{len(items)}",
            median_mttd(items),
            f"{mttr_s} (n={mttr_n})" if mttr_n else "—",
            f"{mttr_v_s} (n={mttr_v_n})" if mttr_v_n else "—",
        ])
    lines.append(
        render_table(
            ["Agent", "Context", "Detect", "RCA", "Exec", "MTTD med", "MTTR med", "MTTR verified med"],
            ac_rows,
        )
    )
    lines.append("")

    err_agents: dict[str, int] = defaultdict(int)
    for r in new_runs:
        if r.get("agent_error"):
            err_agents[_agent_id(r)] += 1
    if err_agents:
        lines.append("## Agent errors")
        lines.append("")
        for ag, cnt in sorted(err_agents.items(), key=lambda x: -x[1]):
            denom = len(by_agent.get(ag, [])) or per_agent or "?"
            lines.append(f"- `{AGENT_DISPLAY_LABELS.get(ag, ag)}`: {cnt}/{denom}")
        lines.append("")

    exec_ranked = sorted(
        ((aid, rate(items, "remediation_executed")[0], len(items)) for aid, items in by_agent.items()),
        key=lambda x: -x[1],
    )
    rca_ranked = sorted(
        ((aid, rate(items, "rca_correct")[0], len(items)) for aid, items in by_agent.items()),
        key=lambda x: -x[1],
    )
    top_exec = [a for a, ex, _ in exec_ranked if ex > 0][:2]
    top_rca = [a for a, _, _ in rca_ranked[:2]]
    lines.extend([
        "---",
        "",
        "## Recommendations",
        "",
        f"1. **Default RCA pair:** {' + '.join(f'**{AGENT_DISPLAY_LABELS.get(a, a)}**' for a in top_rca) or 'top RCA agents'} for diagnosis"
        + (
            f"; pair with {' / '.join(f'**{AGENT_DISPLAY_LABELS.get(a, a)}**' for a in top_exec)} when execution is required."
            if top_exec
            else "."
        ),
        f"2. **Executors this run:** {', '.join(f'**{AGENT_DISPLAY_LABELS.get(a, a)}**' for a in top_exec) or 'none with exec>0'}.",
        "3. **Llama:** Dropped from default lineup after 0% detect (optional agent code remains).",
        "4. **Context:** Prefer high-RCA contexts from the by-context / agent×context tables.",
        "5. **Re-add omitted models** (Nemotron/Kimi) only after MaaS preflight is green.",
        "",
    ])
    return "\n".join(lines)



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="docs/EVALUATION_RESULTS_12Aug26.md")
    ap.add_argument("--kickoff", default=NEW_RUN_KICKOFF_AEST)
    args = ap.parse_args()

    kickoff = datetime.strptime(args.kickoff, "%Y-%m-%d %H:%M:%S").timestamp()
    out_dir = ROOT / "out"

    new_runs = load_harness_runs(out_dir, mtime_after=kickoff)
    aug11_runs = load_harness_runs(out_dir, mtime_before=kickoff)

    new_end_ms = int(datetime.now().timestamp() * 1000) + 60_000
    new_start_ms = int(kickoff * 1000)
    mlflow_new = load_mlflow_runs(new_start_ms, new_end_ms)
    mlflow_aug = load_mlflow_runs(AUG11_START_MS, AUG11_END_MS)

    val = validate_mlflow(new_runs, mlflow_new)

    report = build_report(new_runs, aug11_runs, val, mlflow_aug)
    report_path = ROOT / args.report
    report_path.write_text(report)
    print(f"Wrote {report_path}")
    print(f"new_runs={len(new_runs)} aug11_json_remaining={len(aug11_runs)} mlflow_new={len(mlflow_new)} mismatches={len(val['mismatches'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
