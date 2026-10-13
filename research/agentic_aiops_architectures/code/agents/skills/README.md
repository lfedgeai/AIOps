# Shared AIOps agent skills

Reusable skill packs for **all** detector agents in this repo. They encode the detect → diagnose → remediate → verify loop learned from matrix runs ([docs/Learnings_across_runs_report.md](../../../docs/Learnings_across_runs_report.md)).

Skills stay **application-agnostic**: no hard-coded product topology or service names. Workload identity comes from tools (and optional RAG when `CONTEXT_C` enables it).

## Fairness

Matrix comparisons stay valid only if every agent loads the **same** skill set the same way (via [`prompts.py`](../prompts.py) / shared harness env). Do **not** enable a skill for one model and skip it for another without treating that as a new experiment axis.

These files are the source of truth for skill text. They are loaded for **every** agent via
[`prompts.build_system_prompt_base()`](../prompts.py) / `BASE_SYSTEM_PROMPT` (disable with `AGENT_SKILLS=0`).

## Skills

| Skill | Path | Role |
|-------|------|------|
| Detect | [detect/SKILL.md](detect/SKILL.md) | Decide `detected` from live signals |
| Diagnose | [diagnose/SKILL.md](diagnose/SKILL.md) | Name root-cause workload from tools (+ optional RAG) |
| Remediate | [remediate/SKILL.md](remediate/SKILL.md) | Call write tools — do not only suggest |
| Verify | [verify/SKILL.md](verify/SKILL.md) | Re-check cluster/telemetry before finish |

## Intended composition order

1. Detect (read tools)  
2. Diagnose (correlate live signals; optional RAG)  
3. Remediate (write tools from observed state)  
4. Verify (read tools again)  
5. Final JSON
