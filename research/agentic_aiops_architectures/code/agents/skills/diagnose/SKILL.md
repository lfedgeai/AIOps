---
name: aiops-diagnose
description: >-
  Diagnose root-cause workload using live cluster/telemetry and optional RAG.
  Use after detect and before remediation for all AIOps agents.
---

# AIOps — Diagnose

## Goal

Set `suggested_root_cause` to the failing **workload** name as observed in tools
(deployment, service, or pod label) — not a vague symptom. Do not assume a fixed
application inventory.

## How to use context (RAG)

When CONTEXT_C ≠ 0 and RAG tools are enabled:

1. Call each required `rag_search_*` tool early (harness may force this).
2. Use hits only as **hints** for names and dependencies seen in the repo.
3. Prefer live cluster/telemetry over docs when they disagree.
4. Never treat RAG as proof the cluster is healthy or broken — confirm with live tools.

## Correlate

- Match failing pod/deployment/workload name from cluster tools to the strongest post-T0 signal.
- Prefer the workload with abnormal replicas, restarts, or error-rate change after T0.
- If multiple candidates, pick the one with strongest **post-T0** evidence.

## Rules

- Name a concrete workload string in `suggested_root_cause` (from tool output).
- `log_action` the RCA and which tools backed it.
- Proceed to remediate when write tools are available — diagnosis is not the end.

## Anti-patterns

- Stopping at prose diagnosis without calling remediation tools when they exist.
- Blaming a dependency when only the injected workload’s replicas/config changed.
