---
name: aiops-remediate
description: >-
  Execute Kubernetes remediation tools after a fault is found. Use for all AIOps
  agents with scale_deployment, restart_deployment, or delete_pod available.
---

# AIOps — Remediate

## Goal

**Call** write tools to fix the fault. Suggestions in final JSON do not count as execution.

## Mandatory rule

When remediation tools are in the allowed tool list and `detected=true`:

1. Choose the write tool that matches **observed** cluster state (replicas, events, pod phase).
2. **Invoke** `scale_deployment`, `restart_deployment`, and/or `delete_pod` on the diagnosed workload.
3. Do **not** finish with only `suggested_remediations` text.

## Tool choice (summary)

| Observed condition | Preferred tool |
|--------------------|----------------|
| replicas = 0 / scaled down | `scale_deployment` (restore expected replica count) |
| Bad config / CrashLoop after config change | Fix path if available; else `restart_deployment` after config is healthy |
| Single dead/stuck pod, deploy otherwise OK | `delete_pod` (let controller recreate) or `restart_deployment` |

If unsure between restart and scale: check `get_pod_status` replica counts first.

## Rules

- Target the diagnosed workload’s deployment/pod — not an unrelated sibling.
- One clear write attempt, then verify (see verify skill).
- `log_action` with tool name, target, and arguments.

## Anti-patterns

- Writing remediation advice in JSON with **zero** write-tool calls.
- Restart when replicas are 0 (will not restore capacity).
- Calling write tools before any read confirms the target.
