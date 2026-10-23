---
name: aiops-remediate
description: >-
  Execute Kubernetes remediation tools after a fault is found. Use for all AIOps
  agents with scale_deployment, restart_deployment, delete_pod, or patch_deployment
  available.
---

# AIOps — Remediate

## Goal

**Call** write tools to fix the fault. Suggestions in final JSON do not count as execution.

## Mandatory rule

When remediation tools are in the allowed tool list and `detected=true`:

1. Choose the write tool that matches **observed** cluster state (replicas, events, pod phase, env config).
2. **Invoke** `scale_deployment`, `restart_deployment`, `delete_pod`, and/or `patch_deployment` on the diagnosed workload.
3. Do **not** finish with only `suggested_remediations` text.

## Tool choice (summary)

| Observed condition | Preferred tool |
|--------------------|----------------|
| replicas = 0 / scaled down | `scale_deployment` (restore expected replica count) |
| Bad config / corrupted or injected env var | `patch_deployment` (remove the bad env vars found via `get_deployment_config`); a restart or scale keeps a bad env var |
| Single dead/stuck pod, deploy otherwise OK | `delete_pod` (let controller recreate) or `restart_deployment` |

If unsure between restart and scale: check `get_pod_status` replica counts first.
If unsure whether config is the problem: run `get_deployment_config` on the diagnosed deployment and look for env vars with invalid/injected values.

## Rules

- Target the diagnosed workload’s deployment/pod — not an unrelated sibling.
- One clear write attempt, then verify (see verify skill).
- `log_action` with tool name, target, and arguments.

## Anti-patterns

- Writing remediation advice in JSON with **zero** write-tool calls.
- Restart when replicas are 0 (will not restore capacity).
- Restarting to "fix" a corrupted env var (the pod comes back with the same bad config).
- Calling write tools before any read confirms the target.
