---
name: netobserv-heal
description: "Remediate todo→PostgreSQL incidents after user confirmation via ansible-automation MCP (AAP job templates). Use when user says yes heal/remediate/restore/fix, or /netobserv-heal."
user-invocable: true
---

# NetObserv heal / restore (via AAP)

Remediation runs through **ansible-automation** MCP → Ansible Automation Platform job templates.
Do **not** use netobserv-openshift MCP for mutations.

## Gates

- Run only after explicit confirmation (or `/netobserv-heal` with clear fix intent).
- Never inject faults. Never ask for kubeconfig.

## Pick the right job template

| Incident type | User signals | `ansible_launch_job` template |
|---------------|--------------|-------------------------------|
| **Policy / microsegmentation** — Scenario B | "restore policy", "fix connectivity" after policy diagnosis | `netobserv-restore-policy` |
| **Latency / path degradation** — Scenario A | "heal", "clear shaping", "fix latency" | `netobserv-heal-db-path` |

If prior evidence classified **policy_connectivity_denial** → **`netobserv-restore-policy`**, not heal-db-path.

## Policy restore (Scenario B)

1. Call `ansible_launch_job(job_template="netobserv-restore-policy", confirmed=true)`
2. Quote `POLICY_VERDICT` from tool output

**Do not** call netobserv probe/status tools after restore in the same turn.

## Latency heal (Scenario A)

1. Call `ansible_launch_job(job_template="netobserv-heal-db-path", confirmed=true)`
2. Quote `VERDICT` / `HEAL_RESULT` from tool output

Agent heal clears **cluster-side** chaos/tc only — presenter still runs bastion `krkn restore`.

## Reply format

- **What we fixed** — one short paragraph
- **Verification** — quote AAP verdict lines
- **Expected impact** — plain language

Do **not** name bastion scripts, Kraken, or loadgen in user-facing replies.

## Fallback

If ansible-automation MCP is unavailable, stop and ask the operator to run `./scripts/wire-openclaw-aap.sh all`.
