---
name: ansible-automation
description: "Launch governed Ansible Automation Platform job templates after user confirmation. Use for remediation, runbooks, and platform tasks — not investigation."
user-invocable: true
---

# Ansible automation (AAP)

Generic automation surface via **ansible-automation** MCP. Investigation stays on domain MCPs
(NetObserv, OpenShift, …); confirmed actions launch **pre-approved AAP job templates**.

## Gates

- **Never** launch during investigation or evidence-analysis turns.
- **Always** pass `confirmed=true` only after the user explicitly approves in the **current** message.
- Prefer `/new` for remediation after a long investigation session.

## Tools (ansible-automation MCP)

| Tool | When |
|------|------|
| `ansible_list_job_templates` | Discover approved templates (optional) |
| `ansible_launch_job` | Launch by template name; default `extra_vars_json={"confirmed": true}` |
| `ansible_job_status` | Poll an existing job by id |

## NetObserv demo templates

| Incident | `job_template` |
|----------|----------------|
| Latency / Kraken / tc path (Scenario A) | `netobserv-heal-db-path` |
| NetworkPolicy / microsegmentation (Scenario B) | `netobserv-restore-policy` |

## Launch example

```
ansible_launch_job(
  job_template="netobserv-heal-db-path",
  confirmed=true,
  extra_vars_json='{"confirmed": true}'
)
```

Quote `AAP_JOB_ID`, `AAP_JOB_STATUS`, and verdict lines (`VERDICT:`, `POLICY_VERDICT:`) to the user.

## Adding future automation

1. Add playbook to Gitea → sync AAP project → create Job Template in AAP.
2. Document template name here (or in a domain skill).
3. Agent calls `ansible_launch_job` with that template name — no MCP code changes required.
