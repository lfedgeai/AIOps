# Greenfield install — phase checklist

Copy to `PROGRESS.local.md` and tick as you go (do not commit secrets).

**Cluster:** ___________________  
**Bastion:** ___________________  
**Started:** ___________________

| ☐ | Phase | Command | Verified |
|---|-------|---------|----------|
| ☐ | 0 Prerequisites | `config init` + `config prompt` (or edit site-secrets YAML) | `config validate` |
| ☐ | 1 NetObserv + todo | `./scripts/greenfield-install.sh netobserv` | Network Traffic in Console |
| ☐ | 2 OpenShell + OpenClaw | `./scripts/greenfield-install.sh openshell` | `phase2-openshell.sh check` · `probe` · `ui-hints` |
| ☐ | 3 RHOAI + MLflow | `./scripts/greenfield-install.sh rhoai` | `phase3-rhoai.sh verify` |
| ☐ | 4 Grafana + OTel | `./scripts/greenfield-install.sh grafana` | `phase4-grafana.sh verify` |
| ☐ | 5 TrustyAI | `./scripts/greenfield-install.sh guardrails` | `phase5-guardrails.sh verify` |
| ☐ | 6 AAP + Gitea | `./scripts/greenfield-install.sh aap` | License in Gateway UI (manual) |
| ☐ | 6b AAP wire | `./scripts/phase6-aap.sh wire` | `phase6-aap.sh verify` · `aap-check` |
| ☐ | 7 Agent seed | `./scripts/greenfield-install.sh agent` | `phase7-agent.sh verify` · MCP doctor |
| ☐ | 8 Slack | `./scripts/greenfield-install.sh slack` | `phase8-slack.sh verify` · @OpenClaw in channel |
| ☐ | 9 Event-AIOps | `./scripts/greenfield-install.sh event` | `phase9-event.sh verify` · `event-aiops-check` |
| ☐ | 10 SPIFFE | `./scripts/greenfield-install.sh spiffe` | `phase10-spiffe.sh verify` · `spiffe-check` |
| ☐ | 11 RHCL (opt) | `./scripts/greenfield-install.sh rhcl` | `phase11-rhcl.sh verify` · incognito OAuth login |
| ☐ | 12 Verify | `./scripts/greenfield-install.sh verify` | `demo-a-fast` trial |

**Auto-detect progress on cluster:**

```bash
./scripts/greenfield-install.sh phases
```

**Resume after cluster power-off:** [CLUSTER-WAKE.md](../CLUSTER-WAKE.md)
