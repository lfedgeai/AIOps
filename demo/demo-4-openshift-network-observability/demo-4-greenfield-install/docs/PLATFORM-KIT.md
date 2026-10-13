# Platform kit (`demo-4-platform-kit`)

Install scripts, OpenShift manifests, OpenClaw skills/MCP, and Ansible playbooks used by the greenfield orchestrator.

```text
demo-4-greenfield-install/     orchestrator, site secrets, phase docs
demo-4-platform-kit/           DEMO_KIT_ROOT — this folder
```

Clone or copy **both folders side by side** (or set `DEMO_KIT_ROOT` explicitly).

---

## What is included

| Area | Contents |
|------|----------|
| Phases 1–12 | NetObserv, OpenShell/OpenClaw, RHOAI, Grafana, TrustyAI, AAP, agent seed, Slack, event-AIOps, SPIFFE, RHCL |
| `manifests/` | OpenShift YAML (operators, wiring, SPIFFE, RHCL, …) |
| `openclaw-skills/` | Agent skills + MCP manifests |
| `ansible/` | AAP playbooks for confirm-gated heal |
| Day-2 helpers | `demo-cluster-preflight.sh`, `post-cluster-spiffe-resume.sh`, `netobserv-e2e-openclaw-test.sh`, … |

---

## `DEMO_KIT_ROOT` resolution

Used by every `greenfield-install.sh` phase and by `scripts/resolve-demo-kit.sh`:

1. `$DEMO_KIT_ROOT` if set
2. Sibling `../demo-4-platform-kit`

```bash
export DEMO_KIT_ROOT="${DEMO_KIT_ROOT:-$(cd ../demo-4-platform-kit && pwd)}"
```

---

## Slack channel

Kit scripts source `scripts/resolve-slack-channel.sh`:

1. `SLACK_CHANNEL_ID` env (optional `config/env.local`)
2. `site.slack_channel_id` in greenfield `config/site-secrets.local.yaml`

Configure once during `./scripts/greenfield-install.sh config prompt`.

---

## Bastion layout

```bash
~/AIOps/demo/demo-4-greenfield-install
~/AIOps/demo/demo-4-platform-kit
export DEMO_KIT_ROOT=~/AIOps/demo/demo-4-platform-kit
```

Sync from laptop:

```bash
cd demo-4-greenfield-install
cp config/env.example config/env.local   # set BASTION_HOST
./scripts/sync-to-bastion.sh all
```

---

## External dependency (not in this repo)

Phase 2 clones the pinned OpenShell lab to `~/labs/openshell-on-openshift-lab` via `clone-openshell-lab.sh` (URLs/commits in `scripts/supply-chain-pins.env`).

---

## Secrets

Never commit `config/site-secrets.local.yaml` or `config/env.local` in the **greenfield** folder. See `.gitignore` in each folder.
