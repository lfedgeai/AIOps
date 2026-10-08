# Greenfield install

Phased installer for the governed agentic AIOps demo on a new OpenShift cluster. This folder holds the orchestrator, site configuration and per-phase runbooks. What it deploys lives in the sibling [`demo-4-platform-kit/`](../demo-4-platform-kit/), which the installer finds automatically as `DEMO_KIT_ROOT`.

For what the demo is and why, see the [top-level README](../README.md). To run the demos after installing, see the [platform kit README](../demo-4-platform-kit/README.md).

---

## Prerequisites

- OpenShift **4.21+** on AWS (IPI), OVN-Kubernetes, `cluster-admin`
- A RHEL 9 bastion with `oc`, `aws`, `jq` and `python3`, logged in to the cluster — [`docs/CLUSTER-LOGIN.md`](docs/CLUSTER-LOGIN.md)
- An S3 bucket and AWS credentials for LokiStack
- An OpenAI-compatible LLM endpoint — [`docs/PHASE-2-LLM-PROVIDERS.md`](docs/PHASE-2-LLM-PROVIDERS.md)
- A Slack app (Phases 8–9) and an AAP subscription (Phase 6)

Full prerequisites, time budget and the credentials you collect by hand: [`INSTALL.md`](INSTALL.md).

---

## Quick start

```bash
git clone https://github.com/lfedgeai/AIOps.git
cd AIOps/demo/demo-4-openshift-network-observability/demo-4-greenfield-install

cp config/env.example config/env.local          # optional overrides
source config/env.local

./scripts/cluster-login.sh check                # confirm the bastion is logged in
./scripts/greenfield-install.sh config prompt   # AWS, LLM and Slack → config/site-secrets.local.yaml
./scripts/greenfield-install.sh preflight
./scripts/greenfield-install.sh all             # or one phase at a time — see below
./scripts/greenfield-install.sh phases          # progress at any point
```

`config/site-secrets.local.yaml` and `config/env.local` are gitignored. Never commit them.


### Installing from a laptop

Edit on your laptop, run on the bastion. Set `BASTION_HOST` in `config/env.local`, then:

```bash
./scripts/sync-to-bastion.sh            # both folders; or: greenfield | kit
```


Syncs **greenfield** and **platform kit** to the target bastion. Set `BASTION_HOST` in `config/env.local`.

**Cluster login:** [docs/CLUSTER-LOGIN.md](docs/CLUSTER-LOGIN.md) · `./scripts/cluster-login.sh check`

---

## Phases

Run any phase by number or name — `./scripts/greenfield-install.sh 6` and `./scripts/greenfield-install.sh aap` are the same. Phases 2–11 also have their own scripts, `scripts/phaseN-*.sh`, with `plan`, `check` and `verify` — and `deploy` for every phase except 3, which the main installer runs.

| Phase | Name | Installs | Runbook |
|---|---|---|---|
| 1 | `netobserv` | Network Observability + LokiStack (S3) + todo app | [INSTALL.md](INSTALL.md#phase-1--netobserv--sample-app) |
| 2 | `openshell` | OpenShell sandbox + OpenClaw agent + LLM provider | [PHASE-2](docs/PHASE-2-OPENSHELL.md) |
| 3 | `rhoai` | Red Hat OpenShift AI + MLflow traces | [PHASE-3](docs/PHASE-3-RHOAI.md) |
| 4 | `grafana` | Grafana network AIOps dashboards + OTel federation | [PHASE-4](docs/PHASE-4-GRAFANA.md) |
| 5 | `guardrails` | TrustyAI guardrails | [PHASE-5](docs/PHASE-5-GUARDRAILS.md) |
| 6 | `aap` | Ansible Automation Platform + Gitea + ansible MCP | [PHASE-6](docs/PHASE-6-AAP.md) |
| 7 | `agent` | Agent skills and MCP servers seeded | [PHASE-7](docs/PHASE-7-AGENT.md) |
| 8 | `slack` | Slack Socket Mode — one-time app setup in Slack first | [PHASE-8](docs/PHASE-8-SLACK.md) |
| 9 | `event` | Event-driven path: Grafana alert → bridge → Slack thread | [PHASE-9](docs/PHASE-9-EVENT.md) |
| 10 | `spiffe` | Zero Trust Workload Identity — mTLS on the bridge-to-agent hop | [PHASE-10](docs/PHASE-10-SPIFFE.md) |
| 11 | `rhcl` | Red Hat Connectivity Link OAuth for the UI *(optional — `SKIP_RHCL=1` to skip)* | [PHASE-11](docs/PHASE-11-RHCL.md) |
| 12 | `verify` | End-to-end verification + trial run of the fast demo | [Checklist](docs/PHASE-CHECKLIST.md) |

Phases 8 and 10 have manual steps — creating the Slack app and its scopes, and subscribing to the ZTWI operator. Their runbooks cover each step and the common mistakes.

---

## Slack setup (Phase 8)

One-time **manual** work in [api.slack.com/apps](https://api.slack.com/apps), then wire on the bastion. Use a **dedicated Slack app per cluster** (Socket Mode allows only one active connection per app token).

### 1. Create the app

1. **Create New App** → **From scratch** (name it anything, e.g. `AgentOps-Test` — that becomes your `@mention` name).
2. **Socket Mode** → turn **ON**.
3. Create an **App-Level Token** with scope `connections:write` → copy `xapp-…` (app token).

### 2. Bot token scopes (OAuth & Permissions)

Under **Scopes** → **Bot Token Scopes**, add **all** of these:

| Scope | Required | Why |
|-------|----------|-----|
| `app_mentions:read` | Yes | Receive `@bot` mentions |
| `chat:write` | Yes | Post replies (**not** `calls:write`) |
| `channels:read` | Yes | Resolve public channel IDs |
| `channels:history` | Yes | Read channel context |
| `groups:read` | Yes | Resolve private channel IDs |
| `groups:history` | Yes | Read private channel context |
| `im:read` | Yes | DM support |
| `im:history` | Yes | DM history |
| `mpim:read` | Yes | Group DM support |
| `mpim:history` | Yes | Group DM history |
| `commands` | Optional | Only if using `/openshell` slash command |

**After adding scopes:** **Install App** → **Reinstall to Workspace** (required when scopes change).

Copy the **Bot User OAuth Token** (`xoxb-…`).

### 3. Event subscriptions

With **Socket Mode** still ON:

1. **Event Subscriptions** → turn **ON**.
2. **Subscribe to bot events** → add **`app_mention`**.

Save changes. Reinstall the app again if Slack prompts you.

### 4. Channel + site secrets (bastion)

1. Create or pick a demo channel (e.g. `#agentops-test`).
2. `/invite @<your-bot-name>` in that channel.
3. Channel details → copy **Channel ID** (`C…`).
4. Edit gitignored `config/site-secrets.local.yaml`:

```yaml
site:
  slack_channel_id: C0123456789   # your channel ID
slack:
  bot_token: xoxb-...
  app_token: xapp-...
```

### 5. Wire on cluster

```bash
./scripts/phase8-slack.sh deploy
./scripts/phase8-slack.sh verify
```

Test in Slack: `@<your-bot-name> hello` (mention required — `requireMention: true`).

**Phase 9 (event-AIOps)** uses the same channel ID. After changing channel or tokens:

```bash
./scripts/phase9-event.sh hooks
```

### Common mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Used `calls:write` instead of `chat:write` | Bot connects but never replies | Add `chat:write`; reinstall app |
| Missing `channels:read` / `groups:read` | `missing_scope` in OpenClaw logs | Add read scopes from table above |
| No `app_mention` event | Socket connected, no responses to `@bot` | Event Subscriptions → `app_mention` |
| Wrong channel ID in site config | Bot silent in your channel | Update `site.slack_channel_id`; re-run `phase8-slack.sh wire` |
| Same app on two clusters | Flaky / one cluster steals connection | One app per cluster |
| Bot not invited | No replies | `/invite @<your-bot-name>` |

Full runbook: [docs/PHASE-8-SLACK.md](docs/PHASE-8-SLACK.md)

---

## SPIFFE / ZTWI (Phase 10)

Automated on the bastion after Phase 9. Installs **Zero Trust Workload Identity Manager** and upgrades the event path to **mTLS**.

```bash
./scripts/phase10-spiffe.sh deploy
./scripts/phase10-spiffe.sh verify
```

| Prereq | Notes |
|--------|--------|
| Phase 9 done | hooks + `netobserv-grafana-bridge` |
| OperatorHub | **Zero Trust Workload Identity Manager** subscription |
| `registry.redhat.io` | SPIFFE Helper image pull (cluster pull secret) |
| Time | ~20–40 min first install |

After cluster reboot: `./scripts/phase10-spiffe.sh repair`

Full runbook: [docs/PHASE-10-SPIFFE.md](docs/PHASE-10-SPIFFE.md)

---

## RHCL OAuth (Phase 11, optional)

Enterprise Control UI login via OpenShift OAuth. **Not required** for Slack/MCP demos.

```bash
./scripts/phase11-rhcl.sh deploy
./scripts/phase11-rhcl.sh verify
```

| Prereq | Notes |
|--------|--------|
| Phase 2 done | `openclaw` service |
| cert-manager | Usually pre-installed on OCP |
| OperatorHub | RHCL operator subscription |
| Skip | `SKIP_RHCL=1` keeps legacy Route + gateway token |

Test: incognito → `https://openclaw-rhcl.apps.<ingress>/`

Full runbook: [docs/PHASE-11-RHCL.md](docs/PHASE-11-RHCL.md)

---

## After install — Demo Day

```bash
export DEMO_KIT_ROOT="${DEMO_KIT_ROOT:-$(cd ../demo-4-platform-kit && pwd)}"
"$DEMO_KIT_ROOT/scripts/demo-cluster-preflight.sh" check
"$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh" demo-a-fast
```

---

## Reference

| Topic | Doc |
|---|---|
| Starting from a brand-new cluster | [`docs/START-NEW-CLUSTER.md`](docs/START-NEW-CLUSTER.md) |
| Cluster powered off mid-install or between demos | [`CLUSTER-WAKE.md`](CLUSTER-WAKE.md) |
| How the installer finds and syncs the platform kit | [`docs/PLATFORM-KIT.md`](docs/PLATFORM-KIT.md) |
| Disconnected or mirrored registries | [`docs/IMAGE-MIRRORS.md`](docs/IMAGE-MIRRORS.md) |

---

## Next

Installed? Run the demos: [`demo-4-platform-kit/README.md`](../demo-4-platform-kit/README.md).
