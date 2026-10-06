# Demo 4 — Greenfield install

Replicate the full **NetObserv + OpenClaw Demo A + platform stack** on a **new OpenShift cluster**.

This folder is the **orchestrator** (phased install, site secrets, docs). Install scripts and manifests live in the sibling **platform kit** (`demo-4-platform-kit`).

---

## Layout

```text
demo-4-openshift-network-observability/
  demo-4-greenfield-install/   ← you are here (orchestrator)
  demo-4-platform-kit/         ← DEMO_KIT_ROOT (sibling folder)
```

See [docs/PLATFORM-KIT.md](docs/PLATFORM-KIT.md) for what the platform kit contains.

---

## Quick start

```bash
# 1. Clone repo; cd into greenfield (platform kit is sibling ../demo-4-platform-kit)
cd demo/demo-4-openshift-network-observability/demo-4-greenfield-install
cp config/env.example config/env.local   # optional
source config/env.local

export DEMO_KIT_ROOT="${DEMO_KIT_ROOT:-$(cd ../demo-4-platform-kit && pwd)}"

# 2. OpenShift login on bastion (see docs/CLUSTER-LOGIN.md)
export OCP_API="https://api.cluster-<name>.<domain>:6443"
oc login "$OCP_API" -u kubeadmin -p '<password>'
./scripts/cluster-login.sh check

# 3. Site secrets (AWS + LLM — required before Phase 1)
./scripts/greenfield-install.sh config prompt

# 4. Preflight + install
chmod +x scripts/*.sh scripts/site_config.py
./scripts/greenfield-install.sh preflight
./scripts/greenfield-install.sh netobserv    # Phase 1
```

**Full guide:** [INSTALL.md](INSTALL.md)  
**Phase helpers:** `./scripts/phase2-openshell.sh` · `phase3-rhoai.sh` · … · `phase11-rhcl.sh`  
**Checklist:** [docs/PHASE-CHECKLIST.md](docs/PHASE-CHECKLIST.md)  
**Cluster was powered off mid-install:** [CLUSTER-WAKE.md](CLUSTER-WAKE.md)

---

## Bastion sync (laptop → cluster bastion)

```bash
./scripts/sync-to-bastion.sh all
```

Syncs **greenfield** and **platform kit** to the target bastion. Set `BASTION_HOST` in `config/env.local`.

**Cluster login:** [docs/CLUSTER-LOGIN.md](docs/CLUSTER-LOGIN.md) · `./scripts/cluster-login.sh check`

---

## What gets installed

| Phase | Components |
|-------|------------|
| 1 | NetObserv + Loki (AWS) + todo app |
| 2 | OpenShell + OpenClaw + LLM |
| 3 | RHOAI + MLflow Traces |
| 4 | Grafana Network AIOps + OTel federation |
| 5 | TrustyAI guardrails |
| 6 | AAP + Gitea + ansible-automation MCP |
| 7 | NetObserv agent skills seed |
| 8–10 | Slack, event-AIOps, SPIFFE mTLS |
| 11 | RHCL OAuth (optional) |
| 12 | Verify + trial `demo-a-fast` |

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

## After install — demo day

```bash
export DEMO_KIT_ROOT="${DEMO_KIT_ROOT:-$(cd ../demo-4-platform-kit && pwd)}"
"$DEMO_KIT_ROOT/scripts/demo-cluster-preflight.sh" check
"$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh" demo-a-fast
```

Cluster wake / heal: [CLUSTER-WAKE.md](CLUSTER-WAKE.md)
