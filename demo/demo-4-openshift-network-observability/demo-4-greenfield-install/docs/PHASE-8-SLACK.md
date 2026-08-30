# Phase 8 — Slack Socket Mode

**Time:** ~15 min manual (Slack app) + ~5 min wire  
**Orchestrator:** `./scripts/greenfield-install.sh slack`  
**Helper:** `./scripts/phase8-slack.sh` (`plan` · `secrets` · `wire` · `deploy` · `check` · `status` · `verify`)

Adds a **second front door** — same OpenClaw agent (skills, MCP, sandbox) in an allowlisted Slack channel via **Socket Mode** (no inbound Route).

**Token policy:** Store `slack.bot_token` and `slack.app_token` in **gitignored** `config/site-secrets.local.yaml` only — never commit to git.

**Quick checklist:** see [README.md — Slack setup](../README.md#slack-setup-phase-8) for the full permissions table.

---

## Before you start

| Requirement | Check |
|-------------|--------|
| Phase 7 complete | `agent ✓` · MCP + skills seeded |
| Slack app | [api.slack.com/apps](https://api.slack.com/apps) — Socket Mode on |
| Demo channel | Bot invited; note channel ID (`C…`) |
| Site config | `site.slack_channel_id` + slack tokens |

```bash
./scripts/greenfield-install.sh config prompt --full
# or edit config/site-secrets.local.yaml (gitignored)
./scripts/phase8-slack.sh plan
```

---

## Step 1 — YOU: Slack app (manual)

### Create app + Socket Mode

1. **Create New App** → **From scratch** (bot `@name` = app display name you choose).
2. **Socket Mode** → **ON**.
3. **App-Level Token** → scope `connections:write` → copy `xapp-…`.

### Bot token scopes

**OAuth & Permissions** → **Bot Token Scopes** — add **all** rows:

| Scope | Why |
|-------|-----|
| `app_mentions:read` | Receive `@bot` mentions |
| `chat:write` | Post replies (**not** `calls:write`) |
| `channels:read` | Resolve public channel IDs |
| `channels:history` | Channel message history |
| `groups:read` | Private channel IDs |
| `groups:history` | Private channel history |
| `im:read` | Direct messages |
| `im:history` | DM history |
| `mpim:read` | Group DMs |
| `mpim:history` | Group DM history |
| `commands` | Optional — `/openshell` only |

**Install App** → **Reinstall to Workspace** after changing scopes. Copy **Bot User OAuth Token** (`xoxb-…`).

### Event subscriptions

With Socket Mode ON:

1. **Event Subscriptions** → **ON**
2. **Subscribe to bot events** → **`app_mention`**

### Channel

1. `/invite @<your-bot-name>` in demo channel
2. Copy **channel ID** (`C…`) from channel details

See [README.md — Slack setup](../README.md#slack-setup-phase-8) for the full permissions table.

---

## Step 2 — Store tokens (site config or oc)

**Recommended** — gitignored site secrets:

```yaml
site:
  slack_channel_id: C0123456789
slack:
  bot_token: xoxb-...
  app_token: xapp-...
```

```bash
./scripts/phase8-slack.sh secrets
```

**Or** direct on cluster:

```bash
oc -n openclaw create secret generic openclaw-slack-tokens \
  --from-literal=SLACK_BOT_TOKEN='xoxb-...' \
  --from-literal=SLACK_APP_TOKEN='xapp-...' \
  --dry-run=client -o yaml | oc apply -f -
oc -n openclaw set env deployment/openclaw --from=secret/openclaw-slack-tokens
```

---

## Step 3 — Wire

```bash
./scripts/phase8-slack.sh wire
# or full path:
./scripts/phase8-slack.sh deploy
```

Wires `plugins.allow` + `channels.slack`, hardens seed init for `@openclaw/slack`, recycles OpenClaw.

**Verify:**

```bash
./scripts/phase8-slack.sh verify
```

Test in Slack: `@<your-bot-name>` in allowlisted channel (`requireMention: true`).

**Next phase:** `./scripts/greenfield-install.sh event`

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `MISSING @openclaw/slack plugin` | Re-run `phase8-slack.sh wire` (seed v5 npm install) |
| Socket Mode probe fails | Check tokens; recycle pod |
| `missing_scope` in OpenClaw logs | Add `channels:read`, `groups:read`, `im:read`, `mpim:read`; reinstall app |
| Bot connects but never replies | Add `chat:write` (not `calls:write`); add `app_mention` event |
| Bot silent in channel | `/invite @bot`; verify channel ID; re-run `wire` |
| Plugin lost after restart | Do not manual `plugins install` — use wire script |
| `slack-check` env missing | `phase8-slack.sh secrets` |
| Two clusters, flaky Slack | Use separate Slack app per cluster (Socket Mode) |

---

## Quick reference

```bash
./scripts/phase8-slack.sh plan
./scripts/phase8-slack.sh secrets
./scripts/phase8-slack.sh wire
./scripts/phase8-slack.sh check
$KIT/netobserv-e2e-openclaw-test.sh slack-check
```
