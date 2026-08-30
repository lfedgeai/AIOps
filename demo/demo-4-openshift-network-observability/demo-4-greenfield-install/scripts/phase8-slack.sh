#!/usr/bin/env bash
# Phase 8 helper — Slack Socket Mode for OpenClaw.
#
# Usage:
#   ./scripts/phase8-slack.sh plan
#   ./scripts/phase8-slack.sh secrets   # apply tokens from site config → K8s secret
#   ./scripts/phase8-slack.sh wire      # pin slack plugin + recycle OpenClaw
#   ./scripts/phase8-slack.sh deploy    # secrets + wire (when tokens in site config)
#   ./scripts/phase8-slack.sh check
#   ./scripts/phase8-slack.sh status
#   ./scripts/phase8-slack.sh verify
#
# Env: DEMO_KIT_ROOT, OPENCLAW_NS, SLACK_CHANNEL_ID, LAB_CFG
# Tokens live in gitignored site-secrets — never commit slack.bot_token / app_token.
set -euo pipefail

GF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GF_ROOT
[[ -f "$GF_ROOT/config/env.local" ]] && source "$GF_ROOT/config/env.local"
# shellcheck source=scripts/resolve-demo-kit.sh
source "$GF_ROOT/scripts/resolve-demo-kit.sh"
# shellcheck source=scripts/site-config.sh
source "$GF_ROOT/scripts/site-config.sh"
resolve_demo_kit

KUBECTL="$(command -v oc || command -v kubectl)"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
LAB_CFG="${LAB_CFG:-$HOME/labs/openshell-on-openshift-lab/manifests/openclaw/config.yaml}"
KIT="$DEMO_KIT_ROOT/scripts"

c_green=$'\033[1;32m'; c_red=$'\033[1;31m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '  %s✓%s %s\n' "$c_green" "$c_reset" "$*"; }
fail() { printf '  %s✗%s %s\n' "$c_red" "$c_reset" "$*"; }
warn() { printf '  %s!%s %s\n' "$c_yellow" "$c_reset" "$*"; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }

gate() {
  local label="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    ok "$label"
    return 0
  fi
  fail "$label"
  return 1
}

slack_secret_present() {
  "$KUBECTL" -n "$OPENCLAW_NS" get secret/openclaw-slack-tokens >/dev/null 2>&1
}

slack_env_wired() {
  "$KUBECTL" -n "$OPENCLAW_NS" set env deployment/openclaw --list 2>/dev/null \
    | grep -qE '^SLACK_BOT_TOKEN=|^# SLACK_BOT_TOKEN from secret'
}

openclaw_slack_configured() {
  "$KUBECTL" -n "$OPENCLAW_NS" get configmap openclaw-config \
    -o jsonpath='{.data.openclaw\.json}' 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); allow=(d.get("plugins") or {}).get("allow") or []; slack=(d.get("channels") or {}).get("slack") or {}; print("slack" in allow and slack.get("enabled") is True)' 2>/dev/null \
    | grep -q True
}

channel_allowlisted() {
  local ch="${SLACK_CHANNEL_ID:-}"
  [[ -n "$ch" ]] || return 1
  "$KUBECTL" -n "$OPENCLAW_NS" get configmap openclaw-config \
    -o jsonpath='{.data.openclaw\.json}' 2>/dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin); ch=(d.get('channels') or {}).get('slack',{}).get('channels',{}); print('${ch}' in ch)" 2>/dev/null \
    | grep -q True
}

load_slack_env() {
  site_config_ensure slack 2>/dev/null || site_config_ensure openshell || true
  eval "$(site_config_load)" 2>/dev/null || true
  export SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}"
}

print_secrets_handoff() {
  step "Manual step — Slack app tokens (not in git)"
  cat <<EOF

  1. Create/configure Slack app: https://api.slack.com/apps
     - Enable Socket Mode
     - Bot scopes: app_mentions:read, chat:write, channels:read, channels:history,
       groups:read, groups:history, im:read, im:history, mpim:read, mpim:history
     - Event Subscriptions (Socket Mode): bot event app_mention
     - Install app to workspace; invite bot to demo channel

  2. Store tokens (pick one):
     a) Site config (gitignored):
        ./scripts/greenfield-install.sh config prompt --full
        # slack.bot_token, slack.app_token, site.slack_channel_id

     b) Direct secret:
        oc -n ${OPENCLAW_NS} create secret generic openclaw-slack-tokens \\
          --from-literal=SLACK_BOT_TOKEN='xoxb-...' \\
          --from-literal=SLACK_APP_TOKEN='xapp-...' \\
          --dry-run=client -o yaml | oc apply -f -
        oc -n ${OPENCLAW_NS} set env deployment/openclaw --from=secret/openclaw-slack-tokens

  3. Continue:
     ./scripts/phase8-slack.sh secrets   # if using site config
     ./scripts/phase8-slack.sh wire

Guide: $GF_ROOT/docs/PHASE-8-SLACK.md
EOF
}

cmd_plan() {
  cat <<EOF
Phase 8 — Slack Socket Mode (~15 min manual + 5 min wire)

Guide: $GF_ROOT/docs/PHASE-8-SLACK.md

Prereq: Phase 7 (agent kit seeded). Slack app + channel ID.

Steps:
  1. YOU — create Slack app, enable Socket Mode, invite bot to channel
  2. Store tokens (gitignored site config or oc secret)
       ./scripts/greenfield-install.sh config prompt --full
       ./scripts/phase8-slack.sh secrets
  3. Wire plugin + channel allowlist
       SLACK_CHANNEL_ID=C… ./scripts/phase8-slack.sh wire

Or: ./scripts/phase8-slack.sh deploy   (when site config has slack tokens)

Verify:
  ./scripts/phase8-slack.sh check
  $KIT/netobserv-e2e-openclaw-test.sh slack-check
  Test in Slack: @OpenClaw in allowlisted channel
EOF
}

cmd_secrets() {
  step "Apply Slack tokens from site config"
  load_slack_env
  if [[ -z "${SLACK_BOT_TOKEN:-}" || -z "${SLACK_APP_TOKEN:-}" ]]; then
    print_secrets_handoff
    fail "slack.bot_token and slack.app_token required in site config"
    exit 1
  fi
  site_config_apply slack || { fail "site_config apply slack failed"; exit 1; }
  ok "secret/openclaw-slack-tokens applied"
}

cmd_wire() {
  step "Wire Slack Socket Mode"
  load_slack_env
  slack_secret_present || { print_secrets_handoff; fail "Run ./scripts/phase8-slack.sh secrets first"; exit 1; }
  [[ -n "${SLACK_CHANNEL_ID:-}" ]] || {
    fail "site.slack_channel_id / SLACK_CHANNEL_ID required"
    exit 1
  }
  [[ -f "$LAB_CFG" ]] || "$KIT/clone-openshell-lab.sh" install
  RECYCLE_POD=1 SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" OPENCLAW_NS="$OPENCLAW_NS" \
    "$KIT/wire-openclaw-slack.sh"
  ok "Wire complete — test @OpenClaw in channel ${SLACK_CHANNEL_ID}"
}

cmd_deploy() {
  step "Phase 8 deploy — Slack"
  "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw >/dev/null 2>&1 \
    || { fail "Phase 2 required"; exit 1; }
  "$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-mcp >/dev/null 2>&1 \
    || warn "Phase 7 recommended — run phase7-agent.sh deploy first"

  if slack_secret_present && slack_env_wired; then
    ok "Slack secret already present — skipping secrets step"
  elif cmd_secrets; then
    :
  else
    exit 1
  fi
  cmd_wire
  ok "Deploy complete — run: ./scripts/phase8-slack.sh verify"
}

cmd_check() {
  local fails=0 ch
  step "Phase 8 readiness — $($KUBECTL whoami 2>/dev/null || echo '?')"
  load_slack_env
  ch="${SLACK_CHANNEL_ID:-}"

  gate "Phase 2 — openclaw deployment" "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw || fails=$((fails + 1))
  gate "openclaw-slack-tokens secret" slack_secret_present || fails=$((fails + 1))
  if slack_env_wired; then
    ok "SLACK_* env on deployment/openclaw"
  else
    fail "SLACK_* env missing — run phase8-slack.sh secrets"
    fails=$((fails + 1))
  fi
  if openclaw_slack_configured; then
    ok "plugins.allow + channels.slack enabled"
  else
    fail "Slack not in openclaw-config — run phase8-slack.sh wire"
    fails=$((fails + 1))
  fi
  if [[ -n "$ch" ]]; then
    if channel_allowlisted; then
      ok "Channel ${ch} allowlisted (requireMention)"
    else
      fail "Channel ${ch} not in channels.slack — re-run wire"
      fails=$((fails + 1))
    fi
  else
    warn "SLACK_CHANNEL_ID unset — skip channel allowlist check"
  fi

  printf '\n'
  if [[ "$fails" -gt 0 ]]; then
    warn "$fails check(s) failed — see $GF_ROOT/docs/PHASE-8-SLACK.md#troubleshooting"
    return 1
  fi
  ok "Phase 8 checks passed"
  printf '  Full path: %s/netobserv-e2e-openclaw-test.sh slack-check\n' "$KIT"
  return 0
}

cmd_status() {
  step "Slack / OpenClaw status"
  "$KIT/netobserv-e2e-openclaw-test.sh" slack-check 2>/dev/null || true
}

cmd_verify() {
  cmd_check || true
  step "slack-check (demo kit)"
  "$KIT/netobserv-e2e-openclaw-test.sh" slack-check
}

case "${1:-plan}" in
  plan|help|-h|--help) cmd_plan ;;
  secrets|apply-secrets) cmd_secrets ;;
  wire) cmd_wire ;;
  deploy|install) cmd_deploy ;;
  check|verify-gates) cmd_check ;;
  status) cmd_status ;;
  verify|test) cmd_verify ;;
  *)
    echo "usage: $0 {plan|secrets|wire|deploy|check|status|verify}" >&2
    exit 1
    ;;
esac
