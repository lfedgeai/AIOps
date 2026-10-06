#!/usr/bin/env bash
# Phase 9 helper — Event-driven AIOps (Grafana alert → hooks → Slack).
#
# Usage:
#   ./scripts/phase9-event.sh plan
#   ./scripts/phase9-event.sh metrics   # federate NetObserv → demo Prometheus
#   ./scripts/phase9-event.sh hooks     # openclaw-hooks-token + grafana-bridge
#   ./scripts/phase9-event.sh alerts    # Grafana contact point + RTT rule
#   ./scripts/phase9-event.sh wire      # hooks + alerts (no metrics sync)
#   ./scripts/phase9-event.sh deploy    # metrics + full event path
#   ./scripts/phase9-event.sh check
#   ./scripts/phase9-event.sh status
#   ./scripts/phase9-event.sh verify
#
# Env: DEMO_KIT_ROOT, OPENCLAW_NS, GRAFANA_NS, SLACK_CHANNEL_ID, LAB_CFG
#      RECYCLE_POD (default 1 on hooks), SKIP_SMOKE (default 0)
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
GRAFANA_NS="${GRAFANA_NS:-netobserv-demo}"
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

load_event_env() {
  site_config_ensure event 2>/dev/null || site_config_ensure slack || true
  eval "$(site_config_load)" 2>/dev/null || true
  export SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}"
}

hooks_secret_present() {
  "$KUBECTL" -n "$OPENCLAW_NS" get secret/openclaw-hooks-token >/dev/null 2>&1
}

hooks_env_wired() {
  "$KUBECTL" -n "$OPENCLAW_NS" set env deployment/openclaw --list 2>/dev/null \
    | grep -qE '^OPENCLAW_HOOKS_TOKEN=|^# OPENCLAW_HOOKS_TOKEN from secret'
}

hooks_configured() {
  "$KUBECTL" -n "$OPENCLAW_NS" get configmap openclaw-config \
    -o jsonpath='{.data.openclaw\.json}' 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); h=d.get("hooks") or {}; print(h.get("enabled") is True and bool(h.get("path")) and bool(h.get("token")))' 2>/dev/null \
    | grep -q True
}

bridge_deployed() {
  "$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-grafana-bridge svc/netobserv-grafana-bridge >/dev/null 2>&1
}

bridge_ready() {
  local nready bridge_mtls bridge_ready
  bridge_mtls="$("$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-grafana-bridge \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="bridge")].env[?(@.name=="SPIFFE_MTLS")].value}' 2>/dev/null || echo "")"
  if [[ "$bridge_mtls" == "1" ]]; then
    bridge_ready="$("$KUBECTL" -n "$OPENCLAW_NS" get pods -l app.kubernetes.io/name=netobserv-grafana-bridge \
      -o jsonpath='{.items[?(@.status.phase=="Running")].status.containerStatuses[*].ready}' 2>/dev/null || true)"
    [[ "$bridge_ready" == *"true true"* || "$bridge_ready" == *"truetrue"* ]]
    return
  fi
  nready="$("$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-grafana-bridge \
    -o jsonpath='{.status.readyReplicas}/{.spec.replicas}' 2>/dev/null || echo 0/0)"
  [[ "${nready:-0/0}" == "1/1" ]]
}

slack_prereq() {
  "$KUBECTL" -n "$OPENCLAW_NS" get secret/openclaw-slack-tokens >/dev/null 2>&1
}

cmd_plan() {
  cat <<EOF
Phase 9 — Event-driven AIOps (~10 min wire + metrics sync)

Guide: $GF_ROOT/docs/PHASE-9-EVENT.md

Prereq: Phase 8 (Slack) + Phase 4 (Grafana + OTel). site.slack_channel_id in site config.

Path: NetObserv RTT → Grafana alert → netobserv-grafana-bridge → OpenClaw /hooks/agent → Slack

Steps:
  1. Sync demo metrics (Thanos federation → openclaw-otel-prometheus)
       ./scripts/phase9-event.sh metrics
  2. Wire hooks + bridge + Grafana alert
       ./scripts/phase9-event.sh deploy

Or manual:
       SLACK_CHANNEL_ID=C… $KIT/wire-openclaw-event-aiops.sh all

Verify:
  ./scripts/phase9-event.sh verify
  $KIT/netobserv-e2e-openclaw-test.sh event-aiops-check

Demo: $KIT/prepare-event-demo.sh  (or demo-a-fast → wait ~2m for auto-investigate)
EOF
}

cmd_metrics() {
  step "Sync Grafana demo metrics"
  "$KIT/sync-grafana-demo-metrics.sh" sync
  ok "Metrics federation synced"
}

cmd_hooks() {
  step "Wire OpenClaw hooks + Grafana bridge"
  load_event_env
  slack_prereq || { fail "Phase 8 required — run phase8-slack.sh deploy"; exit 1; }
  [[ -n "${SLACK_CHANNEL_ID:-}" ]] || { fail "site.slack_channel_id / SLACK_CHANNEL_ID required"; exit 1; }
  [[ -f "$LAB_CFG" ]] || "$KIT/clone-openshell-lab.sh" install
  RECYCLE_POD="${RECYCLE_POD:-1}" SKIP_SMOKE="${SKIP_SMOKE:-0}" \
    SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" OPENCLAW_NS="$OPENCLAW_NS" \
    "$KIT/wire-openclaw-hooks.sh"
  ok "Hooks + netobserv-grafana-bridge wired"
}

cmd_alerts() {
  step "Wire Grafana alert + contact point"
  "$KIT/wire-grafana-openclaw-alerts.sh" all
  ok "Grafana event alert provisioned"
}

cmd_wire() {
  cmd_hooks
  cmd_alerts
  ok "Event path wired — run: ./scripts/phase9-event.sh verify"
}

cmd_deploy() {
  step "Phase 9 deploy — event-AIOps"
  load_event_env

  gate "Phase 8 — openclaw-slack-tokens" slack_prereq || { fail "Run phase8-slack.sh deploy first"; exit 1; }
  gate "Phase 4 — Grafana route" "$KUBECTL" get route grafana-network-aiops -n "$GRAFANA_NS" || {
    fail "Grafana missing — run phase4-grafana.sh deploy"; exit 1
  }
  gate "Phase 4 — openclaw-otel-prometheus" "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw-otel-prometheus || {
    fail "OTel Prometheus missing — run phase4-grafana.sh deploy"; exit 1
  }

  cmd_metrics
  if hooks_secret_present && bridge_deployed; then
    ok "Hooks + bridge already present — refreshing wire"
    RECYCLE_POD="${RECYCLE_POD:-0}" cmd_wire
  else
    RECYCLE_POD="${RECYCLE_POD:-1}" cmd_wire
  fi
  ok "Deploy complete — run: ./scripts/phase9-event.sh verify"
}

cmd_check() {
  local fails=0
  step "Phase 9 readiness — $($KUBECTL whoami 2>/dev/null || echo '?')"
  load_event_env

  gate "Phase 8 — Slack secret" slack_prereq || fails=$((fails + 1))
  gate "Grafana route (Phase 4)" "$KUBECTL" get route grafana-network-aiops -n "$GRAFANA_NS" || fails=$((fails + 1))
  gate "openclaw-otel-prometheus" "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw-otel-prometheus || fails=$((fails + 1))
  gate "openclaw-hooks-token secret" hooks_secret_present || fails=$((fails + 1))
  if hooks_env_wired; then
    ok "OPENCLAW_HOOKS_TOKEN on deployment/openclaw"
  else
    fail "OPENCLAW_HOOKS_TOKEN env missing — run phase9-event.sh hooks"
    fails=$((fails + 1))
  fi
  if hooks_configured; then
    ok "hooks.enabled in openclaw-config"
  else
    fail "hooks config missing — run phase9-event.sh hooks"
    fails=$((fails + 1))
  fi
  if bridge_deployed; then
    ok "netobserv-grafana-bridge deployed"
  else
    fail "netobserv-grafana-bridge missing — run phase9-event.sh hooks"
    fails=$((fails + 1))
  fi
  if bridge_ready; then
    ok "netobserv-grafana-bridge ready"
  else
    fail "netobserv-grafana-bridge not ready — oc get pods -n $OPENCLAW_NS -l app.kubernetes.io/name=netobserv-grafana-bridge"
    fails=$((fails + 1))
  fi
  if "$KIT/wire-grafana-openclaw-alerts.sh" status >/dev/null 2>&1; then
    ok "Grafana contact point + RTT alert rule"
  else
    fail "Grafana alert not provisioned — run phase9-event.sh alerts"
    fails=$((fails + 1))
  fi
  if "$KIT/sync-grafana-demo-metrics.sh" status >/dev/null 2>&1; then
    ok "NetObserv metrics federated into demo Prometheus"
  else
    warn "Metrics federation not verified — run phase9-event.sh metrics"
  fi

  printf '\n'
  if [[ "$fails" -gt 0 ]]; then
    warn "$fails check(s) failed — see $GF_ROOT/docs/PHASE-9-EVENT.md#troubleshooting"
    return 1
  fi
  ok "Phase 9 checks passed"
  printf '  Full path: %s/netobserv-e2e-openclaw-test.sh event-aiops-check\n' "$KIT"
  return 0
}

cmd_status() {
  step "Event-AIOps status"
  "$KIT/wire-openclaw-event-aiops.sh" status 2>/dev/null || true
}

cmd_verify() {
  cmd_check || true
  step "event-aiops-check (demo kit)"
  "$KIT/netobserv-e2e-openclaw-test.sh" event-aiops-check
}

case "${1:-plan}" in
  plan|help|-h|--help) cmd_plan ;;
  metrics|sync-metrics) cmd_metrics ;;
  hooks) cmd_hooks ;;
  alerts) cmd_alerts ;;
  wire) cmd_wire ;;
  deploy|install) cmd_deploy ;;
  check|verify-gates) cmd_check ;;
  status) cmd_status ;;
  verify|test) cmd_verify ;;
  *)
    echo "usage: $0 {plan|metrics|hooks|alerts|wire|deploy|check|status|verify}" >&2
    exit 1
    ;;
esac
