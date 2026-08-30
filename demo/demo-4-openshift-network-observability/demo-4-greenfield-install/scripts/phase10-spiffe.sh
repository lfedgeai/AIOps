#!/usr/bin/env bash
# Phase 10 helper — ZTWI / SPIFFE mTLS for event-AIOps path.
#
# Usage:
#   ./scripts/phase10-spiffe.sh plan
#   ./scripts/phase10-spiffe.sh operator   # ZTWI operator subscription
#   ./scripts/phase10-spiffe.sh operands   # SPIRE Server/Agent/CSI CRs
#   ./scripts/phase10-spiffe.sh ztwi       # operator + operands
#   ./scripts/phase10-spiffe.sh wire       # mTLS bridge + openclaw-hooks-mtls
#   ./scripts/phase10-spiffe.sh deploy     # ztwi + wire
#   ./scripts/phase10-spiffe.sh repair     # restart SPIRE after CA drift
#   ./scripts/phase10-spiffe.sh rollback   # revert to Bearer-token bridge
#   ./scripts/phase10-spiffe.sh check
#   ./scripts/phase10-spiffe.sh status
#   ./scripts/phase10-spiffe.sh verify
#
# Env: DEMO_KIT_ROOT, OPENCLAW_NS, SLACK_CHANNEL_ID, SPIRE_STORAGE_CLASS
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

load_spiffe_env() {
  site_config_ensure spiffe 2>/dev/null || site_config_ensure event || site_config_ensure slack || true
  eval "$(site_config_load)" 2>/dev/null || true
  export SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}"
  if [[ -z "${SPIRE_STORAGE_CLASS:-}" ]]; then
    SPIRE_STORAGE_CLASS="$(python3 -c "
import yaml
from pathlib import Path
p=Path('$GF_ROOT/config/site-secrets.local.yaml')
if p.exists():
    d=yaml.safe_load(p.read_text()) or {}
    print((d.get('openshift') or {}).get('storage_class','') or '')
" 2>/dev/null || true)"
    export SPIRE_STORAGE_CLASS="${SPIRE_STORAGE_CLASS:-gp3-csi}"
  fi
}

ztwi_crd_present() {
  "$KUBECTL" get crd clusterspiffeids.spire.spiffe.io >/dev/null 2>&1
}

ztwi_ready() {
  [[ "$("$KUBECTL" get ZeroTrustWorkloadIdentityManager cluster \
    -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "")" == "True" ]]
}

spire_operands_present() {
  "$KUBECTL" get spireserver/cluster spireagent/cluster spiffecsidriver/cluster >/dev/null 2>&1
}

event_prereq() {
  "$KUBECTL" -n "$OPENCLAW_NS" get secret/openclaw-hooks-token deploy/netobserv-grafana-bridge >/dev/null 2>&1
}

clusterspiffeid_present() {
  "$KUBECTL" get clusterspiffeid/netobserv-grafana-bridge clusterspiffeid/openclaw-hooks-mtls >/dev/null 2>&1
}

hooks_mtls_ready() {
  [[ "$("$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw-hooks-mtls \
    -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)" == "1" ]]
}

bridge_spiffe_mtls() {
  [[ "$("$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-grafana-bridge \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="bridge")].env[?(@.name=="SPIFFE_MTLS")].value}' 2>/dev/null || echo "")" == "1" ]]
}

bridge_spiffe_ready() {
  local bridge_ready
  bridge_ready="$("$KUBECTL" -n "$OPENCLAW_NS" get pods -l app.kubernetes.io/name=netobserv-grafana-bridge \
    -o jsonpath='{.items[?(@.status.phase=="Running")].status.containerStatuses[*].ready}' 2>/dev/null || true)"
  [[ "$bridge_ready" == *"true true"* || "$bridge_ready" == *"truetrue"* ]]
}

cmd_plan() {
  cat <<EOF
Phase 10 — ZTWI / SPIFFE mTLS (~20–40 min first install)

Guide: $GF_ROOT/docs/PHASE-10-SPIFFE.md

Prereq: Phase 9 (event-AIOps). site.slack_channel_id in site config.

Upgrades Grafana bridge → OpenClaw hooks path from Bearer token to SPIFFE mTLS:
  Grafana → netobserv-grafana-bridge ──mTLS──▶ openclaw-hooks-mtls → OpenClaw /hooks/agent

Steps:
  1. Install ZTWI operator + SPIRE operands (once per cluster)
       ./scripts/phase10-spiffe.sh ztwi
  2. Wire mTLS workloads + ClusterSPIFFEID
       ./scripts/phase10-spiffe.sh wire

Or: ./scripts/phase10-spiffe.sh deploy

Verify:
  ./scripts/phase10-spiffe.sh verify
  $KIT/netobserv-e2e-openclaw-test.sh spiffe-check

After cluster reboot: ./scripts/phase10-spiffe.sh repair
EOF
}

cmd_operator() {
  step "Install ZTWI operator"
  "$KIT/install-ztwi-spire.sh" operator
  ok "Operator install initiated — run: ./scripts/phase10-spiffe.sh operands when CSV Succeeded"
}

cmd_operands() {
  step "Deploy SPIRE operands"
  SPIRE_STORAGE_CLASS="${SPIRE_STORAGE_CLASS:-gp3-csi}" "$KIT/install-ztwi-spire.sh" operands
  ok "SPIRE operands applied"
}

cmd_ztwi() {
  step "Install ZTWI + SPIRE (operator + operands)"
  SPIRE_STORAGE_CLASS="${SPIRE_STORAGE_CLASS:-gp3-csi}" "$KIT/install-ztwi-spire.sh" all
  ok "ZTWI install complete — verify: ./scripts/phase10-spiffe.sh check"
}

cmd_wire() {
  step "Wire SPIFFE mTLS (bridge + openclaw-hooks-mtls)"
  load_spiffe_env
  event_prereq || { fail "Phase 9 required — run phase9-event.sh deploy"; exit 1; }
  [[ -n "${SLACK_CHANNEL_ID:-}" ]] || { fail "site.slack_channel_id / SLACK_CHANNEL_ID required"; exit 1; }
  ztwi_ready || { fail "ZTWI not Ready — run phase10-spiffe.sh ztwi or repair"; exit 1; }
  SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" OPENCLAW_NS="$OPENCLAW_NS" \
    "$KIT/wire-openclaw-spiffe.sh" all
  ok "SPIFFE mTLS wired — run: ./scripts/phase10-spiffe.sh verify"
}

cmd_deploy() {
  step "Phase 10 deploy — SPIFFE mTLS"
  load_spiffe_env

  gate "Phase 9 — hooks + grafana-bridge" event_prereq || {
    fail "Run phase9-event.sh deploy first"; exit 1
  }
  [[ -n "${SLACK_CHANNEL_ID:-}" ]] || { fail "site.slack_channel_id required"; exit 1; }

  if ztwi_ready && spire_operands_present; then
    ok "ZTWI already Ready — skipping ztwi install"
  else
    cmd_ztwi
  fi

  if clusterspiffeid_present && bridge_spiffe_mtls && hooks_mtls_ready; then
    ok "SPIFFE workloads present — refreshing wire"
    SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" OPENCLAW_NS="$OPENCLAW_NS" \
      "$KIT/wire-openclaw-spiffe.sh" all || cmd_wire
  else
    cmd_wire
  fi
  ok "Deploy complete — run: ./scripts/phase10-spiffe.sh verify"
}

cmd_repair() {
  step "Repair SPIRE / SPIFFE workloads"
  "$KIT/install-ztwi-spire.sh" repair
  ok "Repair complete — run: ./scripts/phase10-spiffe.sh verify"
}

cmd_rollback() {
  step "Rollback to Bearer-token event path"
  load_spiffe_env
  SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}" "$KIT/wire-openclaw-spiffe.sh" rollback
  ok "Rolled back — event path uses Bearer token on bridge again"
}

cmd_check() {
  local fails=0
  step "Phase 10 readiness — $($KUBECTL whoami 2>/dev/null || echo '?')"
  load_spiffe_env

  gate "Phase 9 — event-AIOps prereq" event_prereq || fails=$((fails + 1))
  if ztwi_crd_present; then
    ok "ClusterSPIFFEID CRD (ZTWI installed)"
  else
    fail "ZTWI not installed — run phase10-spiffe.sh ztwi"
    fails=$((fails + 1))
  fi
  if ztwi_ready; then
    ok "ZeroTrustWorkloadIdentityManager Ready"
  else
    fail "ZTWI not Ready — run phase10-spiffe.sh ztwi or repair"
    fails=$((fails + 1))
  fi
  if spire_operands_present; then
    ok "SPIRE operands (server, agent, CSI)"
  else
    fail "SPIRE operands missing — run phase10-spiffe.sh operands"
    fails=$((fails + 1))
  fi
  if clusterspiffeid_present; then
    ok "ClusterSPIFFEID (bridge + hooks-mtls)"
  else
    fail "ClusterSPIFFEID missing — run phase10-spiffe.sh wire"
    fails=$((fails + 1))
  fi
  if hooks_mtls_ready; then
    ok "openclaw-hooks-mtls 1/1"
  else
    fail "openclaw-hooks-mtls not ready"
    fails=$((fails + 1))
  fi
  if bridge_spiffe_mtls; then
    ok "Bridge SPIFFE_MTLS=1 (Bearer removed from bridge pod)"
  else
    fail "Bridge not on SPIFFE mTLS — run phase10-spiffe.sh wire"
    fails=$((fails + 1))
  fi
  if bridge_spiffe_ready; then
    ok "netobserv-grafana-bridge ready (bridge + spiffe-helper)"
  else
    fail "netobserv-grafana-bridge not 2/2 — try phase10-spiffe.sh repair"
    fails=$((fails + 1))
  fi

  printf '\n'
  if [[ "$fails" -gt 0 ]]; then
    warn "$fails check(s) failed — see $GF_ROOT/docs/PHASE-10-SPIFFE.md#troubleshooting"
    return 1
  fi
  ok "Phase 10 checks passed"
  printf '  Full path: %s/netobserv-e2e-openclaw-test.sh spiffe-check\n' "$KIT"
  return 0
}

cmd_status() {
  step "ZTWI / SPIFFE status"
  "$KIT/install-ztwi-spire.sh" status 2>/dev/null || true
  SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}" "$KIT/wire-openclaw-spiffe.sh" status 2>/dev/null || true
}

cmd_verify() {
  load_spiffe_env
  cmd_check || true
  step "spiffe-check (demo kit)"
  SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}" "$KIT/netobserv-e2e-openclaw-test.sh" spiffe-check
}

case "${1:-plan}" in
  plan|help|-h|--help) cmd_plan ;;
  operator) cmd_operator ;;
  operands) cmd_operands ;;
  ztwi|install) cmd_ztwi ;;
  wire) cmd_wire ;;
  deploy|install-all) cmd_deploy ;;
  repair) cmd_repair ;;
  rollback) cmd_rollback ;;
  check|verify-gates) cmd_check ;;
  status) cmd_status ;;
  verify|test) cmd_verify ;;
  *)
    echo "usage: $0 {plan|operator|operands|ztwi|wire|deploy|repair|rollback|check|status|verify}" >&2
    exit 1
    ;;
esac
