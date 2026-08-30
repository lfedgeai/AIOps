#!/usr/bin/env bash
# Phase 4 helper — Grafana Network AIOps + OTel federation (greenfield).
#
# Usage:
#   ./scripts/phase4-grafana.sh plan
#   ./scripts/phase4-grafana.sh deploy
#   ./scripts/phase4-grafana.sh check
#   ./scripts/phase4-grafana.sh status
#   ./scripts/phase4-grafana.sh verify
#
# Env: DEMO_KIT_ROOT, GRAFANA_NS (netobserv-demo), OPENCLAW_NS, STORAGE_CLASS
set -euo pipefail

GF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GF_ROOT
[[ -f "$GF_ROOT/config/env.local" ]] && source "$GF_ROOT/config/env.local"
# shellcheck source=scripts/resolve-demo-kit.sh
source "$GF_ROOT/scripts/resolve-demo-kit.sh"
resolve_demo_kit

KUBECTL="$(command -v oc || command -v kubectl)"
GRAFANA_NS="${GRAFANA_NS:-netobserv-demo}"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
GRAFANA_CSV="${GRAFANA_CSV:-grafana-operator.v5.24.0}"
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

cmd_plan() {
  cat <<EOF
Phase 4 — Grafana + OTel (~20–40 min, operator wait)

Guide: $GF_ROOT/docs/PHASE-4-GRAFANA.md

Prereq: Phase 2 complete (openclaw Running). Phase 3 recommended (MLflow audit).

Steps:
  1. Grafana Operator + Network AIOps dashboard
       $KIT/install-grafana-network-aiops.sh all
  2. OTel collector + mini-Prometheus + Grafana datasource/dashboard
       $KIT/install-openclaw-otel-grafana.sh all
  3. Federate NetObserv metrics into demo Prometheus
       $KIT/sync-grafana-demo-metrics.sh sync

Or: ./scripts/phase4-grafana.sh deploy

Verify:
  ./scripts/phase4-grafana.sh check
  ./scripts/phase4-grafana.sh verify
EOF
}

federation_up() {
  "$KIT/sync-grafana-demo-metrics.sh" status >/dev/null 2>&1
}

cmd_check() {
  local fails=0 warns=0
  step "Phase 4 readiness — $($KUBECTL whoami 2>/dev/null || echo '?')"

  gate "Phase 2 — openclaw deployment" "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw || fails=$((fails + 1))
  gate "Grafana namespace ($GRAFANA_NS)" "$KUBECTL" get ns "$GRAFANA_NS" || fails=$((fails + 1))
  gate "Grafana Operator CSV" "$KUBECTL" get csv "$GRAFANA_CSV" -n "$GRAFANA_NS" || fails=$((fails + 1))
  gate "Grafana deployment" "$KUBECTL" -n "$GRAFANA_NS" get deploy/network-aiops-deployment || fails=$((fails + 1))
  gate "Grafana Route" "$KUBECTL" -n "$GRAFANA_NS" get route/grafana-network-aiops || fails=$((fails + 1))
  gate "OTel collector" "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw-otel-collector || fails=$((fails + 1))
  gate "OTel Prometheus" "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw-otel-prometheus || fails=$((fails + 1))
  gate "GrafanaDatasource prometheus-openclaw-otel" \
    "$KUBECTL" -n "$GRAFANA_NS" get grafanadatasource/prometheus-openclaw-otel || fails=$((fails + 1))
  gate "GrafanaDashboard network-aiops-openclaw" \
    "$KUBECTL" -n "$GRAFANA_NS" get grafanadashboard/network-aiops-openclaw || fails=$((fails + 1))

  if "$KUBECTL" -n "$GRAFANA_NS" rollout status deploy/network-aiops-deployment --timeout=30s >/dev/null 2>&1; then
    ok "Grafana rollout healthy"
  else
    fail "Grafana rollout not ready"
    fails=$((fails + 1))
  fi

  if federation_up; then
    ok "NetObserv federation target up (federate-netobserv)"
  else
    warn "Federation not up yet — NetObserv may still be warming; run: $KIT/sync-grafana-demo-metrics.sh sync"
    warns=$((warns + 1))
  fi

  printf '\n'
  if [[ "$fails" -gt 0 ]]; then
    warn "$fails check(s) failed — see $GF_ROOT/docs/PHASE-4-GRAFANA.md#troubleshooting"
    return 1
  fi
  ok "Phase 4 checks passed${warns:+ ($warns warning(s))}"
  local host
  host="$("$KUBECTL" get route grafana-network-aiops -n "$GRAFANA_NS" -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  [[ -n "$host" ]] && printf '  Grafana: https://%s/ (admin / netobserv-demo)\n' "$host"
  return 0
}

cmd_status() {
  step "Grafana / OTel status"
  "$KIT/install-grafana-network-aiops.sh" status 2>/dev/null || true
  "$KIT/install-openclaw-otel-grafana.sh" status 2>/dev/null || true
  "$KIT/sync-grafana-demo-metrics.sh" status 2>/dev/null || warn "metrics federation check failed"
}

cmd_deploy() {
  step "Phase 4 automated deploy"
  "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw >/dev/null 2>&1 \
    || { fail "Phase 2 required — openclaw missing"; exit 1; }

  step "Install Grafana + OTel stack"
  "$KIT/deploy-platform-merge.sh" grafana \
    || { fail "Grafana/OTel install failed"; exit 1; }

  step "Sync NetObserv metrics federation"
  "$KIT/sync-grafana-demo-metrics.sh" sync \
    || warn "metrics sync had issues — re-run: $KIT/sync-grafana-demo-metrics.sh sync"

  ok "Deploy complete — run: ./scripts/phase4-grafana.sh check"
}

cmd_verify() {
  cmd_check || true
  step "Grafana demo prep (e2e kit)"
  if "$KIT/netobserv-e2e-openclaw-test.sh" grafana-recover 2>/dev/null; then
    ok "grafana-recover / demo prep OK"
  else
    warn "grafana-recover incomplete — dashboard may still work after NetObserv warms up"
  fi
}

case "${1:-plan}" in
  plan|help|-h|--help) cmd_plan ;;
  deploy|install) cmd_deploy ;;
  check|verify-gates) cmd_check ;;
  status) cmd_status ;;
  verify|test) cmd_verify ;;
  *)
    echo "usage: $0 {plan|deploy|check|status|verify}" >&2
    exit 1
    ;;
esac
