#!/usr/bin/env bash
# Phase 3 helper — RHOAI + MLflow Traces wire (greenfield).
#
# Usage:
#   ./scripts/phase3-rhoai.sh plan
#   ./scripts/phase3-rhoai.sh check
#   ./scripts/phase3-rhoai.sh status
#   ./scripts/phase3-rhoai.sh verify    # runs demo kit mlflow-check
#
# Env: DEMO_KIT_ROOT, RHOAI_NS (default redhat-ods-applications), OPENCLAW_NS
set -euo pipefail

GF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GF_ROOT
[[ -f "$GF_ROOT/config/env.local" ]] && source "$GF_ROOT/config/env.local"
# shellcheck source=scripts/resolve-demo-kit.sh
source "$GF_ROOT/scripts/resolve-demo-kit.sh"
resolve_demo_kit

KUBECTL="$(command -v oc || command -v kubectl)"
RHOAI_NS="${RHOAI_NS:-redhat-ods-applications}"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
MLFLOW_EXP="${MLFLOW_EXPERIMENT_NAME:-openclaw-netobserv}"
KIT="$DEMO_KIT_ROOT/scripts"

c_green=$'\033[1;32m'; c_red=$'\033[1;31m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '  %s✓%s %s\n' "$c_green" "$c_reset" "$*"; }
fail() { printf '  %s✗%s %s\n' "$c_red" "$c_reset" "$*"; }
warn() { printf '  %s!%s %s\n' "$c_yellow" "$c_reset" "$*"; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }

cmd_plan() {
  cat <<EOF
Phase 3 — RHOAI + MLflow (~30–50 min, mostly operator wait)

Guide: $GF_ROOT/docs/PHASE-3-RHOAI.md

Prereq: Phase 2 complete (openclaw deployment Running)

Steps:
  1. Install minimal RHOAI (dashboard + MLflow only)
       $KIT/install-rhoai-platform-minimal.sh install
  2. Wire OpenClaw → RHOAI MLflow Traces
       MLFLOW_BACKEND=rhoai MLFLOW_REMOVE_STANDALONE=1 $KIT/wire-openclaw-mlflow.sh
  3. Verify
       ./scripts/phase3-rhoai.sh check
       $KIT/netobserv-e2e-openclaw-test.sh mlflow-check

Note: OpenClaw LLM stays on external LiteMaaS/vLLM — RHOAI is for audit/Traces, not inference.
EOF
}

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

rhods_csv_succeeded() {
  local csv phase
  csv="$("$KUBECTL" get csv -n redhat-ods-operator -o json 2>/dev/null | python3 -c "
import json,sys
for item in json.load(sys.stdin).get('items',[]):
    n=item.get('metadata',{}).get('name','')
    if n.startswith('rhods-operator.'):
        print(n); break
" 2>/dev/null || true)"
  [[ -n "$csv" ]] || return 1
  phase="$("$KUBECTL" get csv "$csv" -n redhat-ods-operator -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  [[ "$phase" == "Succeeded" ]]
}

dsc_ready() {
  local phase ready
  phase="$("$KUBECTL" get dsc default-dsc -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  ready="$("$KUBECTL" get dsc default-dsc -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)"
  [[ "$phase" == "Ready" || "$ready" == "True" ]]
}

openclaw_mlflow_wired() {
  local uri
  uri="$("$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="openclaw")].env[?(@.name=="MLFLOW_TRACKING_URI")].value}' 2>/dev/null || true)"
  [[ -n "$uri" && "$uri" == *"mlflow"* && "$uri" == *"$RHOAI_NS"* ]]
}

cmd_check() {
  local fails=0
  step "Phase 3 readiness — $($KUBECTL whoami 2>/dev/null || echo '?')"

  gate "Phase 2 — openclaw deployment" "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw || fails=$((fails + 1))
  gate "rhods-operator CSV Succeeded" rhods_csv_succeeded || fails=$((fails + 1))
  gate "DataScienceCluster default-dsc Ready" dsc_ready || fails=$((fails + 1))
  gate "MLflow deployment ($RHOAI_NS)" "$KUBECTL" -n "$RHOAI_NS" get deploy/mlflow || fails=$((fails + 1))
  gate "MLflow CR" "$KUBECTL" -n "$RHOAI_NS" get mlflow mlflow || fails=$((fails + 1))
  gate "openclaw-netobserv RBAC" "$KUBECTL" -n "$OPENCLAW_NS" get sa/openclaw-netobserv || fails=$((fails + 1))

  if openclaw_mlflow_wired; then
    ok "OpenClaw MLFLOW_TRACKING_URI → RHOAI"
  else
    fail "OpenClaw not wired to RHOAI MLflow — run wire-openclaw-mlflow.sh"
    fails=$((fails + 1))
  fi

  if "$KUBECTL" -n "$RHOAI_NS" get deploy/mlflow >/dev/null 2>&1; then
    if "$KUBECTL" -n "$RHOAI_NS" rollout status deploy/mlflow --timeout=30s >/dev/null 2>&1; then
      ok "MLflow rollout healthy"
    else
      fail "MLflow rollout not ready"
      fails=$((fails + 1))
    fi
  fi

  printf '\n'
  if [[ "$fails" -gt 0 ]]; then
    warn "$fails check(s) failed — see $GF_ROOT/docs/PHASE-3-RHOAI.md#troubleshooting"
    return 1
  fi
  ok "Phase 3 checks passed"
  printf '  Full audit path: %s/netobserv-e2e-openclaw-test.sh mlflow-check\n' "$KIT"
}

cmd_status() {
  step "RHOAI / MLflow status"
  "$KIT/install-rhoai-platform-minimal.sh" status 2>/dev/null || true
  printf '\n'
  if openclaw_mlflow_wired; then
    ok "OpenClaw wired to RHOAI MLflow"
    "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw \
      -o jsonpath='  MLFLOW_TRACKING_URI={.spec.template.spec.containers[?(@.name=="openclaw")].env[?(@.name=="MLFLOW_TRACKING_URI")].value}{"\n"}' 2>/dev/null || true
  else
    warn "OpenClaw MLflow wire not detected"
  fi
}

cmd_verify() {
  cmd_check || true
  step "mlflow-check (demo kit)"
  "$KIT/netobserv-e2e-openclaw-test.sh" mlflow-check
}

case "${1:-plan}" in
  plan|help|-h|--help) cmd_plan ;;
  check|verify-gates) cmd_check ;;
  status) cmd_status ;;
  verify|test) cmd_verify ;;
  *)
    echo "usage: $0 {plan|check|status|verify}" >&2
    exit 1
    ;;
esac
