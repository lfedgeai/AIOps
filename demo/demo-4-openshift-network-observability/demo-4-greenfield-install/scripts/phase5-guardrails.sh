#!/usr/bin/env bash
# Phase 5 helper — TrustyAI Guardrails + OpenClaw wire (greenfield).
#
# Usage:
#   ./scripts/phase5-guardrails.sh plan
#   ./scripts/phase5-guardrails.sh deploy
#   ./scripts/phase5-guardrails.sh check
#   ./scripts/phase5-guardrails.sh status
#   ./scripts/phase5-guardrails.sh verify
#
# Env: DEMO_KIT_ROOT, GUARDRAILS_NS, OPENCLAW_NS, LLM_BASE_URL (from site config)
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
GUARDRAILS_NS="${GUARDRAILS_NS:-netobserv-guardrails}"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
GATEWAY_PRESET="${GATEWAY_PRESET:-netobserv-sre}"
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

openclaw_wired_to_guardrails() {
  local url
  url="$("$KUBECTL" -n "$OPENCLAW_NS" get configmap openclaw-config \
    -o jsonpath='{.data.openclaw\.json}' 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("models",{}).get("providers",{}).get("openai",{}).get("baseUrl",""))' 2>/dev/null || true)"
  [[ -n "$url" && "$url" == *"netobserv-llm-guard-proxy"* ]]
}

cmd_plan() {
  # shellcheck source=/dev/null
  source "$KIT/supply-chain-pins.env" 2>/dev/null || true
  cat <<EOF
Phase 5 — TrustyAI Guardrails (~30–60 min, DSC + operator wait)

Guide: $GF_ROOT/docs/PHASE-5-GUARDRAILS.md
Upstream: ${LEMONADE_REPO:-https://github.com/rh-ai-quickstart/lemonade-stand-assistant.git}
  branch ${LEMONADE_BRANCH:-nemo-guardrails} @ ${LEMONADE_COMMIT:-b342f224…}
  chart fms-orchestrator/chart — Option A external MaaS (no in-cluster Llama GPU)

Prereq: Phase 2 (openclaw + LLM). Phase 3 recommended (DSC already present).

Steps:
  1. Enable TrustyAI + KServe in DSC; install FMS orchestrator Helm chart
       $KIT/install-trustyai-guardrails.sh install
  2. Wire OpenClaw LLM → guard proxy (Layer 0)
       $KIT/wire-openclaw-trustyai-guardrails.sh all
  3. Re-wire MLflow for guard proxy (if Phase 3 done)
       MLFLOW_BACKEND=rhoai $KIT/wire-openclaw-mlflow.sh

Or: ./scripts/phase5-guardrails.sh deploy

Verify:
  ./scripts/phase5-guardrails.sh check
  $KIT/netobserv-e2e-openclaw-test.sh trustyai-guard-check
EOF
}

cmd_check() {
  local fails=0
  step "Phase 5 readiness — $($KUBECTL whoami 2>/dev/null || echo '?')"

  gate "Phase 2 — openclaw deployment" "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw || fails=$((fails + 1))
  gate "Guardrails namespace" "$KUBECTL" get ns "$GUARDRAILS_NS" || fails=$((fails + 1))
  gate "TrustyAI CRD" "$KUBECTL" get crd guardrailsorchestrators.trustyai.opendatahub.io || fails=$((fails + 1))
  gate "GuardrailsOrchestrator CR" \
    "$KUBECTL" -n "$GUARDRAILS_NS" get guardrailsorchestrator/guardrails-orchestrator || fails=$((fails + 1))
  gate "LLM guard proxy deployment" \
    "$KUBECTL" -n "$GUARDRAILS_NS" get deploy/netobserv-llm-guard-proxy || fails=$((fails + 1))

  if openclaw_wired_to_guardrails; then
    ok "OpenClaw baseUrl → TrustyAI guard proxy"
  else
    fail "OpenClaw not wired to guard proxy — run wire-openclaw-trustyai-guardrails.sh"
    fails=$((fails + 1))
  fi

  if "$KUBECTL" -n "$GUARDRAILS_NS" rollout status deploy/netobserv-llm-guard-proxy --timeout=30s >/dev/null 2>&1; then
    ok "Guard proxy rollout healthy"
  else
    fail "Guard proxy not ready (cold start pip install can take ~90s)"
    fails=$((fails + 1))
  fi

  printf '\n'
  if [[ "$fails" -gt 0 ]]; then
    warn "$fails check(s) failed — see $GF_ROOT/docs/PHASE-5-GUARDRAILS.md#troubleshooting"
    return 1
  fi
  ok "Phase 5 checks passed"
  printf '  Full path: %s/netobserv-e2e-openclaw-test.sh trustyai-guard-check\n' "$KIT"
  return 0
}

cmd_status() {
  step "TrustyAI / guardrails status"
  "$KIT/install-trustyai-guardrails.sh" status 2>/dev/null || true
  "$KIT/wire-openclaw-trustyai-guardrails.sh" status 2>/dev/null || true
}

cmd_deploy() {
  step "Phase 5 automated deploy"
  site_config_ensure openshell || warn "site config incomplete — LLM_BASE_URL may be missing"
  site_config_apply openshell 2>/dev/null || true

  "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw >/dev/null 2>&1 \
    || { fail "Phase 2 required"; exit 1; }

  step "Install TrustyAI Guardrails (DSC patch + Helm)"
  warn "TrustyAI CRD can take 15–30 min on first install"
  "$KIT/install-trustyai-guardrails.sh" install \
    || { fail "install-trustyai-guardrails.sh failed"; exit 1; }

  step "Wire OpenClaw → TrustyAI gateway"
  "$KIT/wire-openclaw-trustyai-guardrails.sh" all \
    || { fail "wire-openclaw-trustyai-guardrails.sh failed"; exit 1; }

  if "$KUBECTL" -n "${RHOAI_NS:-redhat-ods-applications}" get deploy/mlflow >/dev/null 2>&1; then
    step "Re-wire MLflow for guard proxy path (Phase 3)"
    MLFLOW_BACKEND=rhoai MLFLOW_REMOVE_STANDALONE=0 "$KIT/wire-openclaw-mlflow.sh" \
      || warn "MLflow re-wire failed — audit traces may be stale until re-run"
  fi

  ok "Deploy complete — run: ./scripts/phase5-guardrails.sh check"
}

cmd_verify() {
  cmd_check || true
  step "trustyai-guard-check (demo kit)"
  "$KIT/netobserv-e2e-openclaw-test.sh" trustyai-guard-check
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
