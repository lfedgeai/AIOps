#!/usr/bin/env bash
# Phase 11 helper — RHCL / Kuadrant OAuth front door for OpenClaw (optional).
#
# Usage:
#   ./scripts/phase11-rhcl.sh plan
#   ./scripts/phase11-rhcl.sh operator   # RHCL operator + Kuadrant CR
#   ./scripts/phase11-rhcl.sh ingress    # dedicated gateway + OpenShift OAuth
#   ./scripts/phase11-rhcl.sh deploy     # operator + ingress (default full path)
#   ./scripts/phase11-rhcl.sh check
#   ./scripts/phase11-rhcl.sh status
#   ./scripts/phase11-rhcl.sh verify
#
# Env: DEMO_KIT_ROOT, OPENCLAW_NS, SKIP_RHCL, SKIP_RHCL_INGRESS, OPENCLAW_RHCL_HOST
set -euo pipefail

GF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GF_ROOT
[[ -f "$GF_ROOT/config/env.local" ]] && source "$GF_ROOT/config/env.local"
# shellcheck source=scripts/resolve-demo-kit.sh
source "$GF_ROOT/scripts/resolve-demo-kit.sh"
resolve_demo_kit

KUBECTL="$(command -v oc || command -v kubectl)"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
SKIP_RHCL="${SKIP_RHCL:-0}"
SKIP_RHCL_INGRESS="${SKIP_RHCL_INGRESS:-0}"
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

openclaw_prereq() {
  "$KUBECTL" -n "$OPENCLAW_NS" get svc/openclaw deploy/openclaw >/dev/null 2>&1
}

cert_manager_ready() {
  "$KUBECTL" get pods -n cert-manager -l app.kubernetes.io/instance=cert-manager \
    -o jsonpath='{.items[0].status.phase}' 2>/dev/null | grep -q Running
}

kuadrant_ready() {
  [[ "$("$KUBECTL" get kuadrant kuadrant -n kuadrant-system \
    -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "")" == "True" ]]
}

rhcl_csv_succeeded() {
  "$KUBECTL" get csv -n kuadrant-system -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\n"}{end}' 2>/dev/null \
    | grep -E '^rhcl-operator\.' | grep -q Succeeded
}

gateway_present() {
  "$KUBECTL" -n "$OPENCLAW_NS" get gateway/openclaw-gateway >/dev/null 2>&1
}

gateway_programmed() {
  [[ "$("$KUBECTL" get gateway openclaw-gateway -n "$OPENCLAW_NS" \
    -o jsonpath='{.status.listeners[?(@.name=="http")].conditions[?(@.type=="Programmed")].status}' 2>/dev/null || echo "")" == "True" ]]
}

rhcl_route_present() {
  "$KUBECTL" -n "$OPENCLAW_NS" get route/openclaw-rhcl >/dev/null 2>&1
}

oauth_client_present() {
  "$KUBECTL" get oauthclient/openclaw-rhcl >/dev/null 2>&1
}

httproute_accepted() {
  [[ "$("$KUBECTL" get httproute openclaw-ui -n "$OPENCLAW_NS" \
    -o jsonpath='{.status.parents[0].conditions[?(@.type=="Accepted")].status}' 2>/dev/null || echo "")" == "True" ]]
}

rhcl_host() {
  if [[ -n "${OPENCLAW_RHCL_HOST:-}" ]]; then
    echo "$OPENCLAW_RHCL_HOST"
    return
  fi
  "$KUBECTL" -n "$OPENCLAW_NS" get route openclaw-rhcl -o jsonpath='{.spec.host}' 2>/dev/null || true
}

cmd_plan() {
  cat <<EOF
Phase 11 — RHCL OAuth for OpenClaw (optional, ~10–20 min)

Guide: $GF_ROOT/docs/PHASE-11-RHCL.md

Prereq: Phase 2 (openclaw). cert-manager on cluster. OperatorHub: RHCL operator.

Adds enterprise OAuth front door:
  https://openclaw-rhcl.apps.<ingress>/  → OpenShift OAuth → OpenClaw Control UI

Legacy lab Route + gateway token remains available (Slack/MCP unchanged).

Steps:
  1. Install RHCL operator + Kuadrant
       ./scripts/phase11-rhcl.sh operator
  2. Dedicated gateway + Authorino OIDC
       ./scripts/phase11-rhcl.sh ingress

Or: ./scripts/phase11-rhcl.sh deploy

Skip entirely: SKIP_RHCL=1 ./scripts/greenfield-install.sh rhcl
Operator only (no ingress): SKIP_RHCL_INGRESS=1 ./scripts/phase11-rhcl.sh deploy

Verify:
  ./scripts/phase11-rhcl.sh verify
  Incognito: https://openclaw-rhcl.apps.<your-ingress>/
EOF
}

cmd_operator() {
  step "Install RHCL operator + Kuadrant"
  gate "Phase 2 — openclaw service" openclaw_prereq || {
    fail "Run phase2-openshell.sh deploy first"; exit 1
  }
  if cert_manager_ready; then
    ok "cert-manager running"
  else
    warn "cert-manager not detected — RHCL may require it (oc get pods -n cert-manager)"
  fi
  "$KIT/install-rhcl-ingress.sh" install
  ok "RHCL operator + Kuadrant installed"
}

cmd_ingress() {
  step "Wire RHCL ingress (OpenShift OAuth)"
  gate "Phase 2 — openclaw service" openclaw_prereq || {
    fail "Run phase2-openshell.sh deploy first"; exit 1
  }
  kuadrant_ready || { fail "Kuadrant not Ready — run phase11-rhcl.sh operator"; exit 1; }
  PATCH_OPENCLAW_ORIGIN="${PATCH_OPENCLAW_ORIGIN:-1}" \
    OPENCLAW_NS="$OPENCLAW_NS" "$KIT/install-rhcl-ingress.sh" ingress
  step "Re-run OpenShift OAuth patch (idempotent)"
  OPENCLAW_NS="$OPENCLAW_NS" "$KIT/patch-openclaw-oidc-openshift.sh" || \
    warn "OIDC patch incomplete — see docs/PHASE-11-RHCL.md#troubleshooting"
  ok "RHCL ingress wired — test in incognito: https://$(rhcl_host 2>/dev/null || echo 'openclaw-rhcl.apps.<ingress>')/"
}

cmd_deploy() {
  if [[ "$SKIP_RHCL" == "1" ]]; then
    warn "SKIP_RHCL=1 — skipping Phase 11"
    return 0
  fi
  step "Phase 11 deploy — RHCL OAuth"
  gate "Phase 2 — openclaw" openclaw_prereq || {
    fail "Phase 2 required"; exit 1
  }

  if kuadrant_ready && rhcl_csv_succeeded; then
    ok "RHCL operator + Kuadrant already Ready — skipping operator install"
  else
    cmd_operator
  fi

  if [[ "$SKIP_RHCL_INGRESS" == "1" ]]; then
    warn "SKIP_RHCL_INGRESS=1 — operator only, no OAuth ingress"
    ok "Deploy complete (operator only)"
    return 0
  fi

  if gateway_present && rhcl_route_present && oauth_client_present; then
    ok "RHCL ingress already present — refreshing"
    cmd_ingress
  else
    cmd_ingress
  fi
  ok "Deploy complete — run: ./scripts/phase11-rhcl.sh verify"
}

cmd_check() {
  local fails=0 host
  step "Phase 11 readiness — $($KUBECTL whoami 2>/dev/null || echo '?')"

  if [[ "$SKIP_RHCL" == "1" ]]; then
    warn "SKIP_RHCL=1 — phase intentionally skipped"
    return 0
  fi

  gate "Phase 2 — openclaw deployment" openclaw_prereq || fails=$((fails + 1))
  if cert_manager_ready; then
    ok "cert-manager"
  else
    warn "cert-manager not verified — RHCL may need it"
  fi
  if kuadrant_ready; then
    ok "Kuadrant Ready"
  else
    fail "Kuadrant not Ready — run phase11-rhcl.sh operator"
    fails=$((fails + 1))
  fi
  if rhcl_csv_succeeded; then
    ok "RHCL CSV Succeeded"
  else
    fail "RHCL CSV not Succeeded — check oc get csv -n kuadrant-system"
    fails=$((fails + 1))
  fi

  if [[ "$SKIP_RHCL_INGRESS" == "1" ]]; then
    warn "SKIP_RHCL_INGRESS=1 — skipping ingress checks"
    printf '\n'
    [[ "$fails" -eq 0 ]] && ok "Phase 11 operator checks passed"
    return "$([[ "$fails" -eq 0 ]] && echo 0 || echo 1)"
  fi

  if gateway_present; then
    ok "Gateway openclaw-gateway"
  else
    fail "Gateway missing — run phase11-rhcl.sh ingress"
    fails=$((fails + 1))
  fi
  if gateway_programmed; then
    ok "Gateway http listener programmed"
  else
    fail "Gateway not programmed — oc describe gateway openclaw-gateway -n $OPENCLAW_NS"
    fails=$((fails + 1))
  fi
  if rhcl_route_present; then
    ok "Route openclaw-rhcl"
  else
    fail "Route openclaw-rhcl missing"
    fails=$((fails + 1))
  fi
  if oauth_client_present; then
    ok "OAuthClient openclaw-rhcl"
  else
    fail "OAuthClient missing"
    fails=$((fails + 1))
  fi
  if httproute_accepted; then
    ok "HTTPRoute openclaw-ui Accepted"
  else
    fail "HTTPRoute not Accepted — run phase11-rhcl.sh ingress"
    fails=$((fails + 1))
  fi
  if "$KUBECTL" get authpolicy openclaw-ui-oidc -n "$OPENCLAW_NS" \
    -o jsonpath='{.status.conditions[?(@.type=="Enforced")].status}' 2>/dev/null | grep -q True; then
    ok "AuthPolicy openclaw-ui-oidc enforced"
  else
    fail "openclaw-ui-oidc not enforced — run phase11-rhcl.sh ingress (OAuth redirect)"
    fails=$((fails + 1))
  fi
  if "$KUBECTL" get authpolicy openclaw-ui-oidc-callback -n "$OPENCLAW_NS" >/dev/null 2>&1; then
    ok "AuthPolicy openclaw-ui-oidc-callback (OAuth callback)"
  else
    fail "openclaw-ui-oidc-callback missing — re-run patch-openclaw-oidc-openshift.sh"
    fails=$((fails + 1))
  fi
  if "$KUBECTL" get authpolicy openclaw-ui-authorize -n "$OPENCLAW_NS" >/dev/null 2>&1; then
    warn "openclaw-ui-authorize still present (overrides OIDC) — re-run phase11-rhcl.sh ingress"
  fi

  host="$(rhcl_host)"
  if [[ -n "$host" ]]; then
    ok "RHCL URL: https://${host}/"
  fi

  printf '\n'
  if [[ "$fails" -gt 0 ]]; then
    warn "$fails check(s) failed — see $GF_ROOT/docs/PHASE-11-RHCL.md#troubleshooting"
    return 1
  fi
  ok "Phase 11 checks passed"
  printf '  Manual test: incognito login at https://%s/\n' "${host:-openclaw-rhcl.apps.<ingress>}"
  return 0
}

cmd_status() {
  step "RHCL / Kuadrant status"
  "$KIT/install-rhcl-ingress.sh" status 2>/dev/null || true
}

cmd_verify() {
  cmd_check || true
  step "ui-hints (RHCL + legacy URLs)"
  "$KIT/netobserv-e2e-openclaw-test.sh" ui-hints 2>/dev/null | tail -20 || true
}

case "${1:-plan}" in
  plan|help|-h|--help) cmd_plan ;;
  operator|install) cmd_operator ;;
  ingress|oauth) cmd_ingress ;;
  deploy|install-all) cmd_deploy ;;
  check|verify-gates) cmd_check ;;
  status) cmd_status ;;
  verify|test) cmd_verify ;;
  *)
    echo "usage: $0 {plan|operator|ingress|deploy|check|status|verify}" >&2
    exit 1
    ;;
esac
