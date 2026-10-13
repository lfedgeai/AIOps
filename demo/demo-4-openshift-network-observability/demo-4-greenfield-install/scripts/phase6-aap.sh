#!/usr/bin/env bash
# Phase 6 helper — AAP + Gitea + ansible-automation MCP (greenfield).
#
# Usage:
#   ./scripts/phase6-aap.sh plan
#   ./scripts/phase6-aap.sh deploy    # install AAP only — stops for manual license (Gateway UI)
#   ./scripts/phase6-aap.sh wire        # after license: Gitea + bootstrap + ansible-mcp
#   ./scripts/phase6-aap.sh check
#   ./scripts/phase6-aap.sh status
#   ./scripts/phase6-aap.sh verify
#
# Env: DEMO_KIT_ROOT, AAP_NS, GITEA_NS, OPENCLAW_NS
# License is always manual (Gateway UI) — never read from site config or git.
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
AAP_NS="${AAP_NS:-ansible-automation-platform}"
AAP_INSTANCE="${AAP_INSTANCE:-netobserv-aap}"
GITEA_NS="${GITEA_NS:-gitea}"
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

aap_instance_ready() {
  local successful failure
  successful="$("$KUBECTL" get ansibleautomationplatform "$AAP_INSTANCE" -n "$AAP_NS" \
    -o jsonpath='{.status.conditions[?(@.type=="Successful")].status}' 2>/dev/null || true)"
  failure="$("$KUBECTL" get ansibleautomationplatform "$AAP_INSTANCE" -n "$AAP_NS" \
    -o jsonpath='{.status.conditions[?(@.type=="Failure")].status}' 2>/dev/null || true)"
  [[ "$successful" == "True" && "$failure" != "True" ]]
}

aap_gateway_url() {
  "$KIT/install-aap.sh" status 2>/dev/null | sed -n 's/^\[ ok \] Gateway URL: //p' | head -1
}

cmd_plan() {
  # shellcheck source=/dev/null
  source "$KIT/supply-chain-pins.env" 2>/dev/null || true
  cat <<EOF
Phase 6 — AAP + Gitea + ansible-mcp (~20–40 min operator wait + manual license)

Guide: $GF_ROOT/docs/PHASE-6-AAP.md

Prereq: Phase 2 (openclaw). Phases 3–5 recommended (MLflow audit, heal context).

Components:
  - AAP 2.7 operator + minimal instance (controller + gateway; hub disabled)
  - Gitea (quay.io/${QUAY_ORG:-your-org}/gitea-openshift:${GITEA_TAG:-gitea-openshift-v1}) — playbook SCM
  - ansible-mcp + openclaw-aap-launcher (governed heal job templates)

Steps:
  1. Install AAP operator + instance (stops before wire)
       ./scripts/phase6-aap.sh deploy
  2. YOU — apply subscription in AAP Gateway UI (manifest not stored in git)
  3. Wire Gitea + bootstrap templates + ansible-mcp
       ./scripts/phase6-aap.sh wire

Verify:
  ./scripts/phase6-aap.sh check
  $KIT/netobserv-e2e-openclaw-test.sh aap-check
EOF
}

cmd_check() {
  local fails=0
  step "Phase 6 readiness — $($KUBECTL whoami 2>/dev/null || echo '?')"

  gate "Phase 2 — openclaw deployment" "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw || fails=$((fails + 1))
  gate "AAP namespace" "$KUBECTL" get ns "$AAP_NS" || fails=$((fails + 1))
  gate "AAP operator CSV" "$KUBECTL" get csv -n "$AAP_NS" --no-headers 2>/dev/null | grep -q . || fails=$((fails + 1))
  gate "AnsibleAutomationPlatform/$AAP_INSTANCE" \
    "$KUBECTL" -n "$AAP_NS" get ansibleautomationplatform/"$AAP_INSTANCE" || fails=$((fails + 1))

  if aap_instance_ready; then
    ok "AAP instance Successful (AAP 2.7)"
  else
    fail "AAP instance not Successful — oc get ansibleautomationplatform -n $AAP_NS -o yaml | tail -30"
    fails=$((fails + 1))
  fi

  gate "Gitea statefulset" "$KUBECTL" -n "$GITEA_NS" get statefulset/gitea || fails=$((fails + 1))
  gate "aap-netobserv-heal SA" "$KUBECTL" -n "$OPENCLAW_NS" get sa/aap-netobserv-heal || fails=$((fails + 1))
  gate "openclaw-aap-launcher secret" "$KUBECTL" -n "$OPENCLAW_NS" get secret/openclaw-aap-launcher || fails=$((fails + 1))
  gate "ansible-mcp deployment" "$KUBECTL" -n "$OPENCLAW_NS" get deploy/ansible-mcp || fails=$((fails + 1))

  if "$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy/ansible-mcp --timeout=30s >/dev/null 2>&1; then
    ok "ansible-mcp rollout healthy"
  else
    fail "ansible-mcp not ready"
    fails=$((fails + 1))
  fi

  if "$KIT/wire-openclaw-aap.sh" status 2>&1 | grep -q '^\[ ok \] netobserv-heal-db-path'; then
    ok "AAP job templates + confirm gate wired"
  else
    fail "AAP job templates incomplete — run wire-openclaw-aap.sh bootstrap"
    fails=$((fails + 1))
  fi

  printf '\n'
  if [[ "$fails" -gt 0 ]]; then
    warn "$fails check(s) failed — see $GF_ROOT/docs/PHASE-6-AAP.md#troubleshooting"
    return 1
  fi
  ok "Phase 6 checks passed"
  printf '  Full path: %s/netobserv-e2e-openclaw-test.sh aap-check\n' "$KIT"
  return 0
}

cmd_status() {
  step "AAP / Gitea / ansible-mcp status"
  "$KIT/install-aap.sh" status 2>/dev/null || true
  "$KIT/install-gitea.sh" status 2>/dev/null || true
  "$KIT/wire-openclaw-aap.sh" status 2>/dev/null || true
}

print_license_handoff() {
  local url pass_hint
  url="$(aap_gateway_url || true)"
  step "Manual step — AAP subscription license (Gateway UI)"
  printf '\n'
  printf '  Install is complete. Apply your subscription manifest in the AAP Gateway UI.\n'
  printf '  Do not commit the license file to git — upload it only in the browser.\n\n'
  if [[ -n "$url" ]]; then
    printf '  Gateway URL: %s\n' "$url"
  else
    printf '  Gateway URL: ./scripts/phase6-aap.sh status\n'
  fi
  printf '  Login: admin\n'
  printf '  Password: oc -n %s get secret netobserv-aap-admin-password \\\n' "$AAP_NS"
  printf '    -o jsonpath="{.data.password}" | base64 -d; echo\n\n'
  printf '  When license shows active, continue:\n'
  printf '    ./scripts/phase6-aap.sh wire\n\n'
}

cmd_deploy() {
  step "Phase 6 deploy — AAP install only (license is manual)"
  site_config_ensure openshell 2>/dev/null || true

  "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw >/dev/null 2>&1 \
    || { fail "Phase 2 required"; exit 1; }

  if [[ -x "$KIT/verify-image-pulls.sh" ]]; then
    step "Verify Gitea image pull (supply chain)"
    QUAY_ORG="${QUAY_ORG:-your-org}" "$KIT/verify-image-pulls.sh" --gitea 2>/dev/null \
      || warn "Gitea image pull check failed — see docs/IMAGE-MIRRORS.md"
  fi

  if aap_instance_ready; then
    ok "AAP instance already Ready — skipping install"
  else
    step "Install AAP operator + minimal instance"
    warn "AAP operator + instance can take 20–40 min on first install"
    AAP_NS="$AAP_NS" AAP_INSTANCE="$AAP_INSTANCE" "$KIT/install-aap.sh" install \
      || { fail "install-aap.sh failed"; exit 1; }
  fi

  print_license_handoff
}

cmd_wire() {
  step "Phase 6 wire — Gitea + AAP bootstrap + ansible-mcp"
  aap_instance_ready || { fail "AAP instance not Ready — finish deploy first"; exit 1; }

  step "Wire Gitea + AAP bootstrap + ansible-mcp"
  warn "Requires active AAP subscription (applied manually in Gateway UI)"
  AAP_NS="$AAP_NS" OPENCLAW_NS="$OPENCLAW_NS" "$KIT/wire-openclaw-aap.sh" all \
    || { fail "wire-openclaw-aap.sh failed — is the license active?"; exit 1; }

  ok "Wire complete — run: ./scripts/phase6-aap.sh check"
}

cmd_verify() {
  cmd_check || true
  step "aap-check (demo kit)"
  "$KIT/netobserv-e2e-openclaw-test.sh" aap-check
}

case "${1:-plan}" in
  plan|help|-h|--help) cmd_plan ;;
  deploy|install) cmd_deploy ;;
  wire|bootstrap) cmd_wire ;;
  check|verify-gates) cmd_check ;;
  status) cmd_status ;;
  verify|test) cmd_verify ;;
  *)
    echo "usage: $0 {plan|deploy|wire|check|status|verify}" >&2
    exit 1
    ;;
esac
