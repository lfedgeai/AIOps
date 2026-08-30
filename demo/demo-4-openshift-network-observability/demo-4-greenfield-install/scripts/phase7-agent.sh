#!/usr/bin/env bash
# Phase 7 helper — NetObserv agent kit seed (skills + dual MCP).
#
# Usage:
#   ./scripts/phase7-agent.sh plan
#   ./scripts/phase7-agent.sh deploy
#   ./scripts/phase7-agent.sh check
#   ./scripts/phase7-agent.sh status
#   ./scripts/phase7-agent.sh verify
#
# Env: DEMO_KIT_ROOT, OPENCLAW_NS, LAB_CFG, ENABLE_ANSIBLE_MCP (default 1 if Phase 6 wired)
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
OPENSHELL_NS="${OPENSHELL_NS:-openshell}"
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

openclaw_skills_seeded() {
  "$KUBECTL" -n "$OPENCLAW_NS" get configmap openclaw-config \
    -o jsonpath='{.data.openclaw\.json}' 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); s=d.get("skills",{}).get("allowlist",[]) or d.get("agents",{}).get("defaults",{}).get("skills",[]) or []; print("netobserv-investigate" in s and "netobserv-evidence" in s)' 2>/dev/null \
    | grep -q True
}

openclaw_mcp_enabled() {
  local name="$1"
  "$KUBECTL" -n "$OPENCLAW_NS" get configmap openclaw-config \
    -o jsonpath='{.data.openclaw\.json}' 2>/dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin); m=(d.get('mcp',{}) or {}).get('servers',{}) or {}; print(m.get('$name',{}).get('enabled', False))" 2>/dev/null \
    | grep -q True
}

phase6_wired() {
  "$KUBECTL" -n "$OPENCLAW_NS" get secret/openclaw-aap-launcher >/dev/null 2>&1
}

cmd_plan() {
  cat <<EOF
Phase 7 — NetObserv agent kit seed (~10–20 min)

Guide: $GF_ROOT/docs/PHASE-7-AGENT.md

Prereq: Phase 2 (openclaw + openshell). Phase 6 recommended (ansible-mcp + AAP launcher).

Deploys via seed-openclaw-netobserv-skills.sh:
  - netobserv-mcp + openshift-mcp (read-only cluster tools)
  - netobserv-heal-proxy + capture-proxy
  - Workspace skills (investigate / evidence / heal) + AGENTS.md
  - ansible-automation MCP when Phase 6 complete

Steps:
  1. Ensure openshell lab checkout (pinned)
       $KIT/clone-openshell-lab.sh
  2. Full agent seed + OpenClaw recycle
       ENABLE_ANSIBLE_MCP=1 $KIT/seed-openclaw-netobserv-skills.sh

Or: ./scripts/phase7-agent.sh deploy

Verify:
  ./scripts/phase7-agent.sh check
  $KIT/netobserv-e2e-openclaw-test.sh status    # MCP doctor
  Control UI: /new → first message → openclaw-agent-* sandbox Running
EOF
}

cmd_check() {
  local fails=0
  step "Phase 7 readiness — $($KUBECTL whoami 2>/dev/null || echo '?')"

  gate "Phase 2 — openclaw deployment" "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw || fails=$((fails + 1))
  gate "netobserv-mcp" "$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-mcp || fails=$((fails + 1))
  gate "netobserv-heal-proxy" "$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-heal-proxy || fails=$((fails + 1))
  gate "netobserv-capture-proxy" "$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-capture-proxy || fails=$((fails + 1))
  gate "netobserv-sandbox-flatten" "$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-sandbox-flatten || fails=$((fails + 1))

  if "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openshift-mcp >/dev/null 2>&1; then
    ok "openshift-mcp deployment"
  else
    warn "openshift-mcp missing — seed may have skipped (set ENABLE_OPENSHIFT_MCP=1)"
  fi

  if phase6_wired; then
    gate "ansible-mcp (Phase 6)" "$KUBECTL" -n "$OPENCLAW_NS" get deploy/ansible-mcp || fails=$((fails + 1))
    if openclaw_mcp_enabled ansible-automation; then
      ok "OpenClaw MCP ansible-automation enabled"
    else
      fail "ansible-automation MCP not enabled in openclaw-config"
      fails=$((fails + 1))
    fi
  else
    warn "Phase 6 not detected — ansible-mcp checks skipped"
  fi

  if openclaw_skills_seeded; then
    ok "NetObserv skills allowlist in openclaw-config"
  else
    fail "Skills not seeded — run ./scripts/phase7-agent.sh deploy"
    fails=$((fails + 1))
  fi

  if openclaw_mcp_enabled netobserv-openshift; then
    ok "OpenClaw MCP netobserv-openshift enabled"
  else
    fail "netobserv MCP not enabled in openclaw-config"
    fails=$((fails + 1))
  fi

  if "$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy/netobserv-mcp --timeout=30s >/dev/null 2>&1; then
    ok "netobserv-mcp rollout healthy"
  else
    fail "netobserv-mcp not ready"
    fails=$((fails + 1))
  fi

  printf '\n'
  if [[ "$fails" -gt 0 ]]; then
    warn "$fails check(s) failed — see $GF_ROOT/docs/PHASE-7-AGENT.md#troubleshooting"
    return 1
  fi
  ok "Phase 7 checks passed"
  printf '  MCP doctor: %s/netobserv-e2e-openclaw-test.sh status\n' "$KIT"
  printf '  UI proof: /new → first message → %s/openshell-sandbox-proof.sh wait\n' "$KIT"
  return 0
}

cmd_status() {
  step "Agent kit / MCP status"
  "$KIT/netobserv-e2e-openclaw-test.sh" status 2>/dev/null || true
}

cmd_deploy() {
  step "Phase 7 automated deploy"
  site_config_ensure openshell 2>/dev/null || true

  "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw >/dev/null 2>&1 \
    || { fail "Phase 2 required"; exit 1; }

  if ! phase6_wired; then
    warn "Phase 6 not detected — ansible-automation MCP will be skipped unless you wire AAP first"
  fi

  if [[ ! -f "$LAB_CFG" ]]; then
    step "OpenShell lab missing — clone pinned checkout"
    "$KIT/clone-openshell-lab.sh" install \
      || { fail "clone-openshell-lab.sh failed"; exit 1; }
  else
    "$KIT/clone-openshell-lab.sh" status 2>/dev/null || true
  fi

  local ansible_mcp=0
  phase6_wired && ansible_mcp=1

  step "Seed NetObserv skills + MCP stack"
  warn "Recycles OpenClaw + MCP deployments (~3–5 min)"
  LAB_CFG="$LAB_CFG" OPENCLAW_NS="$OPENCLAW_NS" OPENSHELL_NS="$OPENSHELL_NS" \
    ENABLE_ANSIBLE_MCP="$ansible_mcp" \
    ENABLE_RHOAI_PLATFORM="${ENABLE_RHOAI_PLATFORM:-1}" \
    "$KIT/seed-openclaw-netobserv-skills.sh" \
    || { fail "seed-openclaw-netobserv-skills.sh failed"; exit 1; }

  ok "Deploy complete — run: ./scripts/phase7-agent.sh check"
  cat <<EOF

Control UI next:
  1. /new
  2. Send first message (sandbox spawns on first turn, not on /new alone)
  3. Optional: $KIT/openshell-sandbox-proof.sh wait
EOF
}

cmd_verify() {
  cmd_check || true
  step "MCP status (demo kit)"
  "$KIT/netobserv-e2e-openclaw-test.sh" status
}

case "${1:-plan}" in
  plan|help|-h|--help) cmd_plan ;;
  deploy|install|seed) cmd_deploy ;;
  check|verify-gates) cmd_check ;;
  status) cmd_status ;;
  verify|test) cmd_verify ;;
  *)
    echo "usage: $0 {plan|deploy|check|status|verify}" >&2
    exit 1
    ;;
esac
