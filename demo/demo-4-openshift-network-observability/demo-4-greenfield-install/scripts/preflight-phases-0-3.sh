#!/usr/bin/env bash
# Preflight for greenfield Phases 0–3 on a NEW cluster (run from bastion after oc login).
#
# Usage:
#   ./scripts/preflight-phases-0-3.sh           # full check
#   ./scripts/preflight-phases-0-3.sh --local   # skip cluster probes (laptop only)
set -euo pipefail

GF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GF_ROOT
LOCAL_ONLY=0
[[ "${1:-}" == "--local" ]] && LOCAL_ONLY=1

[[ -f "$GF_ROOT/config/env.local" ]] && source "$GF_ROOT/config/env.local"
# shellcheck source=scripts/site-config.sh
source "$GF_ROOT/scripts/site-config.sh"
# shellcheck source=scripts/resolve-demo-kit.sh
source "$GF_ROOT/scripts/resolve-demo-kit.sh"

resolve_kit() {
  resolve_demo_kit
}

KUBECTL="$(command -v oc || command -v kubectl || true)"

c_green=$'\033[1;32m'; c_red=$'\033[1;31m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
pass() { printf '  %s✓%s %s\n' "$c_green" "$c_reset" "$*"; }
fail() { printf '  %s✗%s %s\n' "$c_red" "$c_reset" "$*"; FAILS=$((FAILS + 1)); }
warn() { printf '  %s!%s %s\n' "$c_yellow" "$c_reset" "$*"; WARNS=$((WARNS + 1)); }
section() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }

FAILS=0
WARNS=0

section "Phase 0 — tooling"
for bin in oc jq curl aws git python3 podman; do
  command -v "$bin" >/dev/null 2>&1 && pass "$bin" || fail "$bin missing"
done
if command -v helm >/dev/null 2>&1 && helm version --short 2>/dev/null | grep -qE 'v3\.'; then
  pass "helm $(helm version --short 2>/dev/null | head -1)"
elif [[ "${INSTALL_MISSING_TOOLS:-1}" == "1" ]] && [[ -x "$GF_ROOT/scripts/install-helm3.sh" ]]; then
  warn "helm 3 missing — installing via install-helm3.sh"
  if "$GF_ROOT/scripts/install-helm3.sh" install; then
    pass "helm installed"
  else
    fail "helm install failed"
  fi
else
  fail "helm 3 missing — run: ./scripts/install-helm3.sh install"
fi
if python3 -c "import yaml" 2>/dev/null; then
  pass "python3 PyYAML (site-secrets .yaml)"
else
  warn "PyYAML missing — use site-secrets.local.json or: dnf install python3-pyyaml"
fi

section "Phase 0 — repo layout"
if resolve_kit; then
  pass "DEMO_KIT_ROOT=$DEMO_KIT_ROOT"
else
  fail "platform kit not found — set DEMO_KIT_ROOT or place demo-4-platform-kit as a sibling folder"
fi
[[ -x "$GF_ROOT/scripts/greenfield-install.sh" ]] && pass "greenfield-install.sh" || fail "greenfield-install.sh"
for s in site-config.sh phase2-openshell.sh phase3-rhoai.sh site_config.py; do
  [[ -f "$GF_ROOT/scripts/$s" ]] && pass "scripts/$s" || fail "missing scripts/$s"
done

section "Phase 0 — site secrets (Phases 1–3 fields)"
CFG="$(site_config_path)"
if [[ ! -f "$CFG" ]]; then
  fail "no site config at $CFG — run: ./scripts/greenfield-install.sh config init && config prompt"
else
  pass "site config: $CFG"
  chmod 600 "$CFG" 2>/dev/null || warn "chmod 600 $CFG recommended"
  for phase in netobserv openshell; do
    if site_config_validate "$phase" 2>/dev/null; then
      pass "config valid for phase $phase"
    else
      fail "config incomplete for phase $phase — run: ./scripts/greenfield-install.sh config prompt"
      site_config_validate "$phase" 2>&1 | sed 's/^/    /' || true
    fi
  done
  warn "slack tokens not required until Phase 8 (ok to leave empty for now)"
fi

if [[ "$LOCAL_ONLY" == "1" ]]; then
  section "Skipped (--local)"
  warn "Cluster probes skipped — re-run on bastion after oc login"
else
  section "Cluster — login & prerequisites"
  [[ -n "$KUBECTL" ]] || fail "oc/kubectl not in PATH"
  if WHO="$("$KUBECTL" whoami 2>/dev/null)"; then
    pass "logged in as $WHO"
    SERVER="$("$KUBECTL" whoami --show-server 2>/dev/null || true)"
    [[ -n "$SERVER" ]] && pass "api-server $SERVER"
  else
    fail "not logged in — run: oc login (see $GF_ROOT/docs/CLUSTER-LOGIN.md)"
  fi

  if [[ -n "$WHO" ]]; then
  if [[ "$("$KUBECTL" auth can-i '*' '*' --all-namespaces 2>/dev/null)" == "yes" ]]; then
    pass "cluster-admin"
  else
    fail "need cluster-admin"
  fi

  VER="$("$KUBECTL" version -o json 2>/dev/null | python3 -c "import json,sys; v=json.load(sys.stdin); print(v.get('serverVersion',{}).get('gitVersion','?'))" 2>/dev/null || echo "?")"
  pass "OpenShift/K8s server $VER"

  CNI="$("$KUBECTL" get network.operator cluster -o jsonpath='{.spec.defaultNetwork.type}' 2>/dev/null || true)"
  if [[ "$CNI" == "OVNKubernetes" ]]; then
    pass "CNI OVNKubernetes"
  else
    fail "CNI must be OVNKubernetes (got: ${CNI:-unknown})"
  fi

  SC="${STORAGE_CLASS:-gp3-csi}"
  if "$KUBECTL" get storageclass "$SC" >/dev/null 2>&1; then
    pass "StorageClass $SC"
  else
    warn "StorageClass $SC not found — set openshift.storage_class in site config or install-netobserv will prompt"
    "$KUBECTL" get storageclass -o name 2>/dev/null | head -5 | sed 's/^/    /' || true
  fi

  section "Cluster — phase detection (expect pending on new cluster)"
  if [[ -x "$GF_ROOT/scripts/greenfield-install.sh" ]]; then
    "$GF_ROOT/scripts/greenfield-install.sh" phases 2>/dev/null || true
  fi
  fi
fi

section "Phase 2 note"
warn "Phase 2 is automated: ./scripts/greenfield-install.sh openshell (or phase2-openshell.sh deploy)"
if resolve_kit 2>/dev/null && [[ -x "$DEMO_KIT_ROOT/scripts/verify-image-pulls.sh" ]]; then
  section "Image mirrors (Phases 1–2)"
  if "$DEMO_KIT_ROOT/scripts/verify-image-pulls.sh" >/dev/null 2>&1; then
    pass "P0 images pullable (todo + OpenClaw)"
  else
    warn "P0 images not all pullable — see $GF_ROOT/docs/IMAGE-MIRRORS.md"
    "$DEMO_KIT_ROOT/scripts/verify-image-pulls.sh" 2>&1 | sed 's/^/    /' || true
  fi
fi

section "Summary"
if [[ "$FAILS" -eq 0 ]]; then
  pass "Preflight PASSED ($WARNS warning(s)) — ready to start Phase 1 on this cluster"
  printf '\n  ./scripts/greenfield-install.sh netobserv\n'
  exit 0
fi
fail "Preflight FAILED — $FAILS blocker(s), $WARNS warning(s)"
printf '\n  Fix blockers above, then re-run: ./scripts/preflight-phases-0-3.sh\n'
exit 1
