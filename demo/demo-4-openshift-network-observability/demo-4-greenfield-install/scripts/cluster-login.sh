#!/usr/bin/env bash
# Verify (or document) OpenShift login on the bastion.
#
# Cluster credentials are NOT stored in site-secrets — use oc login.
#
# Usage:
#   ./scripts/cluster-login.sh check     # verify logged in + cluster-admin
#   ./scripts/cluster-login.sh help      # login instructions
set -euo pipefail

GF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-check}"

KUBECTL="$(command -v oc || command -v kubectl || true)"

c_green=$'\033[1;32m'; c_red=$'\033[1;31m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '  %s✓%s %s\n' "$c_green" "$c_reset" "$*"; }
fail() { printf '  %s✗%s %s\n' "$c_red" "$c_reset" "$*"; exit 1; }
warn() { printf '  %s!%s %s\n' "$c_yellow" "$c_reset" "$*"; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }

cmd_help() {
  cat <<EOF
OpenShift login is required before preflight and Phase 1.

Full guide: $GF_ROOT/docs/CLUSTER-LOGIN.md

Quick start:
  export OCP_API="https://api.<cluster-name>.<base-domain>:6443"
  oc login "\$OCP_API" -u kubeadmin -p '<password>'
  ./scripts/cluster-login.sh check

Site secrets (AWS/LLM) are separate:
  ./scripts/greenfield-install.sh config prompt
EOF
}

cmd_check() {
  [[ -n "$KUBECTL" ]] || fail "oc/kubectl not found"
  step "OpenShift session"
  if ! WHO="$("$KUBECTL" whoami 2>/dev/null)"; then
    fail "not logged in — see $GF_ROOT/docs/CLUSTER-LOGIN.md"
  fi
  ok "user: $WHO"
  SERVER="$("$KUBECTL" whoami --show-server 2>/dev/null || true)"
  [[ -n "$SERVER" ]] && ok "api: $SERVER" || warn "could not read api server URL"
  if [[ "$("$KUBECTL" auth can-i '*' '*' --all-namespaces 2>/dev/null)" == "yes" ]]; then
    ok "cluster-admin"
  else
    fail "not cluster-admin — log in as kubeadmin or equivalent"
  fi
  CNI="$("$KUBECTL" get network.operator cluster -o jsonpath='{.spec.defaultNetwork.type}' 2>/dev/null || true)"
  if [[ "$CNI" == "OVNKubernetes" ]]; then
    ok "CNI OVNKubernetes"
  else
    warn "CNI is '${CNI:-unknown}' — NetObserv requires OVNKubernetes"
  fi
  ok "ready for preflight / Phase 1"
}

case "$CMD" in
  check|verify) cmd_check ;;
  help|-h|--help) cmd_help ;;
  *)
    echo "usage: $0 {check|help}" >&2
    exit 1
    ;;
esac
