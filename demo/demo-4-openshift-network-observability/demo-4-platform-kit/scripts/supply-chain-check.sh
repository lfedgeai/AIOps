#!/usr/bin/env bash
# Verify kit supply-chain pins (read-only).
#
# Usage:
#   ./scripts/supply-chain-check.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=supply-chain-pins.env
source "$ROOT/scripts/supply-chain-pins.env"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }

fail=0
record() {
  local status="$1" label="$2" detail="$2"
  shift 2
  detail="$*"
  if [[ "$status" == pass ]]; then
    ok "$label — $detail"
  else
    warn "$label — $detail"
    fail=1
  fi
}

step "Supply chain pins (see docs/SUPPLY-CHAIN-PINS.md)"

# Kit env file self-check
if grep -q "^LEMONADE_COMMIT=.*${LEMONADE_COMMIT}" "$ROOT/scripts/supply-chain-pins.env" 2>/dev/null; then
  record pass "pins.env" "LEMONADE_COMMIT=${LEMONADE_COMMIT:0:12}"
else
  record fail "pins.env" "LEMONADE_COMMIT missing or inconsistent"
fi

# OpenShell lab checkout
LAB_DIR="${OPENCLAW_LAB_DIR:-$HOME/labs/openshell-on-openshift-lab}"
if [[ -d "$LAB_DIR/.git" ]]; then
  got="$(git -C "$LAB_DIR" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$got" == "$OPENCLAW_LAB_COMMIT" ]]; then
    record pass "openshell lab" "${got:0:12}"
  else
    record fail "openshell lab" "checkout ${got:0:12} ≠ pin ${OPENCLAW_LAB_COMMIT:0:12} — run ./scripts/clone-openshell-lab.sh"
  fi
else
  record fail "openshell lab" "not cloned — run ./scripts/clone-openshell-lab.sh"
fi

# Lemonade: if helm release exists, note pin (cannot read git SHA from cluster)
KUBECTL="$(command -v oc || command -v kubectl || true)"
if [[ -n "$KUBECTL" ]] && "$KUBECTL" get ns netobserv-guardrails >/dev/null 2>&1; then
  if helm list -n netobserv-guardrails 2>/dev/null | grep -q netobserv-trustyai-guardrails; then
    record pass "trustyai helm" "release netobserv-trustyai-guardrails (install pin ${LEMONADE_COMMIT:0:12})"
  else
    record fail "trustyai helm" "namespace exists but release missing — install-trustyai-guardrails.sh"
  fi
else
  warn "trustyai — cluster not checked or netobserv-guardrails absent"
fi

# install script defaults
if grep -q "supply-chain-pins.env" "$ROOT/scripts/install-trustyai-guardrails.sh" 2>/dev/null; then
  record pass "install-trustyai" "sources supply-chain-pins.env"
else
  record fail "install-trustyai" "not wired to supply-chain-pins.env"
fi

if grep -q "supply-chain-pins.env" "$ROOT/scripts/netobserv-krkn-fault.sh" 2>/dev/null; then
  record pass "krkn-fault" "KRKN_IMAGE=${KRKN_IMAGE##*/}"
else
  record fail "krkn-fault" "not wired to supply-chain-pins.env"
fi

if [[ -x "$ROOT/scripts/verify-image-pulls.sh" ]]; then
  if OPENCLAW_GATEWAY_IMAGE="${OPENCLAW_GATEWAY_IMAGE}" \
     OPENCLAW_SANDBOX_IMAGE="${OPENCLAW_SANDBOX_IMAGE}" \
     NETOBSERV_TODO_IMAGE="${NETOBSERV_TODO_IMAGE}" \
     "$ROOT/scripts/verify-image-pulls.sh" --phase2 >/dev/null 2>&1; then
    record pass "image pulls (phase2)" "${OPENCLAW_GATEWAY_IMAGE##*/}"
  else
    record fail "image pulls (phase2)" "${OPENCLAW_GATEWAY_IMAGE} — run ./scripts/verify-image-pulls.sh --phase2"
  fi
else
  warn "verify-image-pulls.sh missing"
fi

if command -v oc-netobserv >/dev/null 2>&1 || [[ -x /usr/local/bin/oc-netobserv ]]; then
  oc_dest="${OC_NETOBSERV_DEST:-/usr/local/bin/oc-netobserv}"
  [[ -x "$oc_dest" ]] || oc_dest="$(command -v oc-netobserv)"
  oc_ver="$("$oc_dest" version 2>/dev/null || true)"
  if [[ "$oc_ver" == *"${NETOBSERV_CLI_VERSION}"* ]]; then
    record pass "oc-netobserv" "${NETOBSERV_CLI_VERSION}"
  else
    record fail "oc-netobserv" "drift: ${oc_ver:-missing} — run ./scripts/install-oc-netobserv-cli.sh"
  fi
else
  warn "oc-netobserv — not on PATH (optional for bastion debugging)"
fi

if [[ -n "$KUBECTL" ]] && "$KUBECTL" get csv -A -o name 2>/dev/null | grep -q "clusterserviceversion/${NETOBSERV_OPERATOR_CSV}"; then
  record pass "netobserv operator" "${NETOBSERV_OPERATOR_CSV}"
elif [[ -n "$KUBECTL" ]] && "$KUBECTL" get csv -A 2>/dev/null | grep -q "${NETOBSERV_OPERATOR_CSV}"; then
  record pass "netobserv operator" "${NETOBSERV_OPERATOR_CSV}"
else
  warn "netobserv operator — ${NETOBSERV_OPERATOR_CSV} not found (optional if NetObserv not installed)"
fi

if (( fail > 0 )); then
  warn "supply-chain-check: ${fail} issue(s)"
  exit 1
fi
ok "supply-chain-check: all pins OK"
exit 0
