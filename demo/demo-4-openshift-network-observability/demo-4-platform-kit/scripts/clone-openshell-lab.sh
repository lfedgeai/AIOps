#!/usr/bin/env bash
# Clone or update openshell-on-openshift-lab at the kit-pinned commit.
#
# Usage:
#   ./scripts/clone-openshell-lab.sh              # clone/update to OPENCLAW_LAB_COMMIT
#   ./scripts/clone-openshell-lab.sh status       # show pin vs checkout
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=supply-chain-pins.env
source "$ROOT/scripts/supply-chain-pins.env"

CMD="${1:-install}"
LAB_DIR="${OPENCLAW_LAB_DIR:-$HOME/labs/openshell-on-openshift-lab}"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

checkout_pin() {
  git -C "$LAB_DIR" fetch --depth 1 origin "$OPENCLAW_LAB_COMMIT" 2>/dev/null \
    || git -C "$LAB_DIR" fetch --depth 1 origin
  git -C "$LAB_DIR" checkout --detach "$OPENCLAW_LAB_COMMIT"
}

cmd_install() {
  step "OpenShell lab @ ${OPENCLAW_LAB_COMMIT:0:12} → ${LAB_DIR}"
  mkdir -p "$(dirname "$LAB_DIR")"
  if [[ -d "$LAB_DIR/.git" ]]; then
    checkout_pin
  else
    rm -rf "$LAB_DIR"
    git clone --filter=blob:none --no-checkout "$OPENCLAW_LAB_REPO" "$LAB_DIR"
    checkout_pin
  fi
  local got
  got="$(git -C "$LAB_DIR" rev-parse HEAD)"
  [[ "$got" == "$OPENCLAW_LAB_COMMIT" ]] || die "checkout mismatch: got $got expected $OPENCLAW_LAB_COMMIT"
  ok "openshell-on-openshift-lab @ ${got:0:12} ($(git -C "$LAB_DIR" log -1 --format='%s'))"
  cat <<EOF

Next (if fresh bastion):
  export OPENSHELL_VERSION=${OPENSHELL_CHART_VERSION}
  export AGENT_SANDBOX_RELEASE=${AGENT_SANDBOX_RELEASE}
  # See README "Install tooling" for helm + OpenClaw deploy
EOF
}

cmd_status() {
  if [[ ! -d "$LAB_DIR/.git" ]]; then
    warn "Lab not cloned — run: $0"
    return 1
  fi
  local got msg
  got="$(git -C "$LAB_DIR" rev-parse HEAD)"
  msg="$(git -C "$LAB_DIR" log -1 --format='%s')"
  if [[ "$got" == "$OPENCLAW_LAB_COMMIT" ]]; then
    ok "openshell lab pinned ${got:0:12} — $msg"
  else
    warn "openshell lab drift: checkout ${got:0:12} ≠ pin ${OPENCLAW_LAB_COMMIT:0:12}"
    warn "  run: $0"
    return 1
  fi
}

case "$CMD" in
  install|clone|update) cmd_install ;;
  status) cmd_status ;;
  *)
    echo "usage: $0 [install|status]" >&2
    exit 1
    ;;
esac
