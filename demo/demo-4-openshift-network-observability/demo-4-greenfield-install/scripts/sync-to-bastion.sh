#!/usr/bin/env bash
# Sync greenfield folder + demo kit to bastion (new site replication).
#
# Usage:
#   ./scripts/sync-to-bastion.sh              # rsync both trees
#   ./scripts/sync-to-bastion.sh greenfield   # greenfield folder only
#   ./scripts/sync-to-bastion.sh kit          # demo kit only
#
# Env: see config/env.example (BASTION_HOST, BASTION_*_PATH)
set -euo pipefail

GF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-all}"

[[ -f "$GF_ROOT/config/env.local" ]] && source "$GF_ROOT/config/env.local"

BASTION_HOST="${BASTION_HOST:-bastion.example.com}"
BASTION_USER="${BASTION_USER:-lab-user}"
BASTION_GREENFIELD_PATH="${BASTION_GREENFIELD_PATH:-AIOps/demo/demo-4-openshift-network-observability/demo-4-greenfield-install}"
BASTION_KIT_PATH="${BASTION_KIT_PATH:-AIOps/demo/demo-4-openshift-network-observability/demo-4-platform-kit}"

export GF_ROOT
# shellcheck source=scripts/resolve-demo-kit.sh
source "$GF_ROOT/scripts/resolve-demo-kit.sh"
resolve_demo_kit || true
export DEMO_KIT_ROOT

REMOTE="${BASTION_USER}@${BASTION_HOST}"

c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'; c_reset=$'\033[0m'
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }

ssh_wrap() {
  if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
    sshpass -e ssh -o StrictHostKeyChecking=accept-new "$@"
  else
    ssh -o StrictHostKeyChecking=accept-new "$@"
  fi
}

rsync_ssh() {
  if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
    echo "sshpass -e ssh -o StrictHostKeyChecking=accept-new"
  else
    echo "ssh -o StrictHostKeyChecking=accept-new"
  fi
}

# Remote path under bastion $HOME (never expand local $HOME).
remote_path() {
  local rel="${1#/}"
  printf '~/%s' "$rel"
}

rsync_one() {
  local src="$1" rel="$2"
  local dest
  dest="$(remote_path "$rel")"
  step "Rsync $src → ${REMOTE}:${dest}"
  ssh_wrap "$REMOTE" "mkdir -p ${dest}"
  rsync -avz --delete \
    --exclude '.git' --exclude '.DS_Store' --exclude 'config/env.local' \
    --exclude 'config/site-secrets.local.yaml' --exclude 'config/site-secrets.local.json' \
    -e "$(rsync_ssh)" \
    "$src/" "${REMOTE}:${dest}/"
  ssh_wrap "$REMOTE" "chmod +x ${dest}/scripts/*.sh ${dest}/scripts/site_config.py 2>/dev/null || true"
  ok "Synced ${dest}"
}

case "$CMD" in
  all|"")
    rsync_one "$GF_ROOT" "$BASTION_GREENFIELD_PATH"
    [[ -n "$DEMO_KIT_ROOT" && -d "$DEMO_KIT_ROOT" ]] || {
      echo "error: platform kit not found — set DEMO_KIT_ROOT or place demo-4-platform-kit as a sibling folder" >&2
      exit 1
    }
    rsync_one "$DEMO_KIT_ROOT" "$BASTION_KIT_PATH"
  ;;
  greenfield) rsync_one "$GF_ROOT" "$BASTION_GREENFIELD_PATH" ;;
  kit)
    [[ -n "$DEMO_KIT_ROOT" ]] || exit 1
    rsync_one "$DEMO_KIT_ROOT" "$BASTION_KIT_PATH"
  ;;
  *)
    echo "usage: $0 [all|greenfield|kit]" >&2
    exit 1
  ;;
esac
