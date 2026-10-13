#!/usr/bin/env bash
# Verify container images are pullable before install (fail fast on ImagePullBackOff loops).
#
# Usage:
#   ./scripts/verify-image-pulls.sh              # P0: OpenClaw + todo (greenfield phases 1–2)
#   ./scripts/verify-image-pulls.sh --phase2     # OpenClaw only
#   ./scripts/verify-image-pulls.sh --phase1     # NetObserv todo only
#   ./scripts/verify-image-pulls.sh --extended   # + Kraken, MLflow, Gitea
#   ./scripts/verify-image-pulls.sh --gitea      # Gitea only (Phase 6)
#
# Env: supply-chain-pins.env (+ OPENCLAW_* overrides from greenfield config/env.local)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=supply-chain-pins.env
source "$ROOT/scripts/supply-chain-pins.env"

SCOPE="${1:---default}"
[[ "$SCOPE" == "--default" ]] && SCOPE=""

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_red=$'\033[1;31m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
fail() { printf '%s[fail]%s %s\n' "$c_red" "$c_reset" "$*" >&2; FAILS=$((FAILS + 1)); }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }

FAILS=0

want_phase1() {
  [[ -z "$SCOPE" || "$SCOPE" == "--phase1" ]]
}
want_phase2() {
  [[ -z "$SCOPE" || "$SCOPE" == "--phase2" ]]
}
want_extended() {
  [[ "$SCOPE" == "--extended" ]]
}
want_gitea() {
  [[ "$SCOPE" == "--gitea" ]]
}

inspect_image() {
  local ref="$1" label="$2"
  if command -v skopeo >/dev/null 2>&1; then
    if skopeo inspect --override-os linux "docker://${ref}" >/dev/null 2>&1; then
      ok "$label — $ref"
      return 0
    fi
  fi
  if command -v podman >/dev/null 2>&1; then
    if podman manifest inspect "$ref" >/dev/null 2>&1; then
      ok "$label — $ref"
      return 0
    fi
  fi
  fail "$label — cannot inspect $ref (install podman or skopeo; ensure logged in to registry)"
  return 1
}

warn_latest() {
  local ref="$1"
  if [[ "$ref" == *:latest ]]; then
    warn "Image uses :latest — consider pinning a digest/tag in supply-chain-pins.env"
  fi
}

step "Image pull verification (scope: ${SCOPE:-phases 1–2})"

if want_phase1; then
  inspect_image "$NETOBSERV_TODO_IMAGE" "NetObserv todo"
fi

if want_phase2; then
  inspect_image "$OPENCLAW_GATEWAY_IMAGE" "OpenClaw gateway"
  if [[ "$OPENCLAW_SANDBOX_IMAGE" != "$OPENCLAW_GATEWAY_IMAGE" ]]; then
    inspect_image "$OPENCLAW_SANDBOX_IMAGE" "OpenClaw sandbox"
  else
    ok "OpenClaw sandbox — same as gateway"
  fi
  # Guard against accidental ryan_nix / hummingbird drift after lab checkout
  case "$OPENCLAW_GATEWAY_IMAGE" in
    *ryan_nix/openclaw-openshift:hummingbird*|*ryan_nix/openclaw-openshift:2026.08*|*ryan_nix/openclaw-openshift:latest)
      fail "OpenClaw gateway still points at known-bad upstream tag — set OPENCLAW_GATEWAY_IMAGE to quay.io/${QUAY_ORG}/openclaw-openshift:openclaw-v2026.6.11"
      ;;
  esac
fi

if want_extended || want_gitea; then
  inspect_image "${GITEA_IMAGE}:${GITEA_TAG}" "Gitea"
  warn_latest "${GITEA_IMAGE}:${GITEA_TAG}"
fi

if want_extended; then
  inspect_image "${KRKN_IMAGE}" "Kraken fault"
  inspect_image "${KRKN_HOG_IMAGE}" "Kraken hog"
  inspect_image "${KRKN_TOOLS_IMAGE}" "Kraken tools"
  inspect_image "ghcr.io/mlflow/mlflow:v3.1.1" "MLflow standalone"
fi

printf '\n'
if (( FAILS > 0 )); then
  warn "verify-image-pulls: ${FAILS} image(s) not pullable"
  printf '  OpenClaw mirror: demo-4-greenfield-install/scripts/mirror-openclaw-image-to-quay.sh\n'
  printf '  Docs: demo-4-greenfield-install/docs/IMAGE-MIRRORS.md\n'
  exit 1
fi
ok "verify-image-pulls: all required images reachable"
