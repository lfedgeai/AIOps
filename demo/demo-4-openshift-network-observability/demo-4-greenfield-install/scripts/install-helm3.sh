#!/usr/bin/env bash
# Install Helm 3 on the bastion (official get-helm-3 script).
#
# Installs to ~/.local/bin by default (no sudo) — typical for lab-user bastions.
#
# Usage:
#   ./scripts/install-helm3.sh           # install if missing
#   ./scripts/install-helm3.sh install   # same
#   ./scripts/install-helm3.sh check     # exit 1 if helm 3 not found
#
# Env: HELM_INSTALL_DIR (default ~/.local/bin), USE_SUDO=false
set -euo pipefail

CMD="${1:-install}"
HELM_DIR="${HELM_INSTALL_DIR:-$HOME/.local/bin}"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

ensure_helm_path() {
  export PATH="$HELM_DIR:$PATH"
}

helm_version_ok() {
  ensure_helm_path
  command -v helm >/dev/null 2>&1 || return 1
  helm version --short 2>/dev/null | grep -qE 'v3\.' || helm version 2>/dev/null | grep -qE 'Version:"v3'
}

cmd_check() {
  if helm_version_ok; then
    ok "helm $(helm version --short 2>/dev/null | head -1)"
    return 0
  fi
  die "helm 3 not found — run: $0 install"
}

cmd_install() {
  ensure_helm_path
  if helm_version_ok; then
    ok "helm already installed ($(helm version --short 2>/dev/null | head -1))"
    return 0
  fi

  command -v curl >/dev/null 2>&1 || die "curl required to install helm"
  mkdir -p "$HELM_DIR"

  step "Installing Helm 3 to $HELM_DIR (no sudo)"
  export HELM_INSTALL_DIR="$HELM_DIR"
  export USE_SUDO=false
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

  helm_version_ok || die "helm install finished but helm 3 not in PATH ($HELM_DIR)"
  ok "helm $(helm version --short 2>/dev/null | head -1)"
  printf '\n  Add to your shell profile if needed:\n'
  printf '    export PATH="%s:$PATH"\n' "$HELM_DIR"
}

case "$CMD" in
  install|"") cmd_install ;;
  check|verify) cmd_check ;;
  *)
    echo "usage: $0 [install|check]" >&2
    exit 1
    ;;
esac
