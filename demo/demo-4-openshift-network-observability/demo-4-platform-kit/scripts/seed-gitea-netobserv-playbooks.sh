#!/usr/bin/env bash
# Push NetObserv Ansible playbooks to in-cluster Gitea (AAP git source).
#
# Usage:
#   ./scripts/seed-gitea-netobserv-playbooks.sh
#   ./scripts/seed-gitea-netobserv-playbooks.sh status
#
# Prereq: ./scripts/install-gitea.sh install
#
# Env:
#   GITEA_NS=gitea
#   GITEA_REPO_NAME=netobserv-heal
#   GITEA_ADMIN_USER=netobserv
#   GITEA_ADMIN_PASSWORD=   (from gitea-admin-credentials secret if unset)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-seed}"

GITEA_NS="${GITEA_NS:-gitea}"
GITEA_REPO_NAME="${GITEA_REPO_NAME:-netobserv-heal}"
GITEA_ADMIN_USER="${GITEA_ADMIN_USER:-netobserv}"
GITEA_SECRET="${GITEA_SECRET:-gitea-admin-credentials}"
GITEA_BRANCH="${GITEA_BRANCH:-main}"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -n "$KUBECTL" ]] || die "oc/kubectl required"

route_host() {
  "$KUBECTL" -n "$GITEA_NS" get route gitea -o jsonpath='{.spec.host}' 2>/dev/null || true
}

cluster_git_url() {
  printf 'http://gitea.%s.svc.cluster.local:3000/%s/%s.git' \
    "$GITEA_NS" "$GITEA_ADMIN_USER" "$GITEA_REPO_NAME"
}

load_password() {
  if [[ -n "${GITEA_ADMIN_PASSWORD:-}" ]]; then
    return 0
  fi
  GITEA_ADMIN_PASSWORD="$("$KUBECTL" -n "$GITEA_NS" get secret "$GITEA_SECRET" \
    -o jsonpath='{.data.password}' 2>/dev/null | base64 -d || true)"
  [[ -n "$GITEA_ADMIN_PASSWORD" ]] || die "set GITEA_ADMIN_PASSWORD or run install-gitea.sh install"
}

cmd_status() {
  local host
  host="$(route_host)"
  [[ -n "$host" ]] && ok "Route: https://${host}/${GITEA_ADMIN_USER}/${GITEA_REPO_NAME}"
  ok "In-cluster git: $(cluster_git_url)"
}

cmd_seed() {
  load_password
  local host repo_url work
  host="$(route_host)"
  [[ -n "$host" ]] || die "Gitea route missing — run install-gitea.sh install"

  step "Ensure Gitea repo ${GITEA_ADMIN_USER}/${GITEA_REPO_NAME}"
  if ! curl -sk -u "${GITEA_ADMIN_USER}:${GITEA_ADMIN_PASSWORD}" \
      "https://${host}/api/v1/repos/${GITEA_ADMIN_USER}/${GITEA_REPO_NAME}" | grep -q '"name"'; then
    curl -sk -X POST \
      -u "${GITEA_ADMIN_USER}:${GITEA_ADMIN_PASSWORD}" \
      -H 'Content-Type: application/json' \
      -d "{\"name\":\"${GITEA_REPO_NAME}\",\"private\":false,\"auto_init\":false,\"default_branch\":\"${GITEA_BRANCH}\"}" \
      "https://${host}/api/v1/user/repos" >/dev/null
    ok "created repo ${GITEA_REPO_NAME}"
  else
    ok "repo ${GITEA_REPO_NAME} exists"
  fi

  work="$(mktemp -d)"
  trap '[[ -n "${work:-}" ]] && rm -rf "$work"' EXIT
  mkdir -p "$work/playbooks"
  cp "$ROOT/ansible/playbooks/"*.yml "$work/playbooks/"
  cp "$ROOT/ansible/requirements.yml" "$work/" 2>/dev/null || true
  cat >"$work/README.md" <<EOF
# NetObserv heal playbooks (AAP source)

Consumed by Ansible Automation Platform job templates:
- \`playbooks/netobserv-heal-db-path.yml\` — Scenario A
- \`playbooks/netobserv-restore-policy.yml\` — Scenario B

Do not edit on the controller — commit here and sync the AAP project.
EOF

  step "Git push to Gitea"
  git -C "$work" init -b "$GITEA_BRANCH"
  git -C "$work" config user.email "netobserv@gitea.local"
  git -C "$work" config user.name "NetObserv Demo Kit"
  git -C "$work" add .
  git -C "$work" commit -m "NetObserv heal playbooks from demo kit"
  repo_url="https://${GITEA_ADMIN_USER}:${GITEA_ADMIN_PASSWORD}@${host}/${GITEA_ADMIN_USER}/${GITEA_REPO_NAME}.git"
  git -C "$work" push --force "$repo_url" "$GITEA_BRANCH"

  ok "Playbooks pushed to ${GITEA_ADMIN_USER}/${GITEA_REPO_NAME}@${GITEA_BRANCH}"
  ok "AAP scm_url: $(cluster_git_url)"
  cmd_status
}

case "$CMD" in
  status) cmd_status ;;
  seed) cmd_seed ;;
  *) die "usage: $0 {status|seed}" ;;
esac
