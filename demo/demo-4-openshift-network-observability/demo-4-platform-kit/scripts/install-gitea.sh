#!/usr/bin/env bash
# Deploy Gitea on OpenShift for NetObserv/AAP playbook source control.
# Based on https://github.com/kwkoo/gitea-openshift
#
# Usage:
#   ./scripts/install-gitea.sh status
#   ./scripts/install-gitea.sh install
#   ./scripts/install-gitea.sh uninstall
#
# Env:
#   GITEA_NS=gitea
#   GITEA_USE_FSGROUP=0|1     use root-fsgroup SCC variant if pod fails on PVC (default 0)
#   GITEA_PVC_SIZE=10Gi
#   GITEA_IMAGE=quay.io/your-org/gitea-openshift
#   GITEA_TAG=gitea-openshift-v1
#   GITEA_ADMIN_USER=netobserv
#   GITEA_ADMIN_PASSWORD=    generated and stored in secret if unset
#   GITEA_ADMIN_EMAIL=netobserv@example.com
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFESTS="$ROOT/manifests/gitea"
# shellcheck source=supply-chain-pins.env
source "$ROOT/scripts/supply-chain-pins.env"
KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-status}"

GITEA_NS="${GITEA_NS:-gitea}"
GITEA_USE_FSGROUP="${GITEA_USE_FSGROUP:-0}"
GITEA_PVC_SIZE="${GITEA_PVC_SIZE:-10Gi}"
GITEA_IMAGE="${GITEA_IMAGE:-quay.io/${QUAY_ORG}/gitea-openshift}"
GITEA_TAG="${GITEA_TAG:-gitea-openshift-v1}"
GITEA_ADMIN_USER="${GITEA_ADMIN_USER:-netobserv}"
GITEA_ADMIN_EMAIL="${GITEA_ADMIN_EMAIL:-netobserv@example.com}"
GITEA_SECRET="${GITEA_SECRET:-gitea-admin-credentials}"
WAIT_GITEA_SEC="${WAIT_GITEA_SEC:-600}"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -n "$KUBECTL" ]] || die "oc/kubectl required"

ingress_suffix() {
  local suffix host
  "$KUBECTL" create namespace "$GITEA_NS" --dry-run=client -o yaml | "$KUBECTL" apply -f - >/dev/null
  if ! "$KUBECTL" -n "$GITEA_NS" create route edge dummy --service=dummy --port=8080 \
      --dry-run=client -o yaml 2>/dev/null | "$KUBECTL" apply -f - >/dev/null 2>&1; then
    "$KUBECTL" -n "$GITEA_NS" create route edge dummy --service=dummy --port=8080 2>/dev/null || true
  fi
  host="$("$KUBECTL" -n "$GITEA_NS" get route dummy -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  "$KUBECTL" -n "$GITEA_NS" delete route dummy --ignore-not-found >/dev/null 2>&1 || true
  [[ -n "$host" ]] || die "could not discover ingress suffix (route dummy)"
  suffix="${host#*.}"
  printf '%s' "$suffix"
}

gitea_route_host() {
  "$KUBECTL" -n "$GITEA_NS" get route gitea -o jsonpath='{.spec.host}' 2>/dev/null || true
}

gitea_cluster_url() {
  printf 'http://gitea.%s.svc.cluster.local:3000' "$GITEA_NS"
}

ensure_admin_password() {
  if [[ -n "${GITEA_ADMIN_PASSWORD:-}" ]]; then
    return 0
  fi
  if "$KUBECTL" -n "$GITEA_NS" get secret "$GITEA_SECRET" >/dev/null 2>&1; then
    GITEA_ADMIN_PASSWORD="$("$KUBECTL" -n "$GITEA_NS" get secret "$GITEA_SECRET" \
      -o jsonpath='{.data.password}' | base64 -d)"
    return 0
  fi
  GITEA_ADMIN_PASSWORD="$(openssl rand -base64 18 | tr -d '/+=' | head -c 20)"
  "$KUBECTL" -n "$GITEA_NS" create secret generic "$GITEA_SECRET" \
    --from-literal=username="$GITEA_ADMIN_USER" \
    --from-literal=password="$GITEA_ADMIN_PASSWORD" \
    --from-literal=email="$GITEA_ADMIN_EMAIL" \
    --dry-run=client -o yaml | "$KUBECTL" apply -f -
}

wait_gitea_ready() {
  step "Wait Gitea StatefulSet (timeout ${WAIT_GITEA_SEC}s)"
  "$KUBECTL" -n "$GITEA_NS" wait --for=condition=ready pod -l app=gitea --timeout="${WAIT_GITEA_SEC}s"
}

wait_gitea_api() {
  local host="$1"
  local end=$((SECONDS + 120))
  step "Wait Gitea API on https://${host}"
  while (( SECONDS < end )); do
    if curl -sk -o /dev/null -w '' "https://${host}/api/v1/version" 2>/dev/null; then
      ok "Gitea API ready"
      return 0
    fi
    sleep 5
  done
  warn "Gitea API not responding on route yet"
  return 1
}

create_admin_user() {
  ensure_admin_password
  step "Create Gitea admin user ${GITEA_ADMIN_USER} (idempotent)"
  if "$KUBECTL" -n "$GITEA_NS" exec statefulset/gitea -- \
      gitea admin user list 2>/dev/null | grep -q "^${GITEA_ADMIN_USER}[[:space:]]"; then
    ok "user ${GITEA_ADMIN_USER} already exists"
  else
    "$KUBECTL" -n "$GITEA_NS" exec statefulset/gitea -- \
      gitea admin user create \
        --admin \
        --username "$GITEA_ADMIN_USER" \
        --password "$GITEA_ADMIN_PASSWORD" \
        --email "$GITEA_ADMIN_EMAIL"
    ok "created user ${GITEA_ADMIN_USER}"
  fi
  "$KUBECTL" -n "$GITEA_NS" exec statefulset/gitea -- \
    gitea admin user must-change-password --unset "$GITEA_ADMIN_USER" 2>/dev/null || true
}

cmd_status() {
  step "Gitea namespace"
  "$KUBECTL" get ns "$GITEA_NS" 2>/dev/null || warn "namespace $GITEA_NS missing"
  "$KUBECTL" -n "$GITEA_NS" get statefulset,svc,route,pvc 2>/dev/null || true
  local host
  host="$(gitea_route_host)"
  if [[ -n "$host" ]]; then
    ok "Route: https://${host}"
    ok "In-cluster: $(gitea_cluster_url)"
  fi
  if "$KUBECTL" -n "$GITEA_NS" get secret "$GITEA_SECRET" >/dev/null 2>&1; then
    ok "Credentials secret: ${GITEA_NS}/${GITEA_SECRET}"
  fi
}

cmd_install() {
  step "Namespace ${GITEA_NS}"
  "$KUBECTL" create namespace "$GITEA_NS" --dry-run=client -o yaml | "$KUBECTL" apply -f -
  "$KUBECTL" label namespace "$GITEA_NS" app.kubernetes.io/part-of=netobserv-aap --overwrite

  if "$KUBECTL" -n "$GITEA_NS" get statefulset gitea >/dev/null 2>&1; then
    ok "Gitea StatefulSet already exists — skipping oc new-app"
    wait_gitea_ready
    ensure_admin_password
    create_admin_user
    local host
    host="$(gitea_route_host)"
    [[ -n "$host" ]] && wait_gitea_api "$host" || true
    cmd_status
    return 0
  fi
  local suffix domain root_url template
  suffix="$(ingress_suffix)"
  domain="gitea-${GITEA_NS}.${suffix}"
  root_url="https://${domain}"

  if [[ "$GITEA_USE_FSGROUP" == "1" ]]; then
    step "Apply root-fsgroup SCC (kwkoo/gitea-openshift)"
    "$KUBECTL" apply -f "$MANIFESTS/root-fsgroup.yaml"
    template="$MANIFESTS/gitea-fsgroup-template.yaml"
    [[ -f "$template" ]] || die "missing $template — copy from kwkoo/gitea-openshift"
    oc new-app -n "$GITEA_NS" -f "$template" \
      -p PROJECT="$GITEA_NS" \
      -p DOMAIN="$domain" \
      -p ROOT_URL="$root_url" \
      -p IMAGE="$GITEA_IMAGE" \
      -p TAG="$GITEA_TAG" \
      -p PVC_SIZE="$GITEA_PVC_SIZE"
  else
    step "Deploy Gitea from kwkoo template (sqlite3)"
    oc new-app -n "$GITEA_NS" -f "$MANIFESTS/gitea-template.yaml" \
      -p DOMAIN="$domain" \
      -p ROOT_URL="$root_url" \
      -p IMAGE="$GITEA_IMAGE" \
      -p TAG="$GITEA_TAG" \
      -p PVC_SIZE="$GITEA_PVC_SIZE" \
      -p LOG_LEVEL=WARN
  fi

  wait_gitea_ready
  ensure_admin_password
  create_admin_user
  wait_gitea_api "$domain" || true

  cmd_status
  cat <<EOF

Gitea ready for NetObserv playbooks.
  Seed repo:  ./scripts/seed-gitea-netobserv-playbooks.sh
  AAP wire:   ./scripts/wire-openclaw-aap.sh all

Admin password (if needed): oc get secret ${GITEA_SECRET} -n ${GITEA_NS} -o jsonpath='{.data.password}' | base64 -d; echo
EOF
}

cmd_uninstall() {
  warn "Removing Gitea from ${GITEA_NS}"
  "$KUBECTL" delete all,pvc,sa,route,statefulset,svc -l app=gitea -n "$GITEA_NS" --ignore-not-found
  "$KUBECTL" delete secret "$GITEA_SECRET" -n "$GITEA_NS" --ignore-not-found
  ok "Gitea resources deleted (namespace kept)"
}

case "$CMD" in
  status) cmd_status ;;
  install) cmd_install ;;
  uninstall) cmd_uninstall ;;
  *) die "usage: $0 {status|install|uninstall}" ;;
esac
