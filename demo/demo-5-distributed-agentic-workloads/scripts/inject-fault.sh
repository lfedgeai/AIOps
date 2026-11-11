#!/usr/bin/env bash
# Inject a documented fault; pause Argo sync so GitOps does not self-heal.
set -euo pipefail

DEMO5="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAULT_ID="${1:-}"
SITE_ID="${SITE_ID:-site-a}"
ARGO_APP="${ARGO_APP:-demo5-site-${SITE_ID}}"

die() { printf '[inject-fault] ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '[inject-fault] %s\n' "$*"; }

[[ -n "${FAULT_ID}" ]] || die "usage: inject-fault.sh <fault_id>  (see config/fault_ground_truth.yaml)"

command -v oc >/dev/null || die "oc not found"
oc whoami >/dev/null 2>&1 || die "oc not logged in"

log "Pausing Argo sync for ${ARGO_APP} (ignore errors if app missing)…"
oc -n openshift-gitops patch application "${ARGO_APP}" --type merge \
  -p '{"spec":{"syncPolicy":null}}' 2>/dev/null || log "WARN: could not pause ${ARGO_APP}"

case "${FAULT_ID}" in
  todo-scale-zero)
    log "Scaling todo to 0 in todo-demo…"
    oc -n todo-demo scale deployment/todo --replicas=0
    ;;
  *)
    die "unknown fault_id=${FAULT_ID} — add handler or see fault_ground_truth.yaml"
    ;;
esac

log "Fault ${FAULT_ID} injected. Expect signature in ClickHouse per fault_ground_truth.yaml."
log "Remember to re-enable Argo sync after the demo."
