#!/usr/bin/env bash
# Phase 0: install Multicloud GitOps hub + demo-5 observe plane.
# Requires: oc logged in as cluster-admin; podman (for pattern.sh); network to pull images.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PATTERN_DIR="${ROOT}/ValidatedPattern"
DEMO5="${ROOT}/demo/demo-5-distributed-agentic-workloads"
HUB_KUSTOMIZE="${DEMO5}/manifests/hub"

log() { printf '[install-hub-pattern] %s\n' "$*"; }
die() { printf '[install-hub-pattern] ERROR: %s\n' "$*" >&2; exit 1; }

command -v oc >/dev/null || die "oc not found"
oc whoami >/dev/null 2>&1 || die "oc not logged in (cluster-admin required)"

log "cluster: $(oc whoami -c 2>/dev/null || oc config current-context) user=$(oc whoami)"

# 1) Multicloud GitOps pattern (ArgoCD + RHACM + Vault)
if [[ "${SKIP_PATTERN:-0}" != "1" ]]; then
  [[ -x "${PATTERN_DIR}/pattern.sh" ]] || die "missing ${PATTERN_DIR}/pattern.sh"
  log "Installing Validated Pattern from ${PATTERN_DIR} (make install)…"
  log "Ensure ~/values-secret-multicloud-gitops.yaml (or pattern secret file) exists per pattern docs — never commit it."
  (
    cd "${PATTERN_DIR}"
    ./pattern.sh make install
  )
else
  log "SKIP_PATTERN=1 — skipping pattern.sh make install"
fi

# 2) Break-glass / immediate hub observe plane (also synced via Argo when Application is healthy)
log "Applying demo-5 hub kustomize (${HUB_KUSTOMIZE})…"
oc apply -k "${HUB_KUSTOMIZE}"

log "Waiting for ClickHouse and hub OTEL collector…"
oc -n demo5-observability rollout status deployment/clickhouse --timeout=300s
oc -n demo5-observability rollout status deployment/otel-collector --timeout=300s

ROUTE_HOST="$(oc -n demo5-observability get route otel-intake -o jsonpath='{.spec.host}' 2>/dev/null || true)"
if [[ -z "${ROUTE_HOST}" ]]; then
  die "otel-intake Route not found — is this an OpenShift cluster with Route CRD?"
fi

log "Hub OTLP Route: https://${ROUTE_HOST}"
log "Record this host in ${DEMO5}/config/sites.yaml (hub_otlp_route)."
log "Phase 0 hub observe plane applied. Next: Phase 1 site VM (MicroShift) + point site collector HUB_OTLP_ENDPOINT at the Route."
