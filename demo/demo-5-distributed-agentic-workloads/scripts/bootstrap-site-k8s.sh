#!/usr/bin/env bash
# Phase 1: install MicroShift (first slice) on a site VM and register with RHACM or Argo.
set -euo pipefail

DEMO5="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE_ID="${SITE_ID:-site-a}"
DISTRO="${DISTRO:-microshift}" # microshift | k3s (k3s = full PoC)

log() { printf '[bootstrap-site-k8s] %s\n' "$*"; }
die() { printf '[bootstrap-site-k8s] ERROR: %s\n' "$*" >&2; exit 1; }

[[ "${DISTRO}" == "microshift" || "${DISTRO}" == "k3s" ]] || die "DISTRO must be microshift or k3s"

HUB_OTLP="$(grep -E '^\s*hub_otlp_route:' "${DEMO5}/config/sites.yaml" 2>/dev/null | awk '{print $2}' | tr -d '"' || true)"
[[ -n "${HUB_OTLP}" && "${HUB_OTLP}" != "https://otel-intake.apps.REPLACE_ME" ]] \
  || log "WARN: set hub_otlp_route in config/sites.yaml to the real Route host from install-hub-pattern.sh"

log "site_id=${SITE_ID} distro=${DISTRO}"
log "1) On the site VM: install ${DISTRO}; ensure API up and *.apps hub Routes reachable."
log "2) Try RHACM import (preferred). On failure: create ArgoCD cluster secret + Application destination.name."
log "3) Always label/register site_id=${SITE_ID} (do not pretend ACM-managed if Argo-only)."
log "4) Deploy site collector overlay: oc apply -k ${DEMO5}/manifests/site-workloads/todo/overlays/${SITE_ID}"
log "   (sets SITE_ID and HUB_OTLP_ENDPOINT)."
log "5) Smoke: hub ClickHouse receives collector traffic (todo not required yet)."
exit 0
