#!/usr/bin/env bash
# Create namespaces, relax Pod Security for the OTEL demo, and install harness RBAC.
set -euo pipefail

. "$(dirname "$0")/lib.sh"
oc_login

for ns in "$AIOPS_NAMESPACE" "$OTEL_DEMO_NAMESPACE" "$HARNESS_NAMESPACE"; do
  if ! "$(kube)" get namespace "$ns" >/dev/null 2>&1; then
    log "creating namespace $ns"
    "$(kube)" create namespace "$ns"
  fi
done

# The upstream OTEL demo chart uses images that need privileged SCCs.
log "relaxing Pod Security for $OTEL_DEMO_NAMESPACE (required by OTEL demo components)"
"$(kube)" label namespace "$OTEL_DEMO_NAMESPACE" \
  pod-security.kubernetes.io/enforce=privileged \
  pod-security.kubernetes.io/audit=privileged \
  pod-security.kubernetes.io/warn=privileged --overwrite

# Namespace-scoped fault-injection RBAC.
apply_manifest "$ROOT/manifests/rbac/harness-rbac.yaml"

if [[ "${ENABLE_NODE_FAULTS:-0}" =~ ^(1|true|yes)$ ]]; then
  warn "applying cluster-scoped node RBAC (node_taint can evict any node)"
  apply_manifest "$ROOT/manifests/rbac/harness-node-rbac.yaml"
fi

log "namespaces and RBAC ready"
