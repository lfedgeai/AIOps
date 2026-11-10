#!/usr/bin/env bash
# Deploy standalone MLflow and print its in-cluster and Route URLs.
set -euo pipefail

. "$(dirname "$0")/lib.sh"
oc_login

apply_manifest "$ROOT/manifests/mlflow.yaml"
log "waiting for MLflow"
"$(kube)" rollout status deployment/mlflow -n "$AIOPS_NAMESPACE" --timeout=600s

log "MLflow in-cluster: http://mlflow.${AIOPS_NAMESPACE}.svc.cluster.local:5000"
if command -v oc >/dev/null 2>&1; then
  host="$(oc get route mlflow -n "$AIOPS_NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  [[ -n "$host" ]] && log "MLflow Route:     https://${host}"
fi
