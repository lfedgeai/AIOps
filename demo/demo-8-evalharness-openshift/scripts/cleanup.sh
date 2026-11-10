#!/usr/bin/env bash
# Remove the standalone stack. By default deletes workloads but keeps namespaces
# and PVCs. Use --all to also remove PVCs and namespaces (destroys telemetry).
set -euo pipefail

. "$(dirname "$0")/lib.sh"
oc_login

ALL=0
[[ "${1:-}" == "--all" ]] && ALL=1

log "deleting routes/workloads"
for f in "$ROOT/manifests/harness-job.yaml" "$ROOT/manifests/vllm.yaml" "$ROOT/manifests/mlflow.yaml" \
         "$ROOT/manifests/clickhouse.yaml"; do
  render "$f" | "$(kube)" delete -f - --ignore-not-found 2>/dev/null || true
done

# OTEL demo: delete via the rendered manifest.
sed "s/otel-demo/${OTEL_DEMO_NAMESPACE}/g" "$ROOT/manifests/otel-demo.yaml" \
  | "$(kube)" delete -n "$OTEL_DEMO_NAMESPACE" -f - --ignore-not-found 2>/dev/null || true

if [[ $ALL -eq 1 ]]; then
  warn "deleting PVCs and namespaces (irreversible)"
  for ns in "$AIOPS_NAMESPACE" "$OTEL_DEMO_NAMESPACE"; do
    "$(kube)" delete namespace "$ns" --ignore-not-found --timeout=300s || true
  done
fi

log "cleanup complete"
