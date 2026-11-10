#!/usr/bin/env bash
# Deploy ClickHouse (telemetry store) and wait for it to become ready.
set -euo pipefail

. "$(dirname "$0")/lib.sh"
oc_login

apply_manifest "$ROOT/manifests/clickhouse.yaml"
log "waiting for ClickHouse"
"$(kube)" rollout status deployment/clickhouse -n "$AIOPS_NAMESPACE" --timeout=900s

log "ClickHouse status:"
"$(kube)" get pods,svc,pvc -n "$AIOPS_NAMESPACE" -l app=clickhouse 2>/dev/null || true
