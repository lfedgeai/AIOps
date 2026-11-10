#!/usr/bin/env bash
# Patch the OTEL Collector to export to ClickHouse and restart it.
set -euo pipefail

. "$(dirname "$0")/lib.sh"
oc_login

log "patching collector ConfigMap"
"$PYTHON_BIN" "$ROOT/scripts/patch-otel-collector-clickhouse.py" \
  --namespace "$OTEL_DEMO_NAMESPACE" \
  --endpoint "${CLICKHOUSE_COLLECTOR_ENDPOINT:-tcp://clickhouse.${AIOPS_NAMESPACE}.svc.cluster.local:9000?dial_timeout=10s}"

log "restarting collector"
"$(kube)" rollout restart deployment/otel-collector -n "$OTEL_DEMO_NAMESPACE" >/dev/null 2>&1 || true
"$(kube)" rollout status deployment/otel-collector -n "$OTEL_DEMO_NAMESPACE" --timeout=300s \
  || warn "collector rollout did not complete; check pods"

log "waiting for ClickHouse schema/tables (first ingest can take ~30s)"
for i in $(seq 1 12); do
  tables=$("$(kube)" exec -n "$AIOPS_NAMESPACE" deploy/clickhouse -- \
    clickhouse-client --query "SHOW TABLES FROM otel" 2>/dev/null || true)
  if echo "$tables" | grep -q otel_logs; then
    log "ClickHouse schema ready: $(echo "$tables" | tr '\n' ' ')"
    exit 0
  fi
  sleep 10
done
warn "ClickHouse tables not visible yet. Check collector logs:"
"$(kube)" logs -n "$OTEL_DEMO_NAMESPACE" deploy/otel-collector --tail=40 2>/dev/null || true
