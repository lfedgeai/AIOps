#!/usr/bin/env bash
# Deploy the complete standalone AIOps evaluation stack on OCP/RHOAI.
#
#   ./scripts/deploy.sh                 # core stack (OTEL demo + ClickHouse + MLflow)
#   VLLM_ENABLED=1 ./scripts/deploy.sh  # also deploy the raw vLLM fallback
#   ENABLE_EVALHUB=1 ./scripts/deploy.sh
#
# Prereqs: cluster-admin (or enough namespace/SCC rights), oc/kubectl,
# python3 + requirements, and a filled .env.
set -euo pipefail

. "$(dirname "$0")/lib.sh"
ensure_dirs

log "=== Phase 0: preflight ==="
"$ROOT/scripts/00-preflight.sh"

log "=== Phase 1: namespaces + RBAC ==="
"$ROOT/scripts/10-namespaces.sh"

log "=== Phase 2: OTEL Demo ==="
"$ROOT/scripts/20-otel-demo.sh"

log "=== Phase 3: ClickHouse ==="
"$ROOT/scripts/30-clickhouse.sh"

log "=== Phase 4: collector -> ClickHouse ==="
"$ROOT/scripts/40-collector.sh"

log "=== Phase 5: MLflow (RHOAI-managed — skipping standalone deploy) ==="
# MLflow is now managed by the RHOAI MLflow operator (MLflow CR in agentic-aiops-harness).
# The standalone deployment was removed. The RHOAI MLflow runs in redhat-ods-applications.
# To create the MLflow CR: oc apply -f - <<EOF
# apiVersion: mlflow.opendatahub.io/v1
# kind: MLflow
# metadata: {name: mlflow, namespace: agentic-aiops-harness}
# spec: {replicas: 1, backendStoreUri: "sqlite:////mlflow/data/mlflow.db", storage: {accessModes: [ReadWriteOnce], storageClassName: gp3-csi, resources: {requests: {storage: 10Gi}}}, serveArtifacts: true}
# EOF

if [[ "${VLLM_ENABLED:-0}" =~ ^(1|true|yes)$ ]]; then
  log "=== Phase 6: vLLM fallback ==="
  "$ROOT/scripts/60-vllm.sh"
fi

if [[ "${ENABLE_EVALHUB:-0}" =~ ^(1|true|yes)$ ]]; then
  log "=== Phase 7: RHOAI EvalHub ==="
  "$ROOT/scripts/70-evalhub.sh"
fi

log "deployment complete"
cat <<EOF

Next steps:
  1. Create model credentials:  ./scripts/create-secrets.sh
  2. Smoke test one cell:       ./scripts/smoke_test.sh
  3. Run the harness:           ./scripts/run_harness.sh --scenario a --flag scale_zero --variant cart
  4. Run a matrix:              ./scripts/run_matrix.sh
  5. MLflow Route:
EOF
if command -v oc >/dev/null 2>&1; then
  oc get route mlflow -n "$AIOPS_NAMESPACE" -o jsonpath='     https://{.spec.host}{"\n"}' 2>/dev/null || true
fi
