#!/usr/bin/env bash
# Deploy LF Edge AIOps telemetry stack on OpenShift (new cluster / new namespace layout).
# Idempotent where possible. Requires: oc, helm, helm repo open-telemetry.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS_DEMO="${OTEL_DEMO_NAMESPACE:-otel-demo}"
HELM_RELEASE="${OTEL_HELM_RELEASE:-lfedge-otel-demo}"
CHART_VERSION="${OTEL_DEMO_CHART_VERSION:-0.38.6}"
WAIT_OPT="${HELM_WAIT:-false}" # set HELM_WAIT=true to block until all demo pods healthy (often slow)

echo "==> Helm: OpenTelemetry demo (${HELM_RELEASE}, chart ${CHART_VERSION}, ns ${NS_DEMO})"
if [[ "${WAIT_OPT}" == "true" ]]; then
  helm upgrade --install "${HELM_RELEASE}" open-telemetry/opentelemetry-demo \
    --version "${CHART_VERSION}" -n "${NS_DEMO}" --create-namespace \
    --wait --timeout=25m
else
  helm upgrade --install "${HELM_RELEASE}" open-telemetry/opentelemetry-demo \
    --version "${CHART_VERSION}" -n "${NS_DEMO}" --create-namespace
fi

echo "==> ClickHouse (telemetry store — wait until Ready)"
oc apply -f "${ROOT}/manifests/clickhouse.yaml"
oc rollout status deployment/clickhouse -n "${NS_DEMO}" --timeout=600s

echo "==> Patch OTEL collector → ClickHouse"
python3 "${ROOT}/scripts/patch-otel-collector-clickhouse.py"

echo "==> Restart collector to load new relay config"
oc rollout restart deployment/otel-collector -n "${NS_DEMO}" || true
oc rollout status deployment/otel-collector -n "${NS_DEMO}" --timeout=300s || true

echo "==> MLflow (agentic-aiops namespace — harness experiment tracking)"
oc apply -f "${ROOT}/manifests/mlflow.yaml"

echo "==> Route: flagd HTTP API (chaos / feature flags for harness)"
oc apply -f "${ROOT}/manifests/flagd-ui-route.yaml"

echo ""
echo "Done. Next steps:"
echo "  1) Wait for demo pods if helm was installed without --wait:  oc get pods -n ${NS_DEMO} -w"
echo "  2) Verify ClickHouse tables:  oc exec -n ${NS_DEMO} deploy/clickhouse -- clickhouse-client --database otel --query 'SHOW TABLES'"
echo "  3) Flagd API (set FLAGD_VERIFY_SSL=false for self-signed routes if needed):"
oc get route flagd-ui-api -n "${NS_DEMO}" -o jsonpath='  export FLAGD_READ_URL=https://{.spec.host}/api/read{'\n'}  export FLAGD_WRITE_URL=https://{.spec.host}/api/write{'\n'}' 2>/dev/null || true
echo "  4) Harness MLflow (edge TLS):"
oc get route mlflow -n agentic-aiops -o jsonpath='  export MLFLOW_TRACKING_URI=https://{.spec.host}{'\n'}  export MLFLOW_TRACKING_INSECURE_TLS=true{'\n'}' 2>/dev/null || true
echo "  5) ClickHouse from your laptop (optional):  oc port-forward -n ${NS_DEMO} svc/clickhouse 8123:8123"
echo "  6) Integrate EvalHub env wiring:  ${ROOT}/scripts/patch-evalhub-lfedge-aiops-env.sh"
