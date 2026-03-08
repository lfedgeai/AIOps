#!/bin/bash
# Deploy OTEL Demo to OpenShift with SCC fixes
# Run from repo root: ./research/agentic_aiops_architectures/scripts/deploy-otel-openshift.sh

set -e
# Script dir -> agentic_aiops_architectures -> research -> AIOps root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OTEL_MANIFEST="${REPO_ROOT}/research/anomaly_detection/performance_comparison/chaos_engineering/otel-demo/kubernetes/opentelemetry-demo.yaml"

echo "=== 0. Ensure otel-demo namespace exists ==="
oc create namespace otel-demo 2>/dev/null || true

echo "=== 1. Relax Pod Security for otel-demo ==="
oc label namespace otel-demo pod-security.kubernetes.io/enforce=privileged \
  pod-security.kubernetes.io/audit=privileged pod-security.kubernetes.io/warn=privileged --overwrite 2>/dev/null || true

echo "=== 2. Add SAs to privileged SCC ==="
for sa in grafana prometheus jaeger opentelemetry-demo; do
  oc adm policy add-scc-to-user privileged system:serviceaccount:otel-demo:$sa 2>/dev/null || true
done
oc patch scc privileged --type=json -p '[{"op": "add", "path": "/users/-", "value": "system:serviceaccount:otel-demo:grafana"}, {"op": "add", "path": "/users/-", "value": "system:serviceaccount:otel-demo:opentelemetry-demo"}, {"op": "add", "path": "/users/-", "value": "system:serviceaccount:otel-demo:jaeger"}]' 2>/dev/null || true

echo "=== 3. Apply OTEL Demo Manifest (all resources to otel-demo ns) ==="
oc apply -f "$OTEL_MANIFEST" -n otel-demo

echo "=== 4. Create Routes ==="
oc expose svc frontend-proxy -n otel-demo --name=otel-demo-frontend 2>/dev/null || true
oc patch route otel-demo-frontend -n otel-demo -p '{"spec":{"tls":{"termination":"edge"}}}' 2>/dev/null || true
oc create route edge grafana --service=grafana -n otel-demo 2>/dev/null || true
# Direct route to flagd-ui for API read/write (bypasses frontend-proxy)
FLAGD_ROUTE="${SCRIPT_DIR}/../manifests/flagd-ui-route.yaml"
if [ -f "$FLAGD_ROUTE" ]; then
  oc apply -f "$FLAGD_ROUTE" -n otel-demo 2>/dev/null || true
else
  oc expose svc flagd -n otel-demo --name=flagd-ui-api --port=4000 2>/dev/null || true
  oc patch route flagd-ui-api -n otel-demo -p '{"spec":{"tls":{"termination":"edge"}}}' 2>/dev/null || true
fi

echo "=== 5. Fix service references (all in same ns; ensure correct hosts) ==="
oc set env deployment/frontend-proxy -n otel-demo FLAGD_UI_HOST=flagd GRAFANA_HOST=grafana 2>/dev/null || true

echo "=== Done. Waiting for pods... ==="
sleep 10
oc get pods -n otel-demo
echo ""
echo "Working URLs (use HTTPS):"
oc get route otel-demo-frontend -n otel-demo -o jsonpath='  Main app:    https://{.spec.host}/
' 2>/dev/null
oc get route otel-demo-frontend -n otel-demo -o jsonpath='  Flagd UI:    https://{.spec.host}/feature
' 2>/dev/null
oc get route flagd-ui-api -n otel-demo -o jsonpath='  Flagd API:   https://{.spec.host}/api/read (direct, for harness)
' 2>/dev/null
oc get route otel-demo-frontend -n otel-demo -o jsonpath='  Grafana:     https://{.spec.host}/grafana/
' 2>/dev/null
oc get route otel-demo-frontend -n otel-demo -o jsonpath='  Jaeger:      https://{.spec.host}/jaeger/
' 2>/dev/null
oc get route grafana -n otel-demo -o jsonpath='  Grafana dir: https://{.spec.host}
' 2>/dev/null
