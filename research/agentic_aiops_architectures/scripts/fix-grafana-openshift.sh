#!/usr/bin/env bash
# Fix Grafana on OpenShift OTEL demo:
# 1) Raise memory on the main grafana container (default chart 150Mi causes OOM / exit 137).
# 2) Align Route targetPort with the Service port that forwards to Grafana :3000.
# 3) Ensure edge TLS on the Grafana route.
# 4) Set GF_SERVER_ROOT_URL / domain from the Route (chart uses empty domain → redirects to localhost:3000).
#
# Usage: ./scripts/fix-grafana-openshift.sh
# Requires: oc, logged in, namespace otel-demo

set -euo pipefail
NS="${NS:-otel-demo}"

if ! oc get deployment grafana -n "$NS" &>/dev/null; then
  echo "No deployment/grafana in namespace $NS — skip."
  exit 0
fi

echo "=== 1. Increase Grafana container memory (avoids OOM / probe failures) ==="
oc set resources deployment/grafana -n "$NS" -c grafana \
  --limits=memory=512Mi --requests=memory=256Mi

echo "=== 2. Align Grafana Route to Service port (name that targets pod :3000) ==="
if oc get route grafana -n "$NS" &>/dev/null; then
  PORT_NAME="$(oc get svc grafana -n "$NS" -o json | python3 -c "
import json, sys
svc = json.load(sys.stdin)
ports = svc.get('spec', {}).get('ports') or []
for p in ports:
    tp = p.get('targetPort')
    if tp == 3000 or str(tp) == '3000':
        print(p.get('name', p.get('port', 'service')))
        break
else:
    if ports:
        print(ports[0].get('name') or str(ports[0].get('port', 'service')))
")"
  if [[ -n "$PORT_NAME" ]]; then
    oc patch route grafana -n "$NS" --type=json \
      -p="[{\"op\": \"replace\", \"path\": \"/spec/port/targetPort\", \"value\": \"${PORT_NAME}\"}]" 2>/dev/null || true
    echo "    Route targetPort -> ${PORT_NAME}"
  fi
  oc patch route grafana -n "$NS" -p '{"spec":{"tls":{"termination":"edge"}}}' 2>/dev/null || true
else
  echo "    No route/grafana; create with: oc create route edge grafana --service=grafana -n $NS"
fi

echo "=== 3. Public URL env (stops redirect to http://localhost:3000/grafana/) ==="
ROUTE_HOST="$(oc get route grafana -n "$NS" -o jsonpath='{.spec.host}' 2>/dev/null || true)"
if [[ -n "$ROUTE_HOST" ]]; then
  oc set env deployment/grafana -n "$NS" -c grafana \
    GF_SERVER_ROOT_URL="https://${ROUTE_HOST}/grafana" \
    GF_SERVER_DOMAIN="${ROUTE_HOST}" \
    GF_SERVER_SERVE_FROM_SUB_PATH=true
  echo "    GF_SERVER_ROOT_URL=https://${ROUTE_HOST}/grafana"
else
  echo "    Skip (no route/grafana host yet)"
fi

echo "=== 4. Rollout status ==="
oc rollout status deployment/grafana -n "$NS" --timeout=180s || true

echo ""
echo "Check:"
echo "  oc get pods -n $NS -l app.kubernetes.io/name=grafana   # want 4/4 Ready"
echo "  oc get endpoints grafana -n $NS"
echo "  oc get route grafana -n $NS"
TARGET_HOST=$(oc get route grafana -n "$NS" -o jsonpath='{.spec.host}' 2>/dev/null || true)
[[ -n "$TARGET_HOST" ]] && echo "  https://${TARGET_HOST}"
