#!/usr/bin/env bash
# Deploy the vendored OpenTelemetry Demo (Astronomy Shop) and OpenShift routes.
set -euo pipefail

. "$(dirname "$0")/lib.sh"
oc_login

log "applying vendored OTEL Demo manifest to $OTEL_DEMO_NAMESPACE"
sed "s/otel-demo/${OTEL_DEMO_NAMESPACE}/g" "$ROOT/manifests/otel-demo.yaml" \
  | "$(kube)" apply -n "$OTEL_DEMO_NAMESPACE" -f -

# The upstream demo needs privileged SCCs on OpenShift for several components.
if command -v oc >/dev/null 2>&1; then
  for sa in default grafana prometheus jaeger opentelemetry-demo otel-collector opensearch; do
    oc adm policy add-scc-to-user privileged -z "$sa" -n "$OTEL_DEMO_NAMESPACE" >/dev/null 2>&1 || \
      warn "could not bind privileged SCC to $sa (cluster-admin required)"
  done
fi

log "waiting for statefulset/valkey-cart"
"$(kube)" rollout status statefulset/valkey-cart -n "$OTEL_DEMO_NAMESPACE" --timeout=300s \
  || warn "statefulset/valkey-cart not ready yet; check pods"
for dep in frontend-proxy frontend cart checkout otel-collector load-generator \
           ad currency email image-provider payment product-catalog quote recommendation shipping \
           accounting fraud-detection; do
  log "waiting for deployment/$dep"
  "$(kube)" rollout status "deployment/$dep" -n "$OTEL_DEMO_NAMESPACE" --timeout=300s \
    || warn "deployment/$dep not ready yet; check pods"
done

if [[ "${LEAN_OTEL_DEMO:-0}" =~ ^(1|true|yes)$ ]]; then
  log "LEAN_OTEL_DEMO=1: scaling down OpenSearch/Jaeger/Prometheus/Grafana"
  "$(kube)" scale statefulset/opensearch --replicas=0 -n "$OTEL_DEMO_NAMESPACE" 2>/dev/null || true
  "$(kube)" scale deployment/jaeger --replicas=0 -n "$OTEL_DEMO_NAMESPACE" 2>/dev/null || true
  "$(kube)" scale deployment/prometheus --replicas=0 -n "$OTEL_DEMO_NAMESPACE" 2>/dev/null || true
  "$(kube)" scale deployment/grafana --replicas=0 -n "$OTEL_DEMO_NAMESPACE" 2>/dev/null || true
fi

# Routes (idempotent).
if command -v oc >/dev/null 2>&1; then
  oc expose svc/frontend-proxy -n "$OTEL_DEMO_NAMESPACE" --name=otel-demo-frontend >/dev/null 2>&1 || true
  oc patch route otel-demo-frontend -n "$OTEL_DEMO_NAMESPACE" \
    -p '{"spec":{"tls":{"termination":"edge"}}}' >/dev/null 2>&1 || true
  oc create route edge grafana --service=grafana -n "$OTEL_DEMO_NAMESPACE" >/dev/null 2>&1 || true
  oc create route edge flagd-ui-api --service=flagd --port=tcp-service-0 -n "$OTEL_DEMO_NAMESPACE" >/dev/null 2>&1 || true
  oc create route edge jaeger --service=jaeger-query -n "$OTEL_DEMO_NAMESPACE" >/dev/null 2>&1 || true
fi

log "OTEL Demo routes:"
"$(kube)" get route -n "$OTEL_DEMO_NAMESPACE" 2>/dev/null || true
log "OTEL Demo pods:"
"$(kube)" get pods -n "$OTEL_DEMO_NAMESPACE" 2>/dev/null | head -40 || true
