# OTEL Demo on OpenShift — Deployment Guide

## Overview

The OpenTelemetry Demo (Astronomy Shop) runs on OpenShift with SCC and namespace adjustments.

## Prerequisites

- OpenShift cluster (4.11+)
- `oc` CLI logged in
- ~6 GB free RAM for the full demo

## Deployment Steps

### 1. Relax Pod Security for otel-demo namespace

```bash
oc label namespace otel-demo pod-security.kubernetes.io/enforce=privileged \
  pod-security.kubernetes.io/audit=privileged pod-security.kubernetes.io/warn=privileged --overwrite
```

### 2. Add Service Accounts to privileged SCC

```bash
oc adm policy add-scc-to-user privileged system:serviceaccount:otel-demo:grafana
oc adm policy add-scc-to-user privileged system:serviceaccount:otel-demo:prometheus
oc adm policy add-scc-to-user privileged system:serviceaccount:otel-demo:jaeger
oc adm policy add-scc-to-user privileged system:serviceaccount:otel-demo:opentelemetry-demo
oc patch scc privileged --type=json -p '[{"op": "add", "path": "/users/-", "value": "system:serviceaccount:otel-demo:grafana"}, {"op": "add", "path": "/users/-", "value": "system:serviceaccount:otel-demo:opentelemetry-demo"}, {"op": "add", "path": "/users/-", "value": "system:serviceaccount:otel-demo:jaeger"}, {"op": "add", "path": "/users/-", "value": "system:serviceaccount:otel-demo:default"}]'
```

### 3. Apply the OTEL Demo Manifest

All resources deploy to the `otel-demo` namespace:

```bash
oc apply -f research/anomaly_detection/performance_comparison/chaos_engineering/otel-demo/kubernetes/opentelemetry-demo.yaml -n otel-demo
```

### 4. Create Routes (with TLS edge termination)

```bash
oc expose svc frontend-proxy -n otel-demo --name=otel-demo-frontend
oc patch route otel-demo-frontend -n otel-demo -p '{"spec":{"tls":{"termination":"edge"}}}'
oc create route edge grafana --service=grafana -n otel-demo
oc apply -f research/agentic_aiops_architectures/manifests/flagd-ui-route.yaml -n otel-demo
```

### 5. Fix Service References

Ensure frontend-proxy uses correct hostnames (all services are in the same namespace):

```bash
oc set env deployment/frontend-proxy -n otel-demo FLAGD_UI_HOST=flagd GRAFANA_HOST=grafana
```

## Namespace Layout

All components run in the **otel-demo** namespace:

| Namespace   | Components |
|------------|------------|
| **otel-demo** | Frontend, frontend-proxy, flagd, load-generator, Jaeger, Kafka, PostgreSQL, Grafana, Prometheus, OTEL Collector, OpenSearch, app services |

## Access URLs

- **Frontend (Astronomy Shop):** `https://<route-host>/`
- **Grafana:** `https://<grafana-route-host>/` redirects to `https://<grafana-route-host>/grafana/` (sub-path install). If the UI sends you to `http://localhost:3000`, run `./scripts/fix-grafana-openshift.sh` (sets `GF_SERVER_ROOT_URL` from the Route).
- **Flagd UI:** `https://<route-host>/feature`
- **Flagd API (harness):** `https://<flagd-api-host>/api/read` and `/api/write` — use the `flagd-ui-api` route for direct API access
- **Jaeger:** `https://<route-host>/jaeger/ui/`
- **Load Generator:** `https://<route-host>/loadgen/`

Get route hostnames:

```bash
oc get route otel-demo-frontend -n otel-demo -o jsonpath='{.spec.host}'
oc get route flagd-ui-api -n otel-demo -o jsonpath='{.spec.host}'
oc get route grafana -n otel-demo -o jsonpath='{.spec.host}'
```

For the harness, set:

```bash
export FLAGD_READ_URL="https://$(oc get route flagd-ui-api -n otel-demo -o jsonpath='{.spec.host}')/api/read"
export FLAGD_WRITE_URL="https://$(oc get route flagd-ui-api -n otel-demo -o jsonpath='{.spec.host}')/api/write"
```

## Port-Forward (Alternative to Routes)

If Routes return 503, use port-forward:

```bash
oc port-forward -n otel-demo svc/frontend-proxy 8080:8080
# Then open http://localhost:8080
```

## Troubleshooting

### Pod Stuck in Init or ContainerCreating

- **ConfigMap not found:** Ensure flagd-config and product-catalog-products exist in otel-demo (they are created by the manifest).
- **SCC / SecurityContext:** Ensure the service account is in the privileged SCC (see step 2).

### 503 from Route

- Ensure the Route has TLS edge termination: `oc patch route otel-demo-frontend -n otel-demo -p '{"spec":{"tls":{"termination":"edge"}}}'`
- Fix service refs: `oc set env deployment/frontend-proxy -n otel-demo FLAGD_UI_HOST=flagd GRAFANA_HOST=grafana`

### OpenSearch / Grafana SCC Errors

- Add the relevant service account to the privileged SCC.
- Relax namespace pod security to privileged for otel-demo.

### Grafana Route 503 / pod `3/4` / empty Endpoints / OOM (exit 137)

The bundled chart often sets **Grafana memory limit to 150Mi**, which is too low for Grafana 12 → **OOMKilled**, **readiness never passes**, **no Endpoints**, Route returns **503**.

**Fix (idempotent):** from `agentic_aiops_architectures` run:

```bash
./scripts/fix-grafana-openshift.sh
```

This raises the **`grafana`** container to **512Mi** limit / **256Mi** request, aligns **`route/grafana`** `targetPort` with the Service port that forwards to **:3000**, sets **edge TLS**, and sets **`GF_SERVER_ROOT_URL`** / **`GF_SERVER_DOMAIN`** from the Route so Grafana does not redirect browsers to **`http://localhost:3000/grafana/`** (the bundled `grafana.ini` uses an empty `domain`).

### Grafana opens localhost:3000 in the browser

The demo ConfigMap sets `serve_from_sub_path` and a templated `root_url` with an empty `domain`, which resolves to **localhost** behind the OpenShift Route. Re-run **`./scripts/fix-grafana-openshift.sh`** or manually:

```bash
H="$(oc get route grafana -n otel-demo -o jsonpath='{.spec.host}')"
oc set env deployment/grafana -n otel-demo -c grafana \
  GF_SERVER_ROOT_URL="https://${H}/grafana" GF_SERVER_DOMAIN="${H}" GF_SERVER_SERVE_FROM_SUB_PATH=true
```

**Frontend route:** if the main app URL fails, point **`otel-demo-frontend`** at the HTTP Service port (often name **`service`**, port **8080**), not **`tcp-service`**:

```bash
oc patch route otel-demo-frontend -n otel-demo --type=json \
  -p '[{"op": "replace", "path": "/spec/port/targetPort", "value": "service"}]'
```

### Flagd-ui write returns 404 via frontend-proxy

When `POST /feature/api/write` through the main route returns 404, use the direct `flagd-ui-api` route instead:

```bash
export FLAGD_READ_URL="https://$(oc get route flagd-ui-api -n otel-demo -o jsonpath='{.spec.host}')/api/read"
export FLAGD_WRITE_URL="https://$(oc get route flagd-ui-api -n otel-demo -o jsonpath='{.spec.host}')/api/write"
```

The `flagd-ui-api` route targets flagd:4000 directly, bypassing the frontend-proxy.
