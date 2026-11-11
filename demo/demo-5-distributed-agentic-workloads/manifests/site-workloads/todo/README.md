# demo-5 self-built Python todo app

**Scaffold / not slice-complete** for hub/registry/GitOps. App OTEL signals (traces, metrics, logs) are implemented; hub ClickHouse path and image registry pin still required for Phase 2 exit.

Minimal FastAPI + PostgreSQL todo service with native OpenTelemetry for site workloads.

## Layout

| Path | Purpose |
|------|---------|
| `src/` | App source, `requirements.txt`, `Containerfile` |
| `base/` | Kubernetes manifests (kustomize base) |
| `overlays/<site_id>/` | Per-site `SITE_ID`, `HUB_OTLP_ENDPOINT`, `site_id` labels |

## API

- `GET /health` — liveness
- `GET /ready` — DB ping
- `GET/POST /api` — list / create todos
- `GET/PATCH/DELETE /api/{id}`

## Build & push (OpenShift internal registry)

Enable the registry default Route on the hub, then build/push (hostnames vary by cluster):

```bash
cd src
# Example after: oc registry info / defaultRoute on configs.imageregistry
REG=$(oc get route default-route -n openshift-image-registry -o jsonpath='{.spec.host}')
podman build -t "$REG/todo-demo/demo5-todo:0.1.0" -f Containerfile .
podman login -u $(oc whoami) -p $(oc whoami -t) "$REG" --tls-verify=false   # lab; prefer CA trust
podman push "$REG/todo-demo/demo5-todo:0.1.0"
```

Pin the image via kustomize `images:` (and site pull secret + trusted CA on MicroShift) before deploy. Do **not** use `quay.io/julin/todo`.

## Local run (optional)

```bash
# Discrete POSTGRES_* (cluster path) or DATABASE_URL (local shortcut):
export POSTGRES_USER=todo POSTGRES_PASSWORD=todo POSTGRES_DB=todo
export POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=5432
# export DATABASE_URL=postgresql+psycopg2://todo:todo@127.0.0.1:5432/todo
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
export SITE_ID=site-local
pip install -r requirements.txt
uvicorn app:app --reload --port 8080
```

## OTEL

Exports OTLP/HTTP **traces, metrics, and logs** to `OTEL_EXPORTER_OTLP_ENDPOINT` (site collector).
App metrics: `http.server.request.count|duration|errors`. Postgres metrics via site collector scrape of postgres_exporter.
Resource attributes include `service.name`, `site_id`, `app`, `workload_class`, `component`, `k8s.namespace.name`.
