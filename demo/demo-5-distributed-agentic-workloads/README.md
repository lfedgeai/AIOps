# Demo 5 — Distributed agentic workloads + AIOps overlay

First build slice: **Phase 0 → Phase 2** (see [AGENTS.md](./AGENTS.md)).

## Hub namespaces

| Namespace | Role |
|-----------|------|
| `demo5-observability` | Hub OTEL collector, OTLP **Route** (`otel-intake`), ClickHouse |
| `demo5-aiops` | Central AIOps (empty until Phase 4) |
| `demo5-openshell` | Gateway stub (Phase 3+) |

Pattern-owned: `openshift-gitops`, ACM, `vault`, etc.

## Quickstart (Phase 0)

1. `oc login` as `cluster-admin` on the lab OpenShift hub.
2. Copy pattern secrets template to home dir (never commit):

   ```bash
   cp ../../ValidatedPattern/values-secret.yaml.template ~/values-secret-multicloud-gitops.yaml
   # edit secrets per https://validatedpatterns.io/learn/getting-started-multi-cloud-gitops/
   ```

3. Install pattern + hub observe plane:

   ```bash
   chmod +x scripts/*.sh
   ./scripts/install-hub-pattern.sh
   # Or hub only: SKIP_PATTERN=1 ./scripts/install-hub-pattern.sh
   ```

   Storage (recommended before durable ClickHouse):

   ```bash
   ./scripts/install-lvms.sh   # creates default SC lvms-vg1 (lab loop if no spare disk)
   oc apply -k manifests/hub  # ClickHouse PVC uses lvms-vg1
   ```

4. Record the Route in `config/sites.yaml`:

   ```bash
   oc -n demo5-observability get route otel-intake -o jsonpath='{.spec.host}{"\n"}'
   ```

## ValidatedPattern/

Vendored Multicloud GitOps at repo root `ValidatedPattern/` (upstream tip in `ValidatedPattern/.upstream-commit`).  
`values-hub.yaml` adds `demo5-*` namespaces and an Argo Application pointing at `demo/.../manifests/hub` on `https://github.com/lfedgeai/AIOps.git` (branch `demo/demo-5-distributed-agentic-workloads`). Until that path is pushed/synced, use `oc apply -k manifests/hub`.

## Lab secrets (`.env`)

Copy `.env.example` → `.env` (gitignored) and set RHSM for MicroShift:

```bash
cp .env.example .env
# edit RHSM_ORG + RHSM_ACTIVATION_KEY (preferred) or RHSM_USER + RHSM_PASSWORD
./scripts/provision-site-vms.sh   # sources .env automatically
```

Prefer **activation key** over password. Pattern Vault secrets stay in `~/values-secret-multicloud-gitops.yaml` (separate). Later Phase 2+ can move RHSM into Vault/ESO if needed.

## Phase 1–2 (next)

- `scripts/provision-site-vms.sh` / `bootstrap-site-k8s.sh` — MicroShift site + RHACM or Argo registration  
- Site overlay: `manifests/site-workloads/todo/overlays/site-a` (set `HUB_OTLP_ENDPOINT`)  
- Image: build/push to OpenShift internal registry; pin kustomize `images:`  
- Verify fail-closed: `SITE_ID=site-a ./scripts/deploy-todo-otel.sh`

## Observe / act

- **Observe handover:** ClickHouse in `demo5-observability`  
- **Act:** Phase 4+ via MCP/tools (not via ClickHouse)
