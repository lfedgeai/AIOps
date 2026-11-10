# Deployment notes — OCP / RHOAI on AWS

This file covers the platform-side decisions behind `.env`. The README is the
quick start; this is the checklist for a real cluster.

---

## 1. Cluster shape

Assumptions:
- OpenShift 4.x on AWS (IPI or UPI), cluster-admin available.
- 3 workers minimum for the full OTEL Demo; the demo itself is ~6 GB RAM.
- A default StorageClass (usually `gp3-csi`). ClickHouse wants 50 Gi; MLflow 10 Gi.
- Outbound access to `quay.io`, `ghcr.io`, `registry.access.redhat.com`, and
  your model endpoint. If the cluster is air-gapped, mirror the images and
  pre-seed `context/opentelemetry-demo`.
- `LEAN_OTEL_DEMO=1` scales down OpenSearch/Jaeger/Prometheus/Grafana after the
  demo is healthy. The harness only needs the app services, collector, flagd,
  and load generator.

Suggested AWS sizing:
| Role | Instance | Notes |
|---|---|---|
| control plane | `m6i.xlarge` × 3 | standard IPI |
| workers | `m6i.4xlarge` × 3 | OTEL Demo + ClickHouse + MLflow |
| GPU (optional) | `g5.2xlarge` × 1 | 24 GB; one 7B–14B AWQ model |
| GPU (full lineup) | `g5.12xlarge` or MaaS | 27B/30B-class models need more VRAM |

The `.env` block `AWS_*` captures these values for cluster provisioning. This
repo deploys **onto** an existing cluster; it does not create the cluster.

---

## 2. RHOAI 3.4+ (for EvalHub and model serving)

Install the RHOAI operator from OperatorHub (`stable-3.4`), then create a
DataScienceCluster. For the AIOps harness + EvalHub UI, the relevant components
are:

```yaml
apiVersion: datasciencecluster.opendatahub.io/v2
kind: DataScienceCluster
metadata:
  name: default-dsc
spec:
  components:
    dashboard:
      managementState: Managed
    mlflowoperator:
      managementState: Managed      # optional; the repo also deploys standalone MLflow
    trustyai:
      managementState: Managed      # required for EvalHub / Evaluations UI
    kserve:
      managementState: Managed      # if serving models in-cluster
    workbenches:
      managementState: Managed      # if using a Workbench
```

Wait for `oc get dsc default-dsc -o jsonpath='{.status.phase}'` to be `Ready`.

The repo's standalone MLflow is independent of RHOAI MLflow. If you prefer
RHOAI MLflow, set `MLFLOW_TRACKING_URI` to its service/route and set
`ALLOW_CLUSTER_MLFLOW=1`; the harness code uses standard MLflow APIs, but RHOAI
MLflow may require token/RBAC and workspace configuration.

---

## 3. Model endpoints

Three options, in order of least operational effort:

### A. MaaS / LiteLLM gateway (recommended)

Set per-agent `*_API_BASE`, `*_API_KEY`, `*_MODEL` in `.env`. The harness only
needs an OpenAI-compatible `/v1/chat/completions` endpoint with tool calling.
This is how the original study ran.

### B. RHOAI KServe / vLLM

Serve the model with an OpenAI-compatible runtime and tool calling enabled:

```
--enable-auto-tool-choice --tool-call-parser hermes
```

Then set the agent's `*_API_BASE` to `https://<route>/v1` and `*_MODEL` to the
served model name. `hermes` works for Qwen; other families need the matching
parser.

### C. Raw vLLM fallback in this repo

```bash
VLLM_ENABLED=1 VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct ./scripts/deploy.sh
```

This is a plain Deployment, not a RHOAI InferenceService. It is useful for a
quick single-model demo; the full six-agent lineup needs MaaS or more GPUs.
If the vLLM pod is rejected by the namespace SCC, bind a permissive SCC to the
`default` service account in `agentic-aiops` (lab only):

```bash
oc adm policy add-scc-to-user anyuid -z default -n agentic-aiops
```

---

## 4. EvalHub provider

The provider is real: it runs one harness cell per EvalHub job. It needs:

1. RHOAI 3.4+ with `trustyai` Managed and a cluster pull secret that includes
   `registry.redhat.io` (run `./scripts/check-pull-secret.sh`).
2. Tenant namespace labeled (default `EVALHUB_NAMESPACE`):
   ```bash
   oc label namespace "$EVALHUB_NAMESPACE" evalhub.trustyai.opendatahub.io/tenant=true --overwrite
   ```
3. Tenant EvalHub with `spec.database.type=sqlite`.
4. Platform EvalHub in `EVALHUB_PLATFORM_NAMESPACE` (defaults to
   `RHOAI_APPLICATIONS_NAMESPACE`, which defaults to `redhat-ods-applications`)
   for the UI health check.
5. Provider ConfigMap in the same applications namespace with an ownerReference
   to the `default-trustyai` TrustyAI CR.
6. `lfedge-aiops` in the tenant EvalHub `spec.providers`.
7. RBAC for the EvalHub job service account in the OTEL demo namespace:
   ```bash
   oc create rolebinding aiops-evalhub-harness -n "$OTEL_DEMO_NAMESPACE" \
     --role=aiops-harness-faults \
     --serviceaccount=<job-ns>:<job-sa>
   ```

`./scripts/70-evalhub.sh` automates 1–6. The last step depends on which service
account the EvalHub operator uses for jobs; inspect a running job pod and bind
that SA. If RHOAI was installed into different namespaces, set
`RHOAI_OPERATOR_NAMESPACE`, `RHOAI_APPLICATIONS_NAMESPACE`, and
`EVALHUB_NAMESPACE` in `.env`; the scripts no longer assume the defaults.

For a production install, move `LFEDGE_MODEL_API_KEY` out of the provider
ConfigMap and into a Secret injected into EvalHub jobs; the ConfigMap path is a
demo convenience.

Known RHOAI 3.4 quirks:
- The dashboard's federated Evaluations UI cannot be durably patched; use the
  EvalHub Route with `Authorization` + `X-Tenant` (the scripts do).
- Multiple EvalHub CRs can cross-wire tenant RoleBindings; run
  `scripts/fix-evalhub-tenant-rbac.sh`.
- Provider image must be pullable from the job pod (internal registry or Quay).

---

## 5. Harness RBAC

`manifests/rbac/harness-rbac.yaml` grants, in `otel-demo` only:

- pods get/list/watch/delete
- pods/log get, pods/exec create
- events/services/configmaps get/list/watch
- deployments/replicasets/statefulsets get/list/watch/patch/update
- networkpolicies get/list/create/delete/patch

`manifests/rbac/harness-node-rbac.yaml` is optional and grants cluster-wide
node get/list/watch/patch for `node_taint`. Do not apply it on a shared RHOAI
cluster.

---

## 6. Deployment sequence

```bash
cp .env.example .env
$EDITOR .env

./scripts/build-image.sh

./scripts/00-preflight.sh
./scripts/check-pull-secret.sh   # required before installing RHOAI
./scripts/deploy.sh
./scripts/create-secrets.sh
./scripts/smoke_test.sh

# Optional:
VLLM_ENABLED=1 ./scripts/deploy.sh
ENABLE_EVALHUB=1 ./scripts/deploy.sh
```

Long runs: use `./scripts/run_job.sh` (Job + PVC) or a RHOAI Pipeline. A
Workbench is fine for a smoke test but idle-culling will kill a 10–14 hour
matrix.

---

## 7. Air-gapped / restricted clusters

- Mirror the OTEL Demo, ClickHouse, MLflow, vLLM, and harness images.
- Pre-seed `context/opentelemetry-demo` (already vendored).
- Pre-build `data/rag_cache` or allow the harness to build it locally.
- Set `HARNESS_INSTALL_DEPS=0`; dependencies are in the image.
- If using EvalHub, host the provider image in the internal registry.
