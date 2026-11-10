# AWS / OCP inputs

The deployment scripts run against an **existing** OpenShift cluster. If the
cluster still needs to be built, these are the inputs captured in `.env` and
the decisions behind them.

## AWS account and DNS

| Input | `.env` key | Notes |
|---|---|---|
| AWS region | `AWS_REGION` | pick the region with your GPU quota |
| Cluster name | `AWS_CLUSTER_NAME` | used in the cluster domain |
| Base domain | `AWS_BASE_DOMAIN` | must have a Route 53 hosted zone |
| Pull secret | — | `openshift-install` `pullSecret` (not stored in `.env`) |
| SSH key | — | `openshift-install` `sshKey` |

## Cluster shape

- OCP 4.16+ (4.18 is a good target for RHOAI 3.4).
- 3 control-plane nodes, 3 workers. The OTEL Demo is ~6 GB RAM; ClickHouse
  wants 8 GiB; MLflow 2 GiB. `m6i.4xlarge` workers are comfortable.
- StorageClass: `gp3-csi` (EBS CSI). ClickHouse 50 Gi, MLflow 10 Gi, harness
  state 20 Gi.
- Optional GPU: `g5.2xlarge` (24 GB) for a 7B–14B AWQ model. The full
  six-agent lineup should use MaaS or multiple GPUs.
- If you use the raw vLLM fallback, install the NVIDIA GPU Operator and label
  the GPU node; the manifest tolerates `nvidia.com/gpu:NoSchedule`.

## Red Hat pull secret (required for RHOAI images)

RHOAI operator images are hosted in `registry.redhat.io`. The cluster's global
pull secret must contain credentials for `registry.redhat.io` (the RHOAI
operator/catalog images). The pull secret from cloud.redhat.com normally also
includes `registry.connect.redhat.com`, `quay.io`, and `cloud.openshift.com`.

Check it without printing credentials:

```bash
./scripts/check-pull-secret.sh
```

If `registry.redhat.io` is missing, merge your Red Hat pull secret into
`openshift-config/pull-secret` before installing the RHOAI operator. On a disconnected cluster, mirror the RHOAI catalog/operator
images and configure the appropriate `ImageDigestMirrorSet`/`ImageTagMirrorSet`.

`CHECK_PULL_SECRET=1` in `.env` runs this check from `00-preflight.sh`.

## RHOAI namespaces (not always the defaults)

RHOAI has three relevant namespaces, all configurable in `.env`:

| `.env` key | Default | What lives there |
|---|---|---|
| `RHOAI_OPERATOR_NAMESPACE` | `redhat-ods-operator` | RHOAI operator subscription/CSV |
| `RHOAI_APPLICATIONS_NAMESPACE` | `redhat-ods-applications` | dashboard, MLflow, TrustyAI, platform EvalHub |
| `EVALHUB_NAMESPACE` | `rhods-notebooks` | tenant/workbench namespace where EvalHub jobs run |

`RHOAI_NAMESPACE` remains as a legacy alias for
`RHOAI_APPLICATIONS_NAMESPACE`. `EVALHUB_PLATFORM_NAMESPACE` defaults to the
applications namespace.

The applications namespace is set by RHOAI's `DSCInitialization`; if you
installed RHOAI with a non-default value, put that value in
`RHOAI_APPLICATIONS_NAMESPACE`:

```yaml
apiVersion: dscinitialization.opendatahub.io/v1
kind: DSCInitialization
metadata:
  name: default-dsci
spec:
  applicationsNamespace: redhat-ods-applications
```

## RHOAI install

1. Install the `rhods-operator` from OperatorHub (`stable-3.4`) in
   `RHOAI_OPERATOR_NAMESPACE`. Example subscription (the OperatorGroup must
   exist in the same namespace):

   ```yaml
   apiVersion: operators.coreos.com/v1alpha1
   kind: Subscription
   metadata:
     name: rhods-operator
     namespace: redhat-ods-operator
   spec:
     channel: stable-3.4
     name: rhods-operator
     source: redhat-operators
     sourceNamespace: openshift-marketplace
   ```

2. Create a `DataScienceCluster` with at least:
   - `dashboard: Managed`
   - `trustyai: Managed` (EvalHub / Evaluations UI)
   - `mlflowoperator: Managed` (optional; this repo has standalone MLflow)
   - `kserve: Managed` (if serving models in-cluster)
   - `workbenches: Managed` (if using a Workbench)
3. Wait for `oc get dsc default-dsc -o jsonpath='{.status.phase}'` → `Ready`.

The repo's `scripts/00-preflight.sh` checks the DSC/EvalHub CRDs when
`REQUIRE_RHOAI=1`.

## Model endpoints

- **MaaS/LiteLLM** is the least-effort path: fill the six `*_API_BASE` /
  `*_API_KEY` pairs.
- **RHOAI KServe/vLLM**: create an OpenAI-compatible endpoint with
  `--enable-auto-tool-choice --tool-call-parser hermes`; set the agent's
  `*_API_BASE` to the route + `/v1`.
- **Raw vLLM**: `VLLM_ENABLED=1` in `.env`; the included Deployment uses a
  model-cache PVC and a single GPU.

## What the installer needs from you

Fill `.env` with:
- cluster access (`KUBECONFIG`, or `OCP_API_URL` + `OCP_TOKEN`)
- `STORAGE_CLASS`, namespaces
- model endpoints/keys
- `HARNESS_IMAGE` (after building/pushing)
- optional `AWS_*` values for documentation

The repo never stores AWS credentials, the pull secret, or the SSH key.
