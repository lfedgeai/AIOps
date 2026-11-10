#!/usr/bin/env bash
# Build the real AIOps harness EvalHub provider and register it as a system
# EvalHub provider.
#
# Prereqs:
#   - harness image already built/pushed (HARNESS_IMAGE in .env)
#   - RHOAI 3.4+ with trustyai Managed
#   - tenant namespace labeled and EvalHub CR present
#
# The TrustyAI operator reconciles provider ConfigMaps in the RHOAI platform
# namespace; the EvalHub CR must list `lfedge-aiops` in spec.providers.
set -euo pipefail

. "$(dirname "$0")/lib.sh"
oc_login

[[ "$(kube)" == "oc" ]] || die "the EvalHub provider build requires oc (OpenShift Builds)"
TENANT_NS="${EVALHUB_NAMESPACE:-rhods-notebooks}"
PLATFORM_NS="${RHOAI_APPLICATIONS_NAMESPACE:-${RHOAI_NAMESPACE:-redhat-ods-applications}}"
BUILD_NAME="${LFEDGE_EVALHUB_BUILD:-lfedge-aiops-evalhub-adapter}"
PROVIDER_DIR="$ROOT/evalhub/lfedge_aiops_provider"
CM_NAME="trustyai-service-operator-evalhub-provider-lfedge-aiops"

[[ "$HARNESS_IMAGE" != quay.io/your-org/* ]] || die "set HARNESS_IMAGE in .env first"
log "building provider image ${BUILD_NAME} in ${TENANT_NS}"
if ! "$(kube)" get buildconfig "$BUILD_NAME" -n "$TENANT_NS" >/dev/null 2>&1; then
  "$(kube)" new-build --binary --name="$BUILD_NAME" --strategy=docker -n "$TENANT_NS"
fi
# Feed the harness base image to the provider Dockerfile via the BuildConfig.
"$(kube)" patch buildconfig "$BUILD_NAME" -n "$TENANT_NS" --type=merge \
  -p "{\"spec\":{\"strategy\":{\"dockerStrategy\":{\"buildArgs\":[{\"name\":\"HARNESS_IMAGE\",\"value\":\"${HARNESS_IMAGE}\"}]}}}}" \
  >/dev/null
"$(kube)" start-build "$BUILD_NAME" --from-dir="$PROVIDER_DIR" -n "$TENANT_NS" --follow --wait

IMAGE="$("$(kube)" get is "$BUILD_NAME" -n "$TENANT_NS" -o jsonpath='{.status.dockerImageRepository}'):latest"
log "built provider image: $IMAGE"

log "applying provider ConfigMap to ${PLATFORM_NS}"
LFEDGE_EVALHUB_IMAGE="$IMAGE" \
RHOAI_APPLICATIONS_NAMESPACE="$PLATFORM_NS" \
CLICKHOUSE_HTTP="${CLICKHOUSE_HTTP}" \
OTEL_DEMO_NAMESPACE="${OTEL_DEMO_NAMESPACE}" \
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
LFEDGE_MODEL_API_KEY="${LFEDGE_MODEL_API_KEY:-}" \
  "$PYTHON_BIN" "$ROOT/scripts/render.py" \
  <"$ROOT/evalhub/manifests/trustyai-operator-evalhub-provider-lfedge-aiops.configmap.yaml" \
  | "$(kube)" apply -f -

log "setting ownerReference on ${CM_NAME} (TrustyAI default-trustyai)"
TRUSTY_UID="$("$(kube)" get trustyai.components.platform.opendatahub.io default-trustyai -n "$PLATFORM_NS" -o jsonpath='{.metadata.uid}' 2>/dev/null || true)"
if [[ -n "$TRUSTY_UID" ]]; then
  "$(kube)" patch configmap "$CM_NAME" -n "$PLATFORM_NS" --type=json \
    -p "[{\"op\":\"add\",\"path\":\"/metadata/ownerReferences\",\"value\":[{\"apiVersion\":\"components.platform.opendatahub.io/v1alpha1\",\"blockOwnerDeletion\":true,\"controller\":true,\"kind\":\"TrustyAI\",\"name\":\"default-trustyai\",\"uid\":\"${TRUSTY_UID}\"}]}]" 2>/dev/null \
    || "$(kube)" patch configmap "$CM_NAME" -n "$PLATFORM_NS" --type=json \
      -p "[{\"op\":\"replace\",\"path\":\"/metadata/ownerReferences\",\"value\":[{\"apiVersion\":\"components.platform.opendatahub.io/v1alpha1\",\"blockOwnerDeletion\":true,\"controller\":true,\"kind\":\"TrustyAI\",\"name\":\"default-trustyai\",\"uid\":\"${TRUSTY_UID}\"}]}]"
else
  warn "TrustyAI CR not found; provider ConfigMap may not reconcile"
fi

log "ensuring lfedge-aiops is in EvalHub spec.providers"
if "$(kube)" get evalhub evalhub -n "$TENANT_NS" -o jsonpath='{.spec.providers}' 2>/dev/null | grep -q 'lfedge-aiops'; then
  log "provider lfedge-aiops already present"
else
  "$(kube)" patch evalhub evalhub -n "$TENANT_NS" --type=json \
    -p '[{"op":"add","path":"/spec/providers/-","value":"lfedge-aiops"}]' 2>/dev/null || \
    warn "could not patch EvalHub spec.providers (does the EvalHub CR exist?)"
fi

log "done. Submit a smoke job with: $ROOT/scripts/submit-evalhub-lfedge-aiops-job.sh"
