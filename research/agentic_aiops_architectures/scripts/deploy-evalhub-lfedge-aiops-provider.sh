#!/usr/bin/env bash
# Build the LF Edge AIOps EvalHub adapter image and register it as a system EvalHub provider.
#
# TrustyAI expects provider ConfigMaps in **redhat-ods-applications** with label
# trustyai.opendatahub.io/evalhub-provider-name=<name> and an ownerReference on the
# TrustyAI CR (default-trustyai). The EvalHub CR (tenant namespace) must list the
# provider under spec.providers (e.g. lfedge-aiops).
set -euo pipefail
TENANT_NS="${EVALHUB_NAMESPACE:-rhods-notebooks}"
OPERATOR_NS="${TRUSTYAI_OPERATOR_NAMESPACE:-redhat-ods-applications}"
BUILD_NAME="${LFEDGE_EVALHUB_BUILD:-lfedge-aiops-evalhub-adapter}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROVIDER_DIR="${ROOT}/evalhub/lfedge_aiops_provider"
CM_NAME="trustyai-service-operator-evalhub-provider-lfedge-aiops"

echo "==> OpenShift build (${BUILD_NAME}) in ${TENANT_NS}"
if ! oc get buildconfig "${BUILD_NAME}" -n "${TENANT_NS}" &>/dev/null; then
  oc new-build --binary --name="${BUILD_NAME}" --strategy=docker -n "${TENANT_NS}"
fi
oc start-build "${BUILD_NAME}" --from-dir="${PROVIDER_DIR}" -n "${TENANT_NS}" --follow --wait

IMAGE="$(oc get is "${BUILD_NAME}" -n "${TENANT_NS}" -o jsonpath='{.status.dockerImageRepository}'):latest"
echo "==> Built image: ${IMAGE}"

echo "==> Provider ConfigMap in ${OPERATOR_NS}"
sed "s#__IMAGE__#${IMAGE}#g" "${ROOT}/evalhub/manifests/trustyai-operator-evalhub-provider-lfedge-aiops.configmap.yaml" | oc apply -f -

echo "==> Owner reference (TrustyAI default-trustyai) so the operator reconciles this ConfigMap"
TRUSTY_UID="$(oc get trustyai.components.platform.opendatahub.io default-trustyai -n "${OPERATOR_NS}" -o jsonpath='{.metadata.uid}')"
oc patch configmap "${CM_NAME}" -n "${OPERATOR_NS}" --type=json -p "[{\"op\":\"add\",\"path\":\"/metadata/ownerReferences\",\"value\":[{\"apiVersion\":\"components.platform.opendatahub.io/v1alpha1\",\"blockOwnerDeletion\":true,\"controller\":true,\"kind\":\"TrustyAI\",\"name\":\"default-trustyai\",\"uid\":\"${TRUSTY_UID}\"}]}]" 2>/dev/null \
  || oc patch configmap "${CM_NAME}" -n "${OPERATOR_NS}" --type=json -p "[{\"op\":\"replace\",\"path\":\"/metadata/ownerReferences\",\"value\":[{\"apiVersion\":\"components.platform.opendatahub.io/v1alpha1\",\"blockOwnerDeletion\":true,\"controller\":true,\"kind\":\"TrustyAI\",\"name\":\"default-trustyai\",\"uid\":\"${TRUSTY_UID}\"}]}]"

echo "==> EvalHub CR: ensure spec.providers contains lfedge-aiops"
if oc get evalhub evalhub -n "${TENANT_NS}" -o jsonpath='{.spec.providers}' | grep -q 'lfedge-aiops'; then
  echo "provider lfedge-aiops already in spec.providers"
else
  oc patch evalhub evalhub -n "${TENANT_NS}" --type=json -p '[{"op":"add","path":"/spec/providers/-","value":"lfedge-aiops"}]'
fi

echo "==> Wait for EvalHub Ready (operator projects provider files)"
for _ in $(seq 1 60); do
  MSG=$(oc get evalhub evalhub -n "${TENANT_NS}" -o jsonpath='{.status.conditions[0].message}' 2>/dev/null || true)
  RDY=$(oc get evalhub evalhub -n "${TENANT_NS}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)
  if [[ "${RDY}" == "True" ]]; then
    echo "EvalHub Ready."
    break
  fi
  echo "  waiting… (${MSG})"
  sleep 5
done

echo "Done. Test: ${ROOT}/scripts/submit-evalhub-lfedge-aiops-job.sh"
