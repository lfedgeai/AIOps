#!/usr/bin/env bash
# Fix OpenShift AI UI: "To use evaluations, enable the evaluation service using the TrustyAI Operator."
#
# RHOAI 3.4 eval-hub-ui checks GET /api/v1/evalhub/health for an EvalHub CR named "evalhub"
# in the dashboard/applications namespace. A tenant EvalHub only in the workbench
# namespace is not enough for that gate (health returns cr-not-found / available:false).
# Both namespaces are configurable (RHOAI_APPLICATIONS_NAMESPACE / EVALHUB_NAMESPACE).
#
# This script:
#   1) Labels the workbench namespace as an EvalHub tenant (operator RBAC for jobs).
#   2) Ensures the tenant EvalHub (EVALHUB_NAMESPACE) has spec.database (required on 3.4 GA).
#   3) Creates/updates the platform EvalHub in RHOAI_APPLICATIONS_NAMESPACE for the UI health check.
#
# Usage:
#   EVALHUB_NAMESPACE=rhods-notebooks RHOAI_APPLICATIONS_NAMESPACE=redhat-ods-applications ./scripts/enable-rhoai-evaluations-ui.sh
set -euo pipefail

. "$(dirname "$0")/lib.sh"
oc_login

TENANT_NS="${EVALHUB_NAMESPACE:-rhods-notebooks}"
PLATFORM_NS="${RHODS_DASHBOARD_NAMESPACE:-${RHOAI_APPLICATIONS_NAMESPACE:-${RHOAI_NAMESPACE:-redhat-ods-applications}}}"
MLFLOW_URI="${MLFLOW_URI:-http://mlflow.${AIOPS_NAMESPACE:-agentic-aiops}.svc.cluster.local:5000}"

echo "==> Ensure/label tenant namespace ${TENANT_NS}"
oc get namespace "${TENANT_NS}" >/dev/null 2>&1 || oc create namespace "${TENANT_NS}"
oc label namespace "${TENANT_NS}" evalhub.trustyai.opendatahub.io/tenant=true --overwrite

if oc get evalhub evalhub -n "${TENANT_NS}" &>/dev/null; then
  DB_TYPE="$(oc get evalhub evalhub -n "${TENANT_NS}" -o jsonpath='{.spec.database.type}' 2>/dev/null || true)"
  if [[ -z "${DB_TYPE}" ]]; then
    echo "==> Tenant EvalHub: set spec.database.type=sqlite"
    oc patch evalhub evalhub -n "${TENANT_NS}" --type=merge -p '{"spec":{"database":{"type":"sqlite"}}}'
  fi
else
  echo "WARN: no EvalHub CR in ${TENANT_NS}; create one for evaluation jobs (see EvalHub docs)."
fi

echo "==> Platform EvalHub in ${PLATFORM_NS} (UI health check)"
cat <<EOF | oc apply -f -
apiVersion: trustyai.opendatahub.io/v1alpha1
kind: EvalHub
metadata:
  name: evalhub
  namespace: ${PLATFORM_NS}
spec:
  replicas: 1
  database:
    type: sqlite
  env:
    - name: MLFLOW_TRACKING_URI
      value: ${MLFLOW_URI}
  providers:
    - garak
    - guidellm
    - lighteval
    - lm-evaluation-harness
EOF

echo "==> Wait for platform EvalHub Ready"
for _ in $(seq 1 60); do
  RDY="$(oc get evalhub evalhub -n "${PLATFORM_NS}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)"
  if [[ "${RDY}" == "True" ]]; then
    echo "EvalHub Ready in ${PLATFORM_NS}"
    break
  fi
  sleep 5
done

echo "==> Done. Hard-refresh the console → Develop & train → Evaluations."
