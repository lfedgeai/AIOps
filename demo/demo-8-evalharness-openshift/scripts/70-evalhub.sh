#!/usr/bin/env bash
# Optional RHOAI EvalHub wiring: platform EvalHub + real harness provider.
# Requires RHOAI 3.4+ with the trustyai component Managed.
set -euo pipefail

. "$(dirname "$0")/lib.sh"
oc_login

export EVALHUB_NAMESPACE="${EVALHUB_NAMESPACE:-rhods-notebooks}"
export EVALHUB_PLATFORM_NAMESPACE="${EVALHUB_PLATFORM_NAMESPACE:-$RHOAI_APPLICATIONS_NAMESPACE}"
export RHOAI_NAMESPACE="$RHOAI_APPLICATIONS_NAMESPACE"
# Point EvalHub at the standalone MLflow unless the user overrides it.
export MLFLOW_URI="${MLFLOW_URI:-$MLFLOW_TRACKING_URI}"

if ! "$(kube)" get crd evalhubs.trustyai.opendatahub.io >/dev/null 2>&1; then
  die "EvalHub CRD not found. Set trustyai.managementState=Managed in the DataScienceCluster and wait for the operator."
fi

log "bootstrapping tenant namespace ${EVALHUB_NAMESPACE} (if needed)"
if ! "$(kube)" get evalhub evalhub -n "$EVALHUB_NAMESPACE" >/dev/null 2>&1; then
  "$ROOT/scripts/bootstrap-evalhub-tenant-namespace.sh" "$EVALHUB_NAMESPACE"
fi

log "enabling the RHOAI Evaluations UI platform EvalHub"
"$ROOT/scripts/enable-rhoai-evaluations-ui.sh"

log "building/registering the real harness provider"
"$ROOT/scripts/deploy-evalhub-lfedge-aiops-provider.sh"

log "fixing platform EvalHub job RBAC in tenant namespaces"
"$ROOT/scripts/fix-evalhub-tenant-rbac.sh" "$EVALHUB_NAMESPACE" || true

cat <<EOF
EvalHub wiring complete.

Important: the EvalHub job pod needs write access to ${OTEL_DEMO_NAMESPACE}
for fault injection. Bind the harness Role to the EvalHub job service account:

  oc create rolebinding aiops-evalhub-harness -n ${OTEL_DEMO_NAMESPACE} \\
    --role=aiops-harness-faults \\
    --serviceaccount=<evalhub-job-namespace>:<evalhub-job-sa>

Then submit a smoke job:
  EVALHUB_NAMESPACE=${EVALHUB_NAMESPACE} ./scripts/submit-evalhub-lfedge-aiops-job.sh
EOF
