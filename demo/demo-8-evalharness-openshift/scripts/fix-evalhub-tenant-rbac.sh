#!/usr/bin/env bash
# EvalHub 3.4 operator can cross-wire tenant job RoleBindings when multiple EvalHub CRs
# exist. The platform API schedules jobs as <platform-ns>:evalhub-service.
# Patching operator-owned RoleBindings is reverted on reconcile — we add fix bindings instead.
set -euo pipefail

. "$(dirname "$0")/lib.sh"
oc_login

PLATFORM_NS="${EVALHUB_PLATFORM_NAMESPACE:-${EVALHUB_PLATFORM_NS:-${RHOAI_APPLICATIONS_NAMESPACE:-redhat-ods-applications}}}"
SA="${EVALHUB_PLATFORM_SA:-evalhub-service}"
TENANTS="${*:-${EVALHUB_NAMESPACE:-rhods-notebooks}}"

ensure_binding() {
  local tenant="$1" suffix="$2" cluster_role="$3"
  local name="evalhub-platform-${suffix}-fix"
  oc apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: ${name}
  namespace: ${tenant}
  labels:
    app: eval-hub
    app.kubernetes.io/component: evalhub-rbac-fix
    app.kubernetes.io/part-of: eval-hub
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: ${cluster_role}
subjects:
- kind: ServiceAccount
  name: ${SA}
  namespace: ${PLATFORM_NS}
EOF
  echo "    applied ${tenant}/${name} -> ${cluster_role}"
}

for tenant in ${TENANTS}; do
  echo "==> Tenant namespace ${tenant}"
  ensure_binding "${tenant}" "job-config" "trustyai-service-operator-evalhub-job-config"
  ensure_binding "${tenant}" "job-writer" "trustyai-service-operator-evalhub-jobs-writer"
done

echo "==> Verify platform SA can create ConfigMaps in tenant namespaces"
for tenant in ${TENANTS}; do
  if oc auth can-i create configmaps -n "${tenant}" \
    --as="system:serviceaccount:${PLATFORM_NS}:${SA}" 2>/dev/null | grep -q yes; then
    echo "    yes  ${PLATFORM_NS}/${SA} -> configmaps in ${tenant}"
  else
    echo "    NO   ${PLATFORM_NS}/${SA} -> configmaps in ${tenant}" >&2
    exit 1
  fi
done
