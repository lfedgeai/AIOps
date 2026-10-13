#!/usr/bin/env bash
# Bootstrap an EvalHub tenant namespace (EvalHub CR, MaaS secrets, tenant label).
set -euo pipefail

TENANT_NS="${1:?usage: $0 <namespace> [source-namespace]}"
SRC_NS="${2:-rhods-notebooks}"
PLATFORM_NS="${EVALHUB_PLATFORM_NS:-redhat-ods-applications}"
MLFLOW_URI="${MLFLOW_URI:-https://mlflow.${PLATFORM_NS}.svc.cluster.local:8443}"
MAAS_SECRETS="${MAAS_SECRETS:-maas-litellm-deepseek maas-litellm-llama-scout maas-litellm-qwen3}"

echo "==> Namespace ${TENANT_NS}"
oc create namespace "${TENANT_NS}" 2>/dev/null || true
oc label namespace "${TENANT_NS}" evalhub.trustyai.opendatahub.io/tenant=true --overwrite

echo "==> Copy MaaS secrets from ${SRC_NS}"
for s in ${MAAS_SECRETS}; do
  if ! oc get secret "${s}" -n "${SRC_NS}" &>/dev/null; then
    echo "    skip (missing in ${SRC_NS}): ${s}"
    continue
  fi
  oc get secret "${s}" -n "${SRC_NS}" -o json | python3 -c "
import json, sys
d = json.load(sys.stdin)
meta = d['metadata']
for k in ('resourceVersion', 'uid', 'creationTimestamp', 'managedFields', 'ownerReferences'):
    meta.pop(k, None)
meta['namespace'] = '${TENANT_NS}'
print(json.dumps(d))
" | oc apply -f -
  if ! oc get secret "${s}" -n "${TENANT_NS}" -o jsonpath='{.data.api-key}' 2>/dev/null | grep -q .; then
    if oc get secret "${s}" -n "${TENANT_NS}" -o jsonpath='{.data.token}' 2>/dev/null | grep -q .; then
      TB="$(oc get secret "${s}" -n "${TENANT_NS}" -o jsonpath='{.data.token}')"
      oc patch secret "${s}" -n "${TENANT_NS}" --type=merge -p "{\"data\":{\"api-key\":\"${TB}\"}}"
    fi
  fi
  echo "    ${s}"
done

if oc get evalhub evalhub -n "${TENANT_NS}" &>/dev/null; then
  echo "==> EvalHub CR already exists in ${TENANT_NS}"
else
  echo "==> EvalHub CR in ${TENANT_NS}"
  cat <<EOF | oc apply -f -
apiVersion: trustyai.opendatahub.io/v1alpha1
kind: EvalHub
metadata:
  name: evalhub
  namespace: ${TENANT_NS}
spec:
  replicas: 1
  database:
    type: sqlite
  env:
    - name: MLFLOW_TRACKING_URI
      value: ${MLFLOW_URI}
  collections:
    - leaderboard-v2
    - safety-and-fairness-v1
    - toxicity-and-ethical-principles
  providers:
    - garak
    - guidellm
    - lighteval
    - lm-evaluation-harness
EOF
fi

echo "==> Wait for EvalHub deployment"
for _ in $(seq 1 90); do
  if oc get deploy evalhub -n "${TENANT_NS}" &>/dev/null; then
    oc rollout status deploy/evalhub -n "${TENANT_NS}" --timeout=30s 2>/dev/null && break
  fi
  sleep 5
done

echo "==> Tenant ready: ${TENANT_NS}"
oc get evalhub evalhub -n "${TENANT_NS}" -o jsonpath='{.status.url}{"\n"}' 2>/dev/null || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "${ROOT}/scripts/fix-evalhub-tenant-rbac.sh" ]]; then
  echo "==> Platform EvalHub RBAC for tenant jobs (operator cross-wire workaround)"
  "${ROOT}/scripts/fix-evalhub-tenant-rbac.sh" "${TENANT_NS}" || true
fi
