#!/usr/bin/env bash
# Inject LF Edge AIOps stack connection hints into the EvalHub CR (rhods-notebooks).
# EvalHub ignores unknown env vars today, but BYOF / future adapters can read these.
# Safe to re-run: skips vars that are already present.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_NS="${EVALHUB_NAMESPACE:-rhods-notebooks}"
OTEL_NS="${OTEL_DEMO_NAMESPACE:-otel-demo}"

FLAGD_HOST="$(oc get route flagd-ui-api -n "${OTEL_NS}" -o jsonpath='{.spec.host}' 2>/dev/null || true)"
MLF_AGENTIC="$(oc get route mlflow -n agentic-aiops -o jsonpath='{.spec.host}' 2>/dev/null || true)"

if [[ -z "${FLAGD_HOST}" ]]; then
  echo "No route flagd-ui-api in ${OTEL_NS}; apply manifests/flagd-ui-route.yaml first." >&2
  exit 1
fi

# RHOAI 3.4+ EvalHub operator requires explicit database (no default sqlite).
DB_TYPE="$(oc get evalhub evalhub -n "${EVAL_NS}" -o jsonpath='{.spec.database.type}' 2>/dev/null || true)"
if [[ -z "${DB_TYPE}" ]]; then
  echo "==> Setting spec.database.type=sqlite (required on RHOAI 3.4 GA)"
  oc patch evalhub evalhub -n "${EVAL_NS}" --type=merge -p '{"spec":{"database":{"type":"sqlite"}}}'
else
  echo "database already set: ${DB_TYPE}"
fi

CH_HTTP="http://clickhouse.${OTEL_NS}.svc.cluster.local:8123"

have_env() {
  local key="$1"
  oc get evalhub evalhub -n "${EVAL_NS}" -o jsonpath='{range .spec.env[*]}{.name}{"\n"}{end}' 2>/dev/null | grep -qx "${key}"
}

add_env() {
  local name="$1" val="$2"
  if have_env "${name}"; then
    echo "skip (exists): ${name}"
    return
  fi
  oc patch evalhub evalhub -n "${EVAL_NS}" --type=json -p "[{\"op\":\"add\",\"path\":\"/spec/env/-\",\"value\":{\"name\":\"${name}\",\"value\":\"${val}\"}}]"
  echo "added: ${name}"
}

add_env LFEDGE_AIOPS_FLAGD_READ_URL "https://${FLAGD_HOST}/api/read"
add_env LFEDGE_AIOPS_FLAGD_WRITE_URL "https://${FLAGD_HOST}/api/write"
add_env LFEDGE_AIOPS_CLICKHOUSE_HTTP "${CH_HTTP}"
if [[ -n "${MLF_AGENTIC}" ]]; then
  add_env LFEDGE_AIOPS_MLFLOW_URI "https://${MLF_AGENTIC}"
fi
add_env LFEDGE_AIOPS_OTEL_NAMESPACE "${OTEL_NS}"

echo "Rollout EvalHub to pick up env (if operator propagates to deployment)..."
oc rollout restart deployment/evalhub -n "${EVAL_NS}" 2>/dev/null || true
echo "Done."
