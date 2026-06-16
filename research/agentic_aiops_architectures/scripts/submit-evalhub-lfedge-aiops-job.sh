#!/usr/bin/env bash
# Submit a one-benchmark EvalHub job using provider lfedge_aiops (requires deploy-evalhub-lfedge-aiops-provider.sh).
set -euo pipefail
NS="${EVALHUB_NAMESPACE:-rhods-notebooks}"
URL="${EVALHUB_URL:-https://$(oc get route evalhub -n "${NS}" -o jsonpath='{.spec.host}')}"
TOKEN="$(oc whoami -t)"

BODY=$(cat <<'JSON'
{
  "name": "lfedge-aiops-mttd-mttr-smoke",
  "description": "Smoke test for lfedge_aiops provider (ClickHouse-backed MTTD/MTTR heuristics)",
  "model": {
    "url": "https://example.com",
    "name": "noop"
  },
  "benchmarks": [
    {
      "id": "mttd_mttr_clickhouse",
      "provider_id": "lfedge_aiops",
      "parameters": {
        "window_minutes": 15
      }
    }
  ]
}
JSON
)

echo "==> POST ${URL}/api/v1/evaluations/jobs"
RESP=$(curl -skS -X POST "${URL}/api/v1/evaluations/jobs" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "X-Tenant: ${NS}" \
  -H "Content-Type: application/json" \
  -d "${BODY}")
echo "${RESP}" | python3 -m json.tool 2>/dev/null || echo "${RESP}"
JOB_ID=$(echo "${RESP}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('resource',{}).get('id') or d.get('id',''))" 2>/dev/null || true)
if [[ -z "${JOB_ID}" ]]; then
  echo "ERROR: could not parse job id from response" >&2
  exit 1
fi
echo "==> Job id: ${JOB_ID}"
echo "==> Waiting for terminal state (up to 600s)…"
for i in $(seq 1 120); do
  ST=$(curl -skS "${URL}/api/v1/evaluations/jobs/${JOB_ID}" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "X-Tenant: ${NS}")
  phase=$(echo "${ST}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status',{}).get('state',''))" 2>/dev/null || true)
  echo "  [${i}] phase=${phase}"
  if [[ "${phase}" == "completed" || "${phase}" == "failed" || "${phase}" == "cancelled" ]]; then
    echo "${ST}" | python3 -m json.tool 2>/dev/null || echo "${ST}"
    exit 0
  fi
  sleep 5
done
echo "Timed out waiting for job" >&2
exit 1
