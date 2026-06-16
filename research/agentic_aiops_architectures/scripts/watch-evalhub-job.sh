#!/usr/bin/env bash
# Poll EvalHub job status + show Running benchmark pod log tail (UI list is often empty).
set -euo pipefail

JOB_ID="${1:?Usage: $0 <job-uuid> [poll_seconds]}"
INTERVAL="${2:-30}"
TENANT_NS="${EVALHUB_TENANT:-rhods-notebooks}"
PLATFORM_NS="${EVALHUB_PLATFORM_NS:-redhat-ods-applications}"
HOST="${EVALHUB_URL:-https://$(oc get route evalhub -n "${PLATFORM_NS}" -o jsonpath='{.spec.host}')}"
TOKEN="$(oc whoami -t)"
PREFIX="${JOB_ID:0:8}"

while true; do
  echo "=== $(date -u +%H:%M:%S) job ${JOB_ID} ==="
  curl -skS "${HOST%/}/api/v1/evaluations/jobs/${JOB_ID}" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "X-Tenant: ${TENANT_NS}" -o /tmp/evalhub-job.json
  python3 - <<'PY'
import json
d = json.load(open("/tmp/evalhub-job.json"))
st = d.get("status") or {}
print("job:", d.get("name"), "→", st.get("state"))
msg = st.get("message") or {}
if isinstance(msg, dict):
    print("msg:", msg.get("message", "")[:120])
for b in st.get("benchmarks") or []:
    print(" ", b.get("id"), "→", b.get("status"))
PY
  oc get pods -n "${TENANT_NS}" 2>/dev/null | grep "${PREFIX}" | awk '{print "pod:", $1, $3, $2}' || true
  POD=$(oc get pods -n "${TENANT_NS}" 2>/dev/null | grep "${PREFIX}" | grep Running | head -1 | awk '{print $1}')
  if [[ -n "${POD}" ]]; then
    oc logs "${POD}" -n "${TENANT_NS}" -c adapter 2>/dev/null | grep -E 'Requesting API:|%' | tail -1 || true
  fi
  state=$(python3 -c "import json; print(json.load(open('/tmp/evalhub-job.json'))['status']['state'])")
  if [[ "${state}" == "completed" || "${state}" == "failed" || "${state}" == "cancelled" ]]; then
    break
  fi
  sleep "${INTERVAL}"
done
