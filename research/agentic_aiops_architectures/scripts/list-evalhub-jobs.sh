#!/usr/bin/env bash
# List EvalHub jobs when the RHOAI Federated Evaluations UI shows an empty table
# (BFF calls EvalHub without X-Tenant). Uses the platform route + tenant header.
set -euo pipefail

TENANT_NS="${EVALHUB_TENANT:-rhods-notebooks}"
PLATFORM_NS="${EVALHUB_PLATFORM_NS:-redhat-ods-applications}"
HOST="${EVALHUB_URL:-https://$(oc get route evalhub -n "${PLATFORM_NS}" -o jsonpath='{.spec.host}')}"
TOKEN="$(oc whoami -t)"

JOB_ID="${1:-}"

if [[ -n "${JOB_ID}" ]]; then
  curl -skS "${HOST%/}/api/v1/evaluations/jobs/${JOB_ID}" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "X-Tenant: ${TENANT_NS}" | python3 -m json.tool
  exit 0
fi

TMP="$(mktemp)"
trap 'rm -f "${TMP}"' EXIT
curl -skS "${HOST%/}/api/v1/evaluations/jobs" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "X-Tenant: ${TENANT_NS}" -o "${TMP}"
python3 - "${TMP}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
items = d.get("items") or []
print(f"{'STATE':<12} {'JOB ID':<38} NAME")
print("-" * 90)
for it in items:
    rid = (it.get("resource") or {}).get("id", "")
    state = (it.get("status") or {}).get("state", "")
    name = (it.get("name") or "")[:40]
    print(f"{state:<12} {rid:<38} {name}")
print("-" * 90)
print(f"total: {d.get('total_count', len(items))}")
print()
print("Detail one job:  ./scripts/list-evalhub-jobs.sh <job-id>")
print("Watch:           ./scripts/watch-evalhub-job.sh <job-id>")
PY
