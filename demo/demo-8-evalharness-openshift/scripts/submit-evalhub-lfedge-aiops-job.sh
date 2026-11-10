#!/usr/bin/env bash
# Submit one real AIOps harness cell to EvalHub.
# Requires: deploy-evalhub-lfedge-aiops-provider.sh + tenant EvalHub.
#
# Usage:
#   ./scripts/submit-evalhub-lfedge-aiops-job.sh                          # defaults
#   EVALHUB_AGENT=rhdp_granite-3-2-8b-instruct ./scripts/submit-...      # select agent
#   EVALHUB_CONTEXT=1,4 ./scripts/submit-...                             # select context
#   EVALHUB_FAULT=kill_pod EVALHUB_SCENARIO=b ./scripts/submit-...       # select fault/scenario
#   EVALHUB_JUDGE=true ./scripts/submit-...                              # enable judge
set -euo pipefail

. "$(dirname "$0")/lib.sh"

[[ "$(kube)" == "oc" ]] || die "EvalHub submission requires oc"
NS="${EVALHUB_NAMESPACE:-rhods-notebooks}"
PLATFORM_NS="${EVALHUB_PLATFORM_NAMESPACE:-${RHOAI_APPLICATIONS_NAMESPACE:-redhat-ods-applications}}"
if [[ -z "${EVALHUB_URL:-}" ]]; then
  EVALHUB_URL="https://$("$(kube)" get route evalhub -n "$PLATFORM_NS" -o jsonpath='{.spec.host}' 2>/dev/null || true)"
fi
[[ "$EVALHUB_URL" == https:// ]] && die "could not find EvalHub route in $PLATFORM_NS; set EVALHUB_URL"
TOKEN="$("$(kube)" whoami -t)"

AGENT="${EVALHUB_AGENT:-rhdp_deepseek-r1-distill-qwen-14b}"
SCENARIO="${EVALHUB_SCENARIO:-a}"
FAULT="${EVALHUB_FAULT:-scale_zero}"
VARIANT="${EVALHUB_VARIANT:-cart}"
CONTEXT="${EVALHUB_CONTEXT:-0}"

case "$AGENT" in
  rhdp_deepseek-r1-distill-qwen-14b) _base="${DEEPSEEK_API_BASE:-}"; _model="${DEEPSEEK_MODEL:-deepseek-r1-distill-qwen-14b}" ;;
  litemaas_qwen3.8-27b)              _base="${QWEN36_API_BASE:-}";   _model="${QWEN36_MODEL:-Qwen3.8-27B}" ;;
  rhdp_granite-3-2-8b-instruct)      _base="${GRANITE_API_BASE:-}";  _model="${GRANITE_MODEL:-granite-3-2-8b-instruct}" ;;
  *) die "unknown EVALHUB_AGENT: $AGENT" ;;
esac
MODEL_URL="${EVALHUB_MODEL_URL:-$_base}"
MODEL_NAME="${EVALHUB_MODEL_NAME:-$_model}"
JUDGE="${EVALHUB_JUDGE:-false}"

BODY=$(cat <<JSON
{
  "name": "lfedge-aiops-harness-smoke",
  "description": "One real AIOps harness cell (MTTD/MTTR/RCA)",
  "model": {
    "url": "${MODEL_URL:-https://example.com}",
    "name": "${MODEL_NAME}"
  },
  "benchmarks": [
    {
      "id": "mttd_mttr_harness",
      "provider_id": "lfedge_aiops",
      "parameters": {
        "agent": "${AGENT}",
        "scenario": "${SCENARIO}",
        "fault": "${FAULT}",
        "variant": "${VARIANT}",
        "context_c": "${CONTEXT}",
        "judge": ${JUDGE},
        "run_timeout": 1800
      }
    }
  ]
}
JSON
)

log "POST ${EVALHUB_URL}/api/v1/evaluations/jobs"
RESP=$(curl -skS -X POST "${EVALHUB_URL%/}/api/v1/evaluations/jobs" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "X-Tenant: ${NS}" \
  -H "Content-Type: application/json" \
  -d "${BODY}")
echo "${RESP}" | python3 -m json.tool 2>/dev/null || echo "${RESP}"
JOB_ID=$(echo "${RESP}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('resource',{}).get('id') or d.get('id',''))" 2>/dev/null || true)
[[ -n "$JOB_ID" ]] || die "could not parse job id from response"

log "job id: $JOB_ID"
for i in $(seq 1 180); do
  ST=$(curl -skS "${EVALHUB_URL%/}/api/v1/evaluations/jobs/${JOB_ID}" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "X-Tenant: ${NS}")
  phase=$(echo "${ST}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status',{}).get('state',''))" 2>/dev/null || true)
  echo "  [${i}] phase=${phase}"
  if [[ "$phase" == "completed" || "$phase" == "failed" || "$phase" == "cancelled" ]]; then
    echo "${ST}" | python3 -m json.tool 2>/dev/null || echo "${ST}"
    exit 0
  fi
  sleep 5
done
die "timed out waiting for EvalHub job"
