#!/usr/bin/env bash
# Optional: deploy the raw vLLM fallback endpoint.
# For RHOAI-native serving, use KServe/vLLM or MaaS instead and skip this.
set -euo pipefail

. "$(dirname "$0")/lib.sh"

if [[ "${VLLM_ENABLED:-0}" != "1" && "${VLLM_ENABLED:-0}" != "true" ]]; then
  log "VLLM_ENABLED is not set; skipping vLLM"
  exit 0
fi

oc_login
apply_manifest "$ROOT/manifests/vllm.yaml"
log "waiting for vLLM (model download can take several minutes)"
"$(kube)" rollout status deployment/vllm -n "$AIOPS_NAMESPACE" --timeout=1800s || \
  warn "vLLM not ready yet; check GPU resources and pod logs"

log "vLLM in-cluster: http://vllm.${AIOPS_NAMESPACE}.svc.cluster.local:8000/v1"
if command -v oc >/dev/null 2>&1; then
  host="$(oc get route vllm -n "$AIOPS_NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  [[ -n "$host" ]] && log "vLLM Route:      https://${host}/v1"
fi
