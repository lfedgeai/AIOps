#!/usr/bin/env bash
# Create/update the model-credentials Secret in the AIOps namespace from .env.
# The harness Job and any in-cluster agent run consume it via envFrom.
set -euo pipefail

. "$(dirname "$0")/lib.sh"
oc_login

"$(kube)" get namespace "$AIOPS_NAMESPACE" >/dev/null 2>&1 || \
  "$(kube)" create namespace "$AIOPS_NAMESPACE"

args=()
for key in \
  DEEPSEEK_API_BASE DEEPSEEK_API_KEY DEEPSEEK_MODEL \
  QWEN3_API_BASE QWEN3_API_KEY QWEN3_MODEL \
  QWEN38_API_BASE QWEN38_API_KEY QWEN38_MODEL \
  QWEN36_API_BASE QWEN36_API_KEY QWEN36_MODEL \
  GRANITE_API_BASE GRANITE_API_KEY GRANITE_MODEL \
  GLM_API_BASE GLM_API_KEY GLM_MODEL \
  JUDGE_API_BASE JUDGE_API_KEY JUDGE_MODEL \
  LFEDGE_MODEL_API_KEY HF_TOKEN; do
  val="${!key:-}"
  [[ -n "$val" ]] && args+=("--from-literal=${key}=${val}")
done

if [[ ${#args[@]} -eq 0 ]]; then
  die "no model credentials found in .env (set at least one *_API_BASE and *_API_KEY)"
fi

"$(kube)" create secret generic aiops-model-credentials \
  -n "$AIOPS_NAMESPACE" "${args[@]}" \
  --dry-run=client -o yaml | "$(kube)" apply -f -
log "secret ${AIOPS_NAMESPACE}/aiops-model-credentials updated (${#args[@]} keys)"
