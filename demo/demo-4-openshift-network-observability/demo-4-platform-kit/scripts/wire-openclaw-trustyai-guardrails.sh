#!/usr/bin/env bash
# Wire OpenClaw LLM provider → TrustyAI Guardrails Gateway (Layer 0).
# Removes legacy netobserv-input-guard plugin sidecar path.
#
# Usage:
#   ./scripts/wire-openclaw-trustyai-guardrails.sh all
#   ./scripts/wire-openclaw-trustyai-guardrails.sh status
#   ./scripts/wire-openclaw-trustyai-guardrails.sh revert   # direct LiteMaaS again
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
GUARDRAILS_NS="${GUARDRAILS_NS:-netobserv-guardrails}"
GATEWAY_PRESET="${GATEWAY_PRESET:-netobserv-sre}"
KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-all}"

c_green=$'\033[1;32m'; c_blue=$'\033[1;34m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }

[[ -n "$KUBECTL" ]] || { echo "oc/kubectl required" >&2; exit 1; }

gateway_svc() {
  local svc
  svc="$("$KUBECTL" -n "$GUARDRAILS_NS" get svc guardrails-orchestrator-service \
    -o jsonpath='{.metadata.name}' 2>/dev/null || true)"
  [[ -n "$svc" ]] || svc="guardrails-orchestrator-service"
  printf '%s' "$svc"
}

gateway_base_url() {
  echo "http://netobserv-llm-guard-proxy.${GUARDRAILS_NS}.svc.cluster.local:8080/${GATEWAY_PRESET}/v1"
}

verify_gateway_llm_path() {
  local gw_url="$1"
  local key
  step "Preflight: guard proxy /models + /chat/completions (${gw_url})"
  key="$("$KUBECTL" -n "$OPENCLAW_NS" get secret my-llm-key -o jsonpath='{.data.api-key}' 2>/dev/null | base64 -d || true)"
  [[ -n "$key" ]] || { warn "my-llm-key missing in ${OPENCLAW_NS}"; return 1; }
  local models_code chat_code
  models_code="$("$KUBECTL" -n "$OPENCLAW_NS" exec deploy/openclaw -c openclaw -- \
    curl -sS -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer ${key}" "${gw_url%/}/models" 2>/dev/null || echo 000)"
  chat_code="$("$KUBECTL" -n "$OPENCLAW_NS" exec deploy/openclaw -c openclaw -- \
    curl -sS -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer ${key}" -H "Content-Type: application/json" \
    -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"system","content":"You are an SRE"},{"role":"user","content":"say ok"}],"tools":[{"type":"function","function":{"name":"probe","description":"probe","parameters":{"type":"object","properties":{}}}}],"max_tokens":3,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
    "${gw_url%/}/chat/completions" 2>/dev/null || echo 000)"
  if [[ "$models_code" != "200" ]]; then
    warn "/models returned HTTP ${models_code} (expected 200)"
    return 1
  fi
  if [[ "$chat_code" != "200" ]]; then
    warn "/chat/completions returned HTTP ${chat_code} (expected 200)"
    return 1
  fi
  ok "Guard proxy LLM path OK (/models=${models_code}, chat=${chat_code})"
  return 0
}

teardown_custom_guard() {
  step "Remove legacy netobserv-input-guard (custom plugin path)"
  "$ROOT/scripts/teardown-custom-input-guard.sh" 2>/dev/null || true
}

patch_openclaw_config() {
  local gw_url="${1:?gateway baseUrl}"
  step "Patch openclaw-config → TrustyAI gateway (${gw_url})"
  local raw
  raw="$("$KUBECTL" -n "$OPENCLAW_NS" get configmap openclaw-config -o jsonpath='{.data.openclaw\.json}')"
  GW_URL="$gw_url" GATEWAY_PRESET="$GATEWAY_PRESET" GUARDRAILS_NS="$GUARDRAILS_NS" RAW="$raw" python3 - <<'PY' \
    | "$KUBECTL" -n "$OPENCLAW_NS" patch configmap openclaw-config --type merge -p "$(cat)"
import json, os
d = json.loads(os.environ["RAW"])
gw = os.environ["GW_URL"].rstrip("/")
prov = d.setdefault("models", {}).setdefault("providers", {}).setdefault("openai", {})
prov["baseUrl"] = gw
# Primary model id is openai/Qwen3.6-35B-A3B; thinking mode crashes TrustyAI gateway.
models = d.setdefault("agents", {}).setdefault("defaults", {}).setdefault("models", {})
for key in ("openai/Qwen3.6-35B-A3B", "Qwen3.6-35B-A3B"):
    m = models.setdefault(key, {})
    params = m.setdefault("params", {})
    extra = params.setdefault("extra_body", {})
    kwargs = extra.setdefault("chat_template_kwargs", {})
    kwargs["enable_thinking"] = False
plugins = d.setdefault("plugins", {})
allow = plugins.setdefault("allow", [])
if isinstance(allow, list):
    plugins["allow"] = [x for x in allow if x != "netobserv-input-guard"]
entries = plugins.setdefault("entries", {})
if isinstance(entries, dict):
    entries.pop("netobserv-input-guard", None)
# OpenClaw rejects unknown keys under meta; keep config schema-valid.
d.pop("meta", None)
plugins.setdefault("bundledDiscovery", "compat")
print(json.dumps({"data": {"openclaw.json": json.dumps(d, indent=2)}}))
PY
  "$KUBECTL" -n "$OPENCLAW_NS" annotate configmap openclaw-config \
    "netobserv.demo/trustyai-gateway-preset=${GATEWAY_PRESET}" \
    "netobserv.demo/trustyai-namespace=${GUARDRAILS_NS}" \
    --overwrite >/dev/null 2>&1 || true
  ok "openclaw.json baseUrl → ${gw_url}"
}

RECYCLE_POD="${RECYCLE_POD:-1}"

recycle_openclaw() {
  if [[ "$RECYCLE_POD" != "1" ]]; then
    ok "Skipped OpenClaw recycle (RECYCLE_POD=0)"
    return 0
  fi
  step "Recycle OpenClaw pod"
  "$KUBECTL" -n "$OPENCLAW_NS" rollout restart deployment/openclaw 2>/dev/null \
    || "$KUBECTL" -n "$OPENCLAW_NS" delete pod -l app=openclaw --ignore-not-found
  "$KUBECTL" -n "$OPENCLAW_NS" rollout status deployment/openclaw --timeout=300s 2>/dev/null || true
  ok "OpenClaw recycled"
}

apply_guard_proxy() {
  step "Apply netobserv-llm-guard-proxy ConfigMap + Deployment"
  "$KUBECTL" apply -f "$ROOT/manifests/trustyai-guardrails/03-llm-guard-proxy.yaml"
  if "$KUBECTL" -n "$GUARDRAILS_NS" get deploy/netobserv-llm-guard-proxy >/dev/null 2>&1; then
    "$KUBECTL" -n "$GUARDRAILS_NS" rollout restart deploy/netobserv-llm-guard-proxy
    "$KUBECTL" -n "$GUARDRAILS_NS" rollout status deploy/netobserv-llm-guard-proxy --timeout=420s
  fi
  ok "Guard proxy manifest applied (allow ~90s for pip install on cold start)"
}

cmd_status() {
  local gw
  gw="$(gateway_base_url)"
  step "TrustyAI gateway URL: ${gw}"
  "$KUBECTL" -n "$OPENCLAW_NS" get configmap openclaw-config -o jsonpath='{.data.openclaw\.json}' 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); p=d.get("models",{}).get("providers",{}).get("openai",{}); print("OpenClaw baseUrl:", p.get("baseUrl"))' \
    || warn "openclaw-config missing"
}

cmd_revert() {
  local direct="${LLM_BASE_URL:-https://litemaas.example.com/v1}"
  patch_openclaw_config "$direct"
  recycle_openclaw
  ok "Reverted to direct LLM: ${direct}"
}

case "$CMD" in
  all|wire)
    teardown_custom_guard
    apply_guard_proxy
    gw="$(gateway_base_url)"
    if ! "$KUBECTL" -n "$GUARDRAILS_NS" get guardrailsorchestrator guardrails-orchestrator >/dev/null 2>&1; then
      warn "GuardrailsOrchestrator missing — run: ./scripts/install-trustyai-guardrails.sh install"
      exit 1
    fi
    verify_gateway_llm_path "$gw" || exit 1
    patch_openclaw_config "$gw"
    recycle_openclaw
    ok "Layer 0 = TrustyAI gateway preset ${GATEWAY_PRESET}"
    ;;
  status) cmd_status ;;
  revert) cmd_revert ;;
  *)
    echo "usage: $0 [all|status|revert]" >&2
    exit 1
    ;;
esac
