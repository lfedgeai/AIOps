#!/usr/bin/env bash
# Expose TrustyAI vLLM gateway sidecar (127.0.0.1:8090) on the orchestrator Service.
#
# The GuardrailsOrchestrator controller creates Route targetPort=gateway but omits
# the Service port when the gateway binds localhost only. This script patches
# the Deployment (HOST=0.0.0.0, containerPort 8090) and Service (port gateway).
#
# Usage: ./scripts/patch-trustyai-gateway-exposure.sh [apply|status]
set -euo pipefail

GUARDRAILS_NS="${GUARDRAILS_NS:-netobserv-guardrails}"
DEPLOY="${DEPLOY:-guardrails-orchestrator}"
SVC="${SVC:-guardrails-orchestrator-service}"
GATEWAY_PORT="${GATEWAY_PORT:-8090}"
KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-apply}"

c_green=$'\033[1;32m'; c_blue=$'\033[1;34m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }

[[ -n "$KUBECTL" ]] || { echo "oc/kubectl required" >&2; exit 1; }

cmd_status() {
  step "Service ${SVC}"
  "$KUBECTL" -n "$GUARDRAILS_NS" get svc "$SVC" -o jsonpath='{range .spec.ports[*]}{.name}:{.port}{"\n"}{end}' 2>/dev/null || true
  step "Gateway container env"
  "$KUBECTL" -n "$GUARDRAILS_NS" get deploy "$DEPLOY" \
    -o jsonpath='{range .spec.template.spec.containers[?(@.name=="guardrails-orchestrator-gateway")].env[*]}{.name}={.value}{"\n"}{end}' 2>/dev/null || true
}

patch_deployment() {
  step "Patch ${DEPLOY} gateway sidecar → listen 0.0.0.0:${GATEWAY_PORT}"
  # Preserve GATEWAY_CONFIG / GATEWAY_USE_MTLS — do not replace the whole env array.
  "$KUBECTL" -n "$GUARDRAILS_NS" set env deploy/"$DEPLOY" \
    -c guardrails-orchestrator-gateway \
    "HOST=0.0.0.0" \
    "GATEWAY_CONFIG=/config/config.yaml" \
    "GATEWAY_USE_MTLS=false" \
    "RUST_LOG=info"
  "$KUBECTL" -n "$GUARDRAILS_NS" patch deploy "$DEPLOY" --type=strategic -p "$(cat <<EOF
spec:
  template:
    spec:
      containers:
        - name: guardrails-orchestrator-gateway
          ports:
            - containerPort: ${GATEWAY_PORT}
              name: gateway
              protocol: TCP
EOF
)"
  ok "Deployment patched"
}

patch_service() {
  step "Add Service port gateway:${GATEWAY_PORT}"
  if "$KUBECTL" -n "$GUARDRAILS_NS" get svc "$SVC" -o jsonpath='{.spec.ports[*].name}' | grep -q '\bgateway\b'; then
    ok "Service already exposes gateway port"
    return 0
  fi
  "$KUBECTL" -n "$GUARDRAILS_NS" patch svc "$SVC" --type=json -p "[{\"op\":\"add\",\"path\":\"/spec/ports/-\",\"value\":{\"name\":\"gateway\",\"port\":${GATEWAY_PORT},\"protocol\":\"TCP\",\"targetPort\":${GATEWAY_PORT}}}]"
  ok "Service patched"
}

wait_rollout() {
  step "Wait for orchestrator rollout"
  "$KUBECTL" -n "$GUARDRAILS_NS" rollout status deploy/"$DEPLOY" --timeout=300s
}

case "$CMD" in
  apply|"") patch_deployment; patch_service; wait_rollout ;;
  status) cmd_status ;;
  *) echo "usage: $0 [apply|status]" >&2; exit 1 ;;
esac
