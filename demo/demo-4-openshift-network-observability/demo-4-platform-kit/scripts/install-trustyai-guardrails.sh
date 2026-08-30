#!/usr/bin/env bash
# Install TrustyAI Guardrails (FMS Orchestrator) for NetObserv OpenClaw demo.
# OpenShift AI 3.4 — enables trustyai + kserve in DSC, then upstream
# rh-ai-quickstart/lemonade-stand-assistant (fms-orchestrator/chart, Option A MaaS).
#
# Upstream: https://github.com/rh-ai-quickstart/lemonade-stand-assistant
#
# Usage:
#   ./scripts/install-trustyai-guardrails.sh              # install + enable gateway
#   ./scripts/install-trustyai-guardrails.sh status
#   ./scripts/install-trustyai-guardrails.sh uninstall
#
# Env:
#   GUARDRAILS_NS=netobserv-guardrails
#   LEMONADE_REPO / LEMONADE_BRANCH / LEMONADE_COMMIT — see scripts/supply-chain-pins.env
#   LLM_BASE_URL=https://litemaas.example.com/v1
#   SKIP_HELM=1          skip helm (manifests only — gateway config + patch)
#   SKIP_LEMONADE_STAND=1   default 1 — remove upstream workshop UI (not used by OpenClaw)
#   LIGHT_DETECTORS=1    reduce prompt-injection memory for small clusters
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=supply-chain-pins.env
source "$ROOT/scripts/supply-chain-pins.env"

GUARDRAILS_NS="${GUARDRAILS_NS:-netobserv-guardrails}"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
HELM_RELEASE="${HELM_RELEASE:-netobserv-trustyai-guardrails}"
SKIP_LEMONADE_STAND="${SKIP_LEMONADE_STAND:-1}"
GATEWAY_PRESET="${GATEWAY_PRESET:-netobserv-sre}"
KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-install}"

c_green=$'\033[1;32m'; c_blue=$'\033[1;34m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }

[[ -n "$KUBECTL" ]] || { echo "oc/kubectl required" >&2; exit 1; }

resolve_llm_host_port() {
  local base="${LLM_BASE_URL:-}"
  if [[ -z "$base" ]]; then
    base="$("$KUBECTL" -n "$OPENCLAW_NS" get configmap openclaw-config -o jsonpath='{.data.openclaw\.json}' 2>/dev/null \
      | python3 -c 'import json,sys; print(json.load(sys.stdin).get("models",{}).get("providers",{}).get("openai",{}).get("baseUrl",""))' 2>/dev/null || true)"
  fi
  [[ -n "$base" ]] || base="https://litemaas.example.com/v1"
  python3 - <<PY
from urllib.parse import urlparse
u = urlparse("${base}".rstrip("/"))
host = u.hostname or "litemaas.example.com"
port = u.port or (443 if u.scheme == "https" else 80)
print(f"{host} {port}")
PY
}

check_trustyai_crd() {
  if ! "$KUBECTL" get crd guardrailsorchestrators.trustyai.opendatahub.io >/dev/null 2>&1; then
    return 1
  fi
  ok "TrustyAI GuardrailsOrchestrator CRD present"
}

enable_dsc_trustyai() {
  step "Enable TrustyAI + KServe in DataScienceCluster (Guardrails prerequisite)"
  "$KUBECTL" patch dsc default-dsc --type merge -p '{
    "spec": {
      "components": {
        "kserve": {
          "managementState": "Managed",
          "rawDeploymentServiceConfig": "Headed"
        },
        "trustyai": {
          "managementState": "Managed"
        }
      }
    }
  }'
  ok "DSC patched — waiting for TrustyAI operator + GuardrailsOrchestrator CRD"
}

wait_trustyai_crd() {
  local timeout="${WAIT_TRUSTYAI_CRD_SEC:-1800}"
  local end=$((SECONDS + timeout))
  while (( SECONDS < end )); do
    if "$KUBECTL" get crd guardrailsorchestrators.trustyai.opendatahub.io >/dev/null 2>&1; then
      ok "GuardrailsOrchestrator CRD available"
      return 0
    fi
    local ta
    ta="$("$KUBECTL" get dsc default-dsc -o jsonpath='{.status.components.trustyai.managementState}:{.status.components.kserve.managementState}' 2>/dev/null || true)"
    printf '  waiting CRD… trustyai/kserve status=%s (%ds)\n' "${ta:-unknown}" "$SECONDS"
    sleep 20
  done
  warn "GuardrailsOrchestrator CRD not available within ${timeout}s"
  "$KUBECTL" get dsc default-dsc -o jsonpath='trustyai={.status.components.trustyai}{"\n"}kserve={.status.components.kserve}{"\n"}' 2>/dev/null || true
  return 1
}

ensure_trustyai_platform() {
  if check_trustyai_crd; then
    return 0
  fi
  enable_dsc_trustyai
  wait_trustyai_crd || return 1
}

apply_manifests_pre_helm() {
  step "Apply NetObserv gateway + guard proxy (${GUARDRAILS_NS})"
  "$KUBECTL" apply -f "$ROOT/manifests/trustyai-guardrails/00-namespace.yaml" \
    -f "$ROOT/manifests/trustyai-guardrails/01-gateway-config.yaml" \
    -f "$ROOT/manifests/trustyai-guardrails/03-llm-guard-proxy.yaml"
}

apply_orchestrator_detector_config() {
  step "Patch orchestrator NLP config (built-in-detector)"
  "$KUBECTL" apply -f "$ROOT/manifests/trustyai-guardrails/02-orchestrator-builtin-detector.yaml"
}

patch_orchestrator_gateway() {
  step "Enable Guardrails Gateway on guardrails-orchestrator"
  if ! "$KUBECTL" -n "$GUARDRAILS_NS" get guardrailsorchestrator guardrails-orchestrator >/dev/null 2>&1; then
    warn "GuardrailsOrchestrator guardrails-orchestrator not found — wait for Helm / TrustyAI controller"
    return 1
  fi
  "$KUBECTL" -n "$GUARDRAILS_NS" patch guardrailsorchestrator guardrails-orchestrator --type merge -p '{
    "spec": {
      "enableGuardrailsGateway": true,
      "enableBuiltInDetectors": true,
      "guardrailsGatewayConfig": "netobserv-guardrails-gateway-config"
    }
  }'
  ok "Gateway enabled (preset netobserv-sre → /netobserv-sre/v1/chat/completions)"
}

wait_kserve_webhook() {
  step "Wait for KServe webhook (required for InferenceService CRs)"
  local timeout="${WAIT_KSERVE_WEBHOOK_SEC:-900}"
  local end=$((SECONDS + timeout))
  while (( SECONDS < end )); do
    local eps
    eps="$("$KUBECTL" -n redhat-ods-applications get endpoints kserve-webhook-server-service \
      -o jsonpath='{.subsets[0].addresses[0].ip}' 2>/dev/null || true)"
    if [[ -n "$eps" ]]; then
      ok "KServe webhook ready"
      return 0
    fi
    printf '  waiting kserve-webhook-server-service endpoints… (%ds)\n' "$SECONDS"
    sleep 15
  done
  warn "KServe webhook not ready within ${timeout}s"
  return 1
}

clone_lemonade_repo() {
  local dest="$1"
  rm -rf "$dest"
  mkdir -p "$dest"
  git init "$dest/repo"
  git -C "$dest/repo" remote add origin "$LEMONADE_REPO"
  if ! git -C "$dest/repo" fetch --depth 1 origin "$LEMONADE_COMMIT" 2>/dev/null; then
    warn "fetch commit ${LEMONADE_COMMIT:0:12} failed — trying branch ${LEMONADE_BRANCH}"
    rm -rf "$dest/repo"
    git clone --depth 1 --branch "$LEMONADE_BRANCH" "$LEMONADE_REPO" "$dest/repo" || return 1
  else
    git -C "$dest/repo" checkout --detach FETCH_HEAD
  fi
  local got
  got="$(git -C "$dest/repo" rev-parse HEAD)"
  if [[ "$got" != "$LEMONADE_COMMIT" ]]; then
    warn "LEMONADE_COMMIT mismatch: expected ${LEMONADE_COMMIT:0:12} got ${got:0:12}"
    return 1
  fi
  ok "lemonade-stand-assistant @ ${got:0:12} (${LEMONADE_BRANCH})"
}

helm_install() {
  command -v helm >/dev/null 2>&1 || { warn "helm required for full install"; return 1; }
  wait_kserve_webhook || return 1
  read -r llm_host llm_port <<< "$(resolve_llm_host_port)"
  local api_key=""
  api_key="$("$KUBECTL" -n "$OPENCLAW_NS" get secret my-llm-key -o jsonpath='{.data.api-key}' 2>/dev/null | base64 -d || true)"
  step "Helm install fms-orchestrator (lemonade-stand @ ${LEMONADE_COMMIT:0:12}) → ${llm_host}:${llm_port}"
  step "Chart: ${LEMONADE_REPO} fms-orchestrator/chart (upstream Option A — external MaaS)"

  local tmp chart pi_mem_req pi_mem_lim helm_extra=()
  tmp="$(mktemp -d)"
  clone_lemonade_repo "$tmp" || { rm -rf "$tmp"; return 1; }

  chart="$tmp/repo/fms-orchestrator/chart"
  if [[ ! -d "$chart" ]]; then
    rm -rf "$tmp"
    warn "chart missing at $chart — expected fms-orchestrator/chart in ${LEMONADE_REPO}"
    return 1
  fi

  pi_mem_req="16Gi"
  pi_mem_lim="24Gi"
  if [[ "${LIGHT_DETECTORS:-0}" == "1" ]]; then
    pi_mem_req="8Gi"
    pi_mem_lim="16Gi"
    warn "LIGHT_DETECTORS=1 — reduced prompt-injection memory (lab only)"
  fi
  if [[ -n "$api_key" ]]; then
    helm_extra+=(--set-string "model.api_key=${api_key}")
  else
    warn "my-llm-key not found — Helm model.api_key unset (upstream Option A)"
  fi

  if ! helm upgrade --install "$HELM_RELEASE" "$chart" \
    --namespace "$GUARDRAILS_NS" \
    --create-namespace \
    --wait --timeout 45m \
    --set "model.endpoint=${llm_host}" \
    --set "model.port=${llm_port}" \
    "${helm_extra[@]}" \
    --set detectors.hap.useGpu=false \
    --set detectors.promptInjection.useGpu=false \
    --set "detectors.promptInjection.resources.requests.memory=${pi_mem_req}" \
    --set "detectors.promptInjection.resources.limits.memory=${pi_mem_lim}" \
    --set metrics.dashboard.enabled=false; then
    rm -rf "$tmp"
    warn "Helm install failed — see ${LEMONADE_REPO} README (Prerequisites + Option A)"
    return 1
  fi
  rm -rf "$tmp"
  ok "Helm release ${HELM_RELEASE} installed"
  wait_minio_model_cache || true
}

wait_minio_model_cache() {
  step "Wait for MinIO HuggingFace cache (lemonade-stand detector weights)"
  if ! "$KUBECTL" -n "$GUARDRAILS_NS" get deploy/minio-storage-guardrail-detectors >/dev/null 2>&1; then
    warn "minio-storage-guardrail-detectors missing — skip model cache wait"
    return 0
  fi
  "$KUBECTL" -n "$GUARDRAILS_NS" rollout status deploy/minio-storage-guardrail-detectors --timeout=45m
  ok "MinIO ready — recycling detector predictors (storage-initializer race on fresh install)"
  "$KUBECTL" -n "$GUARDRAILS_NS" delete pod -l component=predictor --ignore-not-found --wait=false
}

# Upstream fms-orchestrator chart (lemonade-stand-assistant) ships workshop-only assets.
# NetObserv uses orchestrator + detectors + netobserv-llm-guard-proxy — prune when SKIP_LEMONADE_STAND=1 (default).
prune_trustyai_workshop_extras() {
  step "Prune unused TrustyAI workshop assets (${GUARDRAILS_NS})"
  local kinds=(
    deployment.apps/lemonade-stand
    service/lemonade-stand
    route/lemonade-stand
    configmap/lemonade-stand-system-prompt
    secret/lemonade-stand-secrets
    servicemonitor.monitoring.coreos.com/lemonade-stand
    inferenceservice.serving.kserve.io/llama-32
    servingruntime.serving.kserve.io/llama-32
  )
  local any=0
  for obj in "${kinds[@]}"; do
    if "$KUBECTL" -n "$GUARDRAILS_NS" get "$obj" >/dev/null 2>&1; then
      "$KUBECTL" -n "$GUARDRAILS_NS" delete "$obj" --ignore-not-found --wait=false >/dev/null 2>&1 || true
      ok "deleted $obj"
      any=1
    fi
  done
  if [[ "$any" == "1" ]]; then
    ok "Workshop extras removed (lemonade-stand UI, optional in-cluster llama-32)"
  else
    ok "No workshop extras present"
  fi
}

maybe_prune_lemonade_stand() {
  if [[ "${SKIP_LEMONADE_STAND}" == "1" ]]; then
    prune_trustyai_workshop_extras
  else
    ok "SKIP_LEMONADE_STAND=0 — keeping upstream lemonade-stand workshop UI + Route"
  fi
}

remove_lemonade_stand() {
  prune_trustyai_workshop_extras
}

wait_orchestrator() {
  step "Wait for GuardrailsOrchestrator + detector InferenceServices"
  local i
  for i in $(seq 1 60); do
    if "$KUBECTL" -n "$GUARDRAILS_NS" get guardrailsorchestrator guardrails-orchestrator \
      -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null | grep -q True; then
      ok "GuardrailsOrchestrator Ready"
      return 0
    fi
    sleep 10
  done
  warn "Orchestrator not Ready within timeout — check: oc get guardrailsorchestrator,isvc -n ${GUARDRAILS_NS}"
  return 1
}

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

cmd_status() {
  step "Supply chain pin"
  ok "lemonade-stand-assistant ${LEMONADE_BRANCH} @ ${LEMONADE_COMMIT:0:12} (scripts/supply-chain-pins.env)"
  check_trustyai_crd || true
  step "Namespace ${GUARDRAILS_NS}"
  "$KUBECTL" get ns "$GUARDRAILS_NS" 2>/dev/null || warn "namespace missing"
  "$KUBECTL" -n "$GUARDRAILS_NS" get guardrailsorchestrator,isvc,svc,pods 2>/dev/null || true
  if "$KUBECTL" -n "$GUARDRAILS_NS" get guardrailsorchestrator guardrails-orchestrator >/dev/null 2>&1; then
    "$KUBECTL" -n "$GUARDRAILS_NS" get guardrailsorchestrator guardrails-orchestrator \
      -o jsonpath='enableGateway={.spec.enableGuardrailsGateway} gatewayCfg={.spec.guardrailsGatewayConfig}{"\n"}' 2>/dev/null || true
    ok "Gateway baseUrl (OpenClaw): $(gateway_base_url)"
  fi
  if "$KUBECTL" -n "$GUARDRAILS_NS" get deploy/lemonade-stand >/dev/null 2>&1 \
      || "$KUBECTL" -n "$GUARDRAILS_NS" get inferenceservice/llama-32 >/dev/null 2>&1; then
    warn "TrustyAI workshop extras still deployed — prune: $0 prune-workshop"
  else
    ok "TrustyAI workshop extras not deployed (expected for NetObserv demo)"
  fi
}

cmd_uninstall() {
  step "Uninstall TrustyAI guardrails (${GUARDRAILS_NS})"
  helm uninstall "$HELM_RELEASE" -n "$GUARDRAILS_NS" 2>/dev/null || true
  "$KUBECTL" delete -k "$ROOT/manifests/trustyai-guardrails" --ignore-not-found
  ok "Removed ${HELM_RELEASE} + manifests"
}

cmd_install() {
  ensure_trustyai_platform || exit 1
  apply_manifests_pre_helm
  if [[ "${SKIP_HELM:-0}" != "1" ]]; then
    helm_install || exit 1
    apply_orchestrator_detector_config
    maybe_prune_lemonade_stand
  else
    warn "SKIP_HELM=1 — assuming orchestrator already deployed"
    apply_orchestrator_detector_config
    maybe_prune_lemonade_stand
  fi
  patch_orchestrator_gateway || true
  if [[ -x "$ROOT/scripts/patch-trustyai-gateway-exposure.sh" ]]; then
    "$ROOT/scripts/patch-trustyai-gateway-exposure.sh" apply || warn "gateway exposure patch failed"
  fi
  wait_orchestrator || true
  step "Next: wire OpenClaw LLM → gateway"
  ok "./scripts/wire-openclaw-trustyai-guardrails.sh all"
  ok "Proof: ./scripts/run-spikee-guard-eval.sh quick"
}

case "$CMD" in
  install|all) cmd_install ;;
  status) cmd_status ;;
  prune-lemonade-stand|prune-workshop) prune_trustyai_workshop_extras ;;
  uninstall|delete) cmd_uninstall ;;
  gateway-url) gateway_base_url ;;
  *)
    echo "usage: $0 [install|status|prune-workshop|prune-lemonade-stand|uninstall|gateway-url]" >&2
    exit 1
    ;;
esac
