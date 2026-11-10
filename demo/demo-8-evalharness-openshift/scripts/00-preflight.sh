#!/usr/bin/env bash
# Validate local tools, cluster access, storage class, and optional RHOAI/GPU.
set -euo pipefail

. "$(dirname "$0")/lib.sh"

need curl
need "$PYTHON_BIN"
"$PYTHON_BIN" -c "import requests, yaml, kubernetes, mlflow, openai, sklearn, pandas" >/dev/null 2>&1 || \
  die "Python dependencies missing; run: $PYTHON_BIN -m pip install -r requirements.txt -r code/harness/requirements.txt"

oc_login

log "cluster: $("$(kube)" config current-context 2>/dev/null || true)"
if [[ "$(kube)" == "oc" ]]; then
  log "user:    $(oc whoami 2>/dev/null || true)"
else
  log "user:    $(kubectl auth whoami -o name 2>/dev/null || true)"
fi
log "namespaces: $AIOPS_NAMESPACE (AIOps), $OTEL_DEMO_NAMESPACE (demo), $HARNESS_NAMESPACE (harness)"
log "storageClass: $STORAGE_CLASS"

if ! "$(kube)" get storageclass "$STORAGE_CLASS" >/dev/null 2>&1; then
  warn "storage class '$STORAGE_CLASS' not found. Set STORAGE_CLASS in .env."
  "$(kube)" get storageclass 2>/dev/null || true
fi

log "RHOAI namespaces: operator=$RHOAI_OPERATOR_NAMESPACE applications=$RHOAI_APPLICATIONS_NAMESPACE platform-evalhub=$EVALHUB_PLATFORM_NAMESPACE tenant=$EVALHUB_NAMESPACE"

if [[ "${REQUIRE_RHOAI:-0}" =~ ^(1|true|yes)$ ]]; then
  "$(kube)" get namespace "$RHOAI_OPERATOR_NAMESPACE" >/dev/null 2>&1 \
    || die "RHOAI operator namespace '$RHOAI_OPERATOR_NAMESPACE' not found; set RHOAI_OPERATOR_NAMESPACE in .env"
  "$(kube)" get namespace "$RHOAI_APPLICATIONS_NAMESPACE" >/dev/null 2>&1 \
    || die "RHOAI applications namespace '$RHOAI_APPLICATIONS_NAMESPACE' not found; set RHOAI_APPLICATIONS_NAMESPACE in .env"
  "$(kube)" get csv -n "$RHOAI_OPERATOR_NAMESPACE" 2>/dev/null | grep -q 'rhods-operator' \
    || die "no rhods-operator CSV in '$RHOAI_OPERATOR_NAMESPACE'; install the RHOAI operator or fix RHOAI_OPERATOR_NAMESPACE"
  "$(kube)" get crd datascienceclusters.datasciencecluster.opendatahub.io >/dev/null 2>&1 \
    || die "RHOAI DataScienceCluster CRD not found; install RHOAI or set REQUIRE_RHOAI=0"
  "$(kube)" get dsc >/dev/null 2>&1 || warn "no DataScienceCluster found"
  "$(kube)" get crd evalhubs.trustyai.opendatahub.io >/dev/null 2>&1 \
    && log "EvalHub CRD present" \
    || warn "EvalHub CRD not present (enable trustyai in the DataScienceCluster for the Evaluations UI)"
fi

if [[ "${CHECK_PULL_SECRET:-1}" =~ ^(1|true|yes)$ ]]; then
  if ! "$ROOT/scripts/check-pull-secret.sh"; then
    if [[ "${REQUIRE_RHOAI:-0}" =~ ^(1|true|yes)$ ]]; then
      die "cluster pull secret is missing Red Hat registry credentials; RHOAI operator images will not pull"
    else
      warn "pull-secret check failed; RHOAI operator images may not pull (see docs/AWS-OCP.md)"
    fi
  fi
fi

if [[ "${VLLM_ENABLED:-0}" =~ ^(1|true|yes)$ ]]; then
  "$(kube)" get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null | head -20 || true
  if ! "$(kube)" get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' 2>/dev/null | grep -q '[1-9]'; then
    warn "no allocatable $VLLM_GPU_RESOURCE found; vLLM will stay Pending unless GPU nodes are available"
  fi
fi

log "preflight complete"
