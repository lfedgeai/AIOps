#!/usr/bin/env bash
# Shared helpers for the standalone AIOps evaluation harness.
# Source this file from other scripts: . "$(dirname "$0")/lib.sh"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT

# Load repo-root .env (gitignored). .env is the source of truth; variables set
# in the shell before this file may be overwritten by .env.
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export AIOPS_NAMESPACE="${AIOPS_NAMESPACE:-agentic-aiops}"
export OTEL_DEMO_NAMESPACE="${OTEL_DEMO_NAMESPACE:-otel-demo}"
export HARNESS_NAMESPACE="${HARNESS_NAMESPACE:-$AIOPS_NAMESPACE}"
export K8S_FAULT_NAMESPACE="${K8S_FAULT_NAMESPACE:-$OTEL_DEMO_NAMESPACE}"
export K8S_NAMESPACE="${K8S_NAMESPACE:-$OTEL_DEMO_NAMESPACE}"
export FLAGD_CONFIGMAP_NAMESPACE="${FLAGD_CONFIGMAP_NAMESPACE:-$OTEL_DEMO_NAMESPACE}"

# RHOAI namespaces: neither the operator namespace nor the applications
# namespace is guaranteed to be the default. Prefer the explicit variables and
# keep the older aliases working.
export RHOAI_APPLICATIONS_NAMESPACE="${RHOAI_APPLICATIONS_NAMESPACE:-${RHOAI_NAMESPACE:-redhat-ods-applications}}"
export RHOAI_NAMESPACE="$RHOAI_APPLICATIONS_NAMESPACE"
export RHOAI_OPERATOR_NAMESPACE="${RHOAI_OPERATOR_NAMESPACE:-redhat-ods-operator}"
export EVALHUB_NAMESPACE="${EVALHUB_NAMESPACE:-rhods-notebooks}"
export EVALHUB_TENANT="${EVALHUB_TENANT:-$EVALHUB_NAMESPACE}"
export EVALHUB_PLATFORM_NAMESPACE="${EVALHUB_PLATFORM_NAMESPACE:-${EVALHUB_PLATFORM_NS:-$RHOAI_APPLICATIONS_NAMESPACE}}"
export EVALHUB_PLATFORM_NS="$EVALHUB_PLATFORM_NAMESPACE"
export RHODS_DASHBOARD_NAMESPACE="${RHODS_DASHBOARD_NAMESPACE:-$RHOAI_APPLICATIONS_NAMESPACE}"
export PULL_SECRET_NAMESPACE="${PULL_SECRET_NAMESPACE:-openshift-config}"
export PULL_SECRET_NAME="${PULL_SECRET_NAME:-pull-secret}"
export CLICKHOUSE_HTTP="${CLICKHOUSE_HTTP:-http://clickhouse.${AIOPS_NAMESPACE}.svc.cluster.local:8123}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-https://mlflow.redhat-ods-applications.svc.cluster.local:8443/mlflow}"
export PYTHON_BIN="${PYTHON_BIN:-${HARNESS_PYTHON:-python3}}"
export HARNESS_PYTHON="$PYTHON_BIN"
export CONTEXT_REPO_PATH="${CONTEXT_REPO_PATH:-$ROOT/context/opentelemetry-demo}"
export MLFLOW_TRACKING_INSECURE_TLS="${MLFLOW_TRACKING_INSECURE_TLS:-true}"
# RHOAI-managed MLflow requires a Kubernetes Bearer token.
# When running in-cluster, read the pod SA token and send it to MLflow.
if [[ -z "${MLFLOW_TRACKING_TOKEN:-}" ]] && [[ -f /var/run/secrets/kubernetes.io/serviceaccount/token ]]; then
  export MLFLOW_TRACKING_TOKEN="$(cat /var/run/secrets/kubernetes.io/serviceaccount/token 2>/dev/null || true)"
fi
export MLFLOW_SERVER_ENABLE_JOB_EXECUTION="${MLFLOW_SERVER_ENABLE_JOB_EXECUTION:-false}"
export PYTHONUNBUFFERED=1

# Deployment defaults (override in .env)
export STORAGE_CLASS="${STORAGE_CLASS:-gp3-csi}"
export CLICKHOUSE_IMAGE="${CLICKHOUSE_IMAGE:-clickhouse/clickhouse-server:24.3}"
export CLICKHOUSE_STORAGE_SIZE="${CLICKHOUSE_STORAGE_SIZE:-50Gi}"
export MLFLOW_IMAGE="${MLFLOW_IMAGE:-ghcr.io/mlflow/mlflow:v3.10.1}"
export MLFLOW_STORAGE_SIZE="${MLFLOW_STORAGE_SIZE:-10Gi}"
export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.8.5}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export VLLM_MODEL_CACHE_SIZE="${VLLM_MODEL_CACHE_SIZE:-50Gi}"
export VLLM_GPU_COUNT="${VLLM_GPU_COUNT:-1}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
export VLLM_GPU_RESOURCE="${VLLM_GPU_RESOURCE:-nvidia.com/gpu}"
export VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-qwen2.5-7b-instruct}"
export VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-hermes}"
export HF_TOKEN="${HF_TOKEN:-}"
export HARNESS_IMAGE="${HARNESS_IMAGE:-quay.io/your-org/aiops-harness:latest}"
export HARNESS_STORAGE_SIZE="${HARNESS_STORAGE_SIZE:-20Gi}"
export HARNESS_ARGS="${HARNESS_ARGS:---scenario a --flag scale_zero --variant cart --agents rhdp_qwen3-14b}"

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[%s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

need() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

kube() {
  if command -v oc >/dev/null 2>&1; then
    echo oc
  elif command -v kubectl >/dev/null 2>&1; then
    echo kubectl
  else
    die "required command not found: oc or kubectl"
  fi
}

render() {
  "$PYTHON_BIN" "$ROOT/scripts/render.py" <"$1"
}

apply_manifest() {
  render "$1" | "$(kube)" apply -f -
}

oc_login() {
  local k; k="$(kube)"
  local extra=()
  [[ "${OCP_INSECURE:-0}" =~ ^(1|true|yes)$ ]] && extra+=(--insecure-skip-tls-verify=true)
  if [[ "$k" == "oc" && -n "${OCP_API_URL:-}" && -n "${OCP_TOKEN:-}" ]]; then
    log "logging in to ${OCP_API_URL} with token"
    "$k" login "$OCP_API_URL" --token="$OCP_TOKEN" "${extra[@]}" >/dev/null
  elif [[ "$k" == "oc" && -n "${OCP_API_URL:-}" && -n "${OCP_USERNAME:-}" && -n "${OCP_PASSWORD:-}" ]]; then
    log "logging in to ${OCP_API_URL} as ${OCP_USERNAME}"
    "$k" login "$OCP_API_URL" --username="$OCP_USERNAME" --password="$OCP_PASSWORD" "${extra[@]}" >/dev/null
  elif [[ "$k" == "kubectl" ]]; then
    log "using kubectl/kubeconfig (oc login is not available)"
  fi
  if [[ "$k" == "oc" ]]; then
    "$k" whoami >/dev/null 2>&1 || die "not logged in to a cluster; set KUBECONFIG or OCP_API_URL/OCP_TOKEN in .env"
  else
    "$k" auth whoami >/dev/null 2>&1 || "$k" get --raw='/readyz' >/dev/null 2>&1 \
      || die "kubeconfig cannot reach the cluster; check KUBECONFIG in .env"
  fi
}

ensure_clickhouse() {
  if curl -sf --max-time 5 "${CLICKHOUSE_HTTP%/}/?query=SELECT%201" >/dev/null 2>&1; then
    log "ClickHouse reachable at ${CLICKHOUSE_HTTP}"
    return 0
  fi

  if [[ "${CLICKHOUSE_PORT_FORWARD:-0}" =~ ^(1|true|yes)$ ]]; then
    local k; k="$(kube)"
    local port="${CLICKHOUSE_LOCAL_PORT:-38123}"
    log "starting ClickHouse port-forward on ${port}"
    pkill -f "port-forward.*${port}:8123" 2>/dev/null || true
    "$k" port-forward -n "$AIOPS_NAMESPACE" svc/clickhouse "${port}:8123" >"$ROOT/logs/clickhouse-port-forward.log" 2>&1 &
    export CLICKHOUSE_HTTP="http://127.0.0.1:${port}"
    for _ in $(seq 1 30); do
      curl -sf --max-time 2 "${CLICKHOUSE_HTTP%/}/?query=SELECT%201" >/dev/null 2>&1 && {
        log "ClickHouse reachable at ${CLICKHOUSE_HTTP}"
        return 0
      }
      sleep 1
    done
  fi

  die "ClickHouse unreachable at ${CLICKHOUSE_HTTP}. Set CLICKHOUSE_HTTP or CLICKHOUSE_PORT_FORWARD=1."
}

ensure_mlflow() {
  local mode="${MLFLOW_MODE:-cluster}"
  if curl -sfkL --max-time 5 "${MLFLOW_TRACKING_URI%/}/health" >/dev/null 2>&1 \
     || curl -sfkL --max-time 5 "${MLFLOW_TRACKING_URI%/}/" >/dev/null 2>&1; then
    log "MLflow reachable at ${MLFLOW_TRACKING_URI}"
    return 0
  fi

  if [[ "$mode" == "local" ]]; then
    local port="${MLFLOW_PORT:-5050}"
    export MLFLOW_TRACKING_URI="http://127.0.0.1:${port}"
    mkdir -p "$ROOT/mlartifacts" "$ROOT/logs"
    log "starting local MLflow on :${port}"
    nohup "$PYTHON_BIN" -m mlflow server \
      --backend-store-uri "sqlite:///${ROOT}/mlflow_local.db" \
      --default-artifact-root "$ROOT/mlartifacts" \
      --serve-artifacts --host 0.0.0.0 --port "$port" --workers 1 \
      --allowed-hosts '*' --cors-allowed-origins '*' \
      >"$ROOT/logs/mlflow.log" 2>&1 &
    echo $! >"$ROOT/logs/mlflow.pid"
    for _ in $(seq 1 30); do
      curl -sfL --max-time 2 "${MLFLOW_TRACKING_URI%/}/health" >/dev/null 2>&1 && {
        log "MLflow ready at ${MLFLOW_TRACKING_URI}"
        return 0
      }
      sleep 1
    done
    die "local MLflow did not start; see $ROOT/logs/mlflow.log"
  fi

  die "MLflow unreachable at ${MLFLOW_TRACKING_URI}. Deploy manifests/mlflow.yaml or set MLFLOW_MODE=local."
}

ensure_dirs() {
  mkdir -p "$ROOT/logs" "$ROOT/out" "$ROOT/data" "$ROOT/mlartifacts"
}

check_python_deps() {
  "$PYTHON_BIN" -c "import requests, yaml, kubernetes, mlflow, openai, sklearn, pandas" >/dev/null 2>&1 || \
    die "Python dependencies missing; run: $PYTHON_BIN -m pip install -r $ROOT/requirements.txt -r $ROOT/code/harness/requirements.txt"
}

run_preflight() {
  "$PYTHON_BIN" "$ROOT/scripts/pre_run_health_gate.py" || exit 1
  "$PYTHON_BIN" "$ROOT/scripts/preflight_agent_apis.py" || exit 1
}
