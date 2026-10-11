#!/bin/bash
# Run MTTD/MTTR harness against OTEL demo on OpenShift
#
# Prerequisites:
# - OTEL demo running (frontend, services, load-generator)
# - ClickHouse receiving logs/traces (verify: oc exec -n agentic-aiops deploy/clickhouse -- wget -qO- 'http://127.0.0.1:8123/?query=SELECT%20count()%20FROM%20otel.otel_logs')
# - Local MLflow on :5050 (started automatically via scripts/ensure_local_mlflow.sh)
#
# Usage:
#   export FLAGD_READ_URL="https://$(oc get route flagd-ui-api -n otel-demo -o jsonpath='{.spec.host}')/api/read"
#   export FLAGD_WRITE_URL="https://$(oc get route flagd-ui-api -n otel-demo -o jsonpath='{.spec.host}')/api/write"
#   ./scripts/run_harness.sh [--flag scale_zero] [--variant cart] [--scenario a]

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Load API keys from config/.env if it exists (gitignored)
if [[ -f config/.env ]]; then
  set -a
  source config/.env
  set +a
fi

# When the MaaS LiteLLM gateway is down, route maas_* agents through local Ollama.
# Set HARNESS_OLLAMA_MAAS_FALLBACK=1 to force; otherwise probe DEEPSEEK_API_BASE.
_apply_ollama_maas_fallback() {
  local probe_url="${DEEPSEEK_API_BASE:-}"
  [[ -z "$probe_url" ]] && return 0
  if [[ "${HARNESS_OLLAMA_MAAS_FALLBACK:-}" =~ ^(1|true|yes)$ ]] \
    || ! curl -sf --max-time 5 -o /dev/null -w "" \
      -H "Authorization: Bearer ${DEEPSEEK_API_KEY:-x}" \
      "${probe_url%/}/models" 2>/dev/null; then
    echo "MaaS LiteLLM unreachable at ${probe_url}; using Ollama fallback for maas_* agents." >&2
    export DEEPSEEK_API_BASE="${HARNESS_OLLAMA_BASE:-http://127.0.0.1:11434/v1}"
    export DEEPSEEK_MODEL="${HARNESS_OLLAMA_DEEPSEEK_MODEL:-mistral}"
    export DEEPSEEK_API_KEY="${HARNESS_OLLAMA_API_KEY:-ollama}"
    export QWEN3_API_BASE="${HARNESS_OLLAMA_BASE:-http://127.0.0.1:11434/v1}"
    export QWEN3_MODEL="${HARNESS_OLLAMA_QWEN_MODEL:-qwen2.5}"
    export QWEN3_API_KEY="${HARNESS_OLLAMA_API_KEY:-ollama}"
  fi
}
_apply_ollama_maas_fallback

# Flagd: prefer ConfigMap patch on OpenShift (flagd-ui write often returns 404)
if [[ -z "${FLAGD_USE_OPENSHIFT_CM}" ]] && oc get configmap flagd-config -n otel-demo &>/dev/null; then
  export FLAGD_USE_OPENSHIFT_CM=1
  export FLAGD_CONFIGMAP_NAMESPACE="${FLAGD_CONFIGMAP_NAMESPACE:-otel-demo}"
  echo "Using flagd ConfigMap (FLAGD_USE_OPENSHIFT_CM=1)"
fi

if [[ -z "${CLICKHOUSE_HTTP}" ]]; then
  export CLICKHOUSE_HTTP="http://127.0.0.1:38123"
fi
CLICKHOUSE_NAMESPACE="${CLICKHOUSE_NAMESPACE:-agentic-aiops}"

# Refresh OpenShift session + ClickHouse port-forward (same pattern as run_context_matrix.sh)
refresh_infra() {
  if [[ -n "${OCP_API_URL:-}" && -n "${KUBEADMIN_PASSWORD:-}" ]]; then
    timeout 20 oc login "$OCP_API_URL" \
      --username=kubeadmin --password="$KUBEADMIN_PASSWORD" \
      --insecure-skip-tls-verify >/dev/null 2>&1 || \
      echo "WARNING: oc login failed or timed out (cluster API may be down)." >&2
  fi
  if ! curl -sf --max-time 5 "$CLICKHOUSE_HTTP/" --data "SELECT 1" >/dev/null 2>&1; then
    echo "Starting port-forward for ClickHouse on ${CLICKHOUSE_HTTP} (background)..."
    pkill -f "port-forward.*38123:8123" 2>/dev/null || true
    sleep 1
    if timeout 10 oc get svc clickhouse -n "$CLICKHOUSE_NAMESPACE" >/dev/null 2>&1; then
      oc port-forward -n "$CLICKHOUSE_NAMESPACE" svc/clickhouse 38123:8123 &
      sleep 6
    else
      echo "WARNING: cannot reach OpenShift API to port-forward ClickHouse (ns=$CLICKHOUSE_NAMESPACE)." >&2
    fi
    if ! curl -sf --max-time 5 "$CLICKHOUSE_HTTP/" --data "SELECT 1" >/dev/null 2>&1; then
      echo "WARNING: ClickHouse still unreachable at $CLICKHOUSE_HTTP." >&2
    fi
  fi
  "${HARNESS_PYTHON:-python3}" "$ROOT/scripts/pre_run_health_gate.py" || exit 1
}
refresh_infra

# Port-forward ClickHouse (legacy path when CLICKHOUSE_HTTP points at :8123)
if ! curl -s "${CLICKHOUSE_HTTP}/?query=SELECT%201" >/dev/null 2>&1; then
  if [[ "$CLICKHOUSE_HTTP" == *":8123"* ]]; then
    echo "Starting port-forward for ClickHouse on :8123 (background)..."
    oc port-forward -n "$CLICKHOUSE_NAMESPACE" svc/clickhouse 8123:8123 &
    PF_PID=$!
    sleep 3
    export CLICKHOUSE_HTTP="http://localhost:8123"
  fi
fi

# Local MLflow only (never auto-discover OpenShift route)
# shellcheck source=ensure_local_mlflow.sh
source "$ROOT/scripts/ensure_local_mlflow.sh"
_ensure_local_mlflow || exit 1

# Ensure deps
pip install -q -r requirements.txt -r code/harness/requirements.txt 2>/dev/null || true

# Prefer real CPython for harness + agent subprocesses (avoid IDE/AppImage as Python)
if [[ -z "${PYTHONBIN}" ]] && command -v python3.13 &>/dev/null; then
  PYTHONBIN="$(command -v python3.13)"
elif [[ -z "${PYTHONBIN}" ]]; then
  PYTHONBIN="$(command -v python3)"
fi
export HARNESS_PYTHON="${HARNESS_PYTHON:-$PYTHONBIN}"
exec "$PYTHONBIN" -u code/harness/run_harness.py "$@"
