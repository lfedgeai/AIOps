#!/bin/bash
# Run MTTD/MTTR harness against OTEL demo on OpenShift
#
# Prerequisites:
# - OTEL demo running (frontend, services, load-generator)
# - ClickHouse receiving logs/traces (verify: oc exec -n otel-demo deploy/clickhouse -- wget -qO- 'http://127.0.0.1:8123/?query=SELECT%20count()%20FROM%20otel.otel_logs')
# - MLflow deployed on OpenShift; set MLFLOW_TRACKING_URI to the Route HTTPS URL (or rely on discovery below)
#
# Usage:
#   export FLAGD_READ_URL="https://$(oc get route flagd-ui-api -n otel-demo -o jsonpath='{.spec.host}')/api/read"
#   export FLAGD_WRITE_URL="https://$(oc get route flagd-ui-api -n otel-demo -o jsonpath='{.spec.host}')/api/write"
#   ./scripts/run_harness.sh [--flag cartFailure] [--variant on]

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Load API keys from config/.env if it exists (gitignored)
if [[ -f config/.env ]]; then
  set -a
  source config/.env
  set +a
fi

# Flagd: prefer ConfigMap patch on OpenShift (flagd-ui write often returns 404)
if [[ -z "${FLAGD_USE_OPENSHIFT_CM}" ]] && oc get configmap flagd-config -n otel-demo &>/dev/null; then
  export FLAGD_USE_OPENSHIFT_CM=1
  export FLAGD_CONFIGMAP_NAMESPACE="${FLAGD_CONFIGMAP_NAMESPACE:-otel-demo}"
  echo "Using flagd ConfigMap (FLAGD_USE_OPENSHIFT_CM=1)"
fi

# Port-forward ClickHouse (if not already exposed)
if ! curl -s "${CLICKHOUSE_HTTP:-http://localhost:8123}/?query=SELECT%201" >/dev/null 2>&1; then
  echo "Starting port-forward for ClickHouse (background)..."
  oc port-forward -n otel-demo svc/clickhouse 8123:8123 &
  PF_PID=$!
  sleep 3
  export CLICKHOUSE_HTTP="http://localhost:8123"
fi

# MLflow: OpenShift tracking server only (no embedded local MLflow process).
# Discover Route when MLFLOW_TRACKING_URI is unset and oc + route exist.
if [[ -z "${MLFLOW_TRACKING_URI}" ]] && oc get route mlflow -n agentic-aiops -o jsonpath='{.spec.host}' &>/dev/null; then
  MLFLOW_HOST=$(oc get route mlflow -n agentic-aiops -o jsonpath='{.spec.host}')
  export MLFLOW_TRACKING_URI="https://${MLFLOW_HOST}"
  export MLFLOW_TRACKING_INSECURE_TLS=true
  echo "Using MLflow (OpenShift route): $MLFLOW_TRACKING_URI"
fi

if [[ -z "${MLFLOW_TRACKING_URI}" ]]; then
  echo "ERROR: MLFLOW_TRACKING_URI is not set." >&2
  echo "Set it to your OpenShift MLflow tracking URL (HTTPS route), e.g. from config/harness.yaml key mlflow.tracking_uri," >&2
  echo "or ensure 'oc get route mlflow -n agentic-aiops' works for automatic discovery." >&2
  exit 1
fi

# Optional: reachability check (self-signed routes need -k)
if [[ "${MLFLOW_TRACKING_INSECURE_TLS}" =~ ^(1|true|yes)$ ]]; then
  if ! curl -sf -k "${MLFLOW_TRACKING_URI}/health" >/dev/null 2>&1; then
    echo "WARNING: MLflow health check failed at ${MLFLOW_TRACKING_URI} (TLS skipped). Check VPN/cluster access." >&2
  fi
else
  if ! curl -sf "${MLFLOW_TRACKING_URI}/health" >/dev/null 2>&1; then
    echo "WARNING: MLflow health check failed at ${MLFLOW_TRACKING_URI}. For self-signed certs set MLFLOW_TRACKING_INSECURE_TLS=true" >&2
  fi
fi

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
