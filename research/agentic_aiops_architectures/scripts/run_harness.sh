#!/bin/bash
# Run MTTD/MTTR harness against OTEL demo on OpenShift
#
# Prerequisites:
# - OTEL demo running (frontend, services, load-generator)
# - ClickHouse receiving logs/traces (verify: oc exec -n otel-demo deploy/clickhouse -- wget -qO- 'http://127.0.0.1:8123/?query=SELECT%20count()%20FROM%20otel.otel_logs')
# - Port-forward ClickHouse and MLflow (or use routes)
#
# Usage:
#   export FLAGD_READ_URL="https://$(oc get route flagd-ui-api -n otel-demo -o jsonpath='{.spec.host}')/api/read"
#   export FLAGD_WRITE_URL="https://$(oc get route flagd-ui-api -n otel-demo -o jsonpath='{.spec.host}')/api/write"
#   ./scripts/run_harness.sh [--flag cartFailure] [--variant on]

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Port-forward ClickHouse (if not already exposed)
if ! curl -s "${CLICKHOUSE_HTTP:-http://localhost:8123}/?query=SELECT%201" >/dev/null 2>&1; then
  echo "Starting port-forward for ClickHouse (background)..."
  oc port-forward -n otel-demo svc/clickhouse 8123:8123 &
  PF_PID=$!
  sleep 3
  export CLICKHOUSE_HTTP="http://localhost:8123"
fi

# Port-forward MLflow if needed
MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
if ! curl -s "$MLFLOW_URI/health" >/dev/null 2>&1; then
  echo "Starting port-forward for MLflow (background)..."
  oc port-forward -n agentic-aiops svc/mlflow 5000:5000 &
  sleep 2
  export MLFLOW_TRACKING_URI="http://localhost:5000"
fi

# Ensure deps
pip install -q -r code/harness/requirements.txt -r code/agents/reference_agent/requirements.txt 2>/dev/null || true

python code/harness/run_harness.py "$@"
