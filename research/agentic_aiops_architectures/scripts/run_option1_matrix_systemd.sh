#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set -a
source config/.env
set +a
export INCLUDE_C0=1
export CLICKHOUSE_HTTP="${CLICKHOUSE_HTTP:-http://127.0.0.1:38123}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5050}"
export MLFLOW_TRACKING_INSECURE_TLS=false
export HARNESS_PYTHON="${HARNESS_PYTHON:-/usr/bin/python3.13}"
# Ensure local services
systemctl --user start mlflow-local.service || true
systemctl --user start clickhouse-pf.service || true
exec bash "$ROOT/scripts/run_c_matrix_scenario_a_c1-c4.sh"
