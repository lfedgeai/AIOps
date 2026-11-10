#!/usr/bin/env bash
# Run one AIOps evaluation cell against the OTEL Demo on OCP/RHOAI.
#
# Prerequisites:
#   - OTEL Demo + ClickHouse deployed (./scripts/deploy.sh)
#   - MLflow deployed, or MLFLOW_MODE=local
#   - model endpoint(s) configured in .env
#   - cluster credentials (KUBECONFIG, or OCP_API_URL + OCP_TOKEN)
#
# Examples:
#   ./scripts/run_harness.sh --scenario a --flag scale_zero --variant cart
#   ./scripts/run_harness.sh --scenario a --flag kill_pod --variant cart --agents rhdp_qwen3-14b
#   ./scripts/run_harness.sh --all-scenarios
set -euo pipefail

. "$(dirname "$0")/lib.sh"
ensure_dirs
check_python_deps

if [[ -z "${KUBERNETES_SERVICE_HOST:-}" ]]; then
  oc_login
else
  log "running in-cluster; using pod service account"
fi

ensure_clickhouse
ensure_mlflow

if [[ "${HARNESS_PREFLIGHT:-0}" =~ ^(1|true|yes)$ ]]; then
  run_preflight
fi

log "starting harness: scenario/fault args: $*"
exec "$PYTHON_BIN" -u "$ROOT/code/harness/run_harness.py" "$@"
