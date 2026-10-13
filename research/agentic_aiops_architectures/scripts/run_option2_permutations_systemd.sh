#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set -a
# shellcheck disable=SC1091
source config/.env
set +a
export CLICKHOUSE_HTTP="${CLICKHOUSE_HTTP:-http://127.0.0.1:38123}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5050}"
export MLFLOW_TRACKING_INSECURE_TLS=false
export HARNESS_PYTHON="${HARNESS_PYTHON:-/usr/bin/python3.13}"
export FORCE_RAG=1
export K8S_FAULT_NAMESPACE="${K8S_FAULT_NAMESPACE:-otel-demo}"

# Local MLflow only
systemctl --user start mlflow-local.service || true

# Fresh OCP session + ensure ClickHouse workload + PF
if [[ -n "${OCP_API_URL:-}" && -n "${KUBEADMIN_PASSWORD:-}" ]]; then
  oc login "$OCP_API_URL" \
    --username=kubeadmin --password="$KUBEADMIN_PASSWORD" \
    --insecure-skip-tls-verify >/dev/null 2>&1 || true
fi
oc scale deploy/clickhouse -n agentic-aiops --replicas=1 >/dev/null 2>&1 || true
oc scale deploy/cart -n otel-demo --replicas=1 >/dev/null 2>&1 || true
oc scale deploy/valkey-cart -n otel-demo --replicas=1 >/dev/null 2>&1 || true
oc exec -n otel-demo deploy/valkey-cart -- redis-cli CONFIG SET stop-writes-on-bgsave-error no >/dev/null 2>&1 || true

systemctl --user stop clickhouse-pf.service 2>/dev/null || true
pkill -f "port-forward.*38123" 2>/dev/null || true
sleep 2
systemctl --user start clickhouse-pf.service || true
for i in $(seq 1 30); do
  if curl -sf --max-time 3 "$CLICKHOUSE_HTTP/" --data "SELECT 1" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if ! curl -sf --max-time 3 "$CLICKHOUSE_HTTP/" --data "SELECT 1" >/dev/null 2>&1; then
  # Fallback: direct PF in this process tree
  oc port-forward -n agentic-aiops svc/clickhouse 38123:8123 >>"$ROOT/out/ch_pf_matrix.log" 2>&1 &
  sleep 6
fi

echo "[option2] lineup: deepseek qwen3 qwen38 qwen36 granite glm (nemotron/llama omitted; AGENT_EXECUTE_GATE=1)"
echo "[option2] see docs/RUN_NOTES_07Oct26.md"
# Agent API preflight also runs inside run_c_matrix_scenario_a_c_permutations.sh
exec bash "$ROOT/scripts/run_c_matrix_scenario_a_c_permutations.sh"
