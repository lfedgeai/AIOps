#!/bin/bash
# Full evaluation matrix: all agents × all faults × scenario a (telemetry+K8s)
# Estimated runtime: ~8 hours
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

set -a && source config/.env && set +a
export CLICKHOUSE_HTTP="http://127.0.0.1:38123"
export AGENT_TIMEOUT=300
export K8S_FAULT_NAMESPACE=otel-demo
export MLFLOW_TRACKING_URI=http://localhost:5050

FAULTS="scale_zero kill_pod memory_limit network_partition readiness_probe_fail config_corruption dependency_removal replica_overload"
AGENTS="code/agents/nemotron_agent/agent.py code/agents/qwen3_agent/agent.py code/agents/deepseek_agent/agent.py code/agents/llama_scout_agent/agent.py"
SCENARIO="a"

echo "[matrix] Starting full evaluation: $(date)"
echo "[matrix] Faults: $FAULTS"
echo "[matrix] Scenario: $SCENARIO"

# Ensure MLflow is running
if ! curl -sf http://localhost:5050/ >/dev/null 2>&1; then
  echo "[matrix] Starting MLflow..."
  cd "$ROOT"
  env -i HOME=/home/redhat PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH="" \
    /usr/bin/python3.13 -m mlflow server \
      --backend-store-uri sqlite:///mlflow_local.db \
      --default-artifact-root ./mlartifacts \
      --serve-artifacts --host 0.0.0.0 --port 5050 --workers 1 &
  sleep 10
fi

refresh_infra() {
  # Re-login to OpenShift (token may have expired)
  OCP_API="${OCP_API_URL:?OCP_API_URL must be set in config/.env}"
  oc login "$OCP_API" \
    --username=kubeadmin --password="$KUBEADMIN_PASSWORD" \
    --insecure-skip-tls-verify >/dev/null 2>&1 || true
  # Re-establish ClickHouse port-forward if dead
  if ! curl -sf "http://127.0.0.1:38123/" --data "SELECT 1" >/dev/null 2>&1; then
    pkill -f "port-forward.*38123" 2>/dev/null || true
    sleep 2
    oc port-forward -n otel-demo svc/clickhouse 38123:8123 &
    sleep 5
  fi
}

count=0
total=32  # 4 agents x 8 faults

for fault in $FAULTS; do
  refresh_infra
  for agent in $AGENTS; do
    count=$((count + 1))
    agent_name=$(basename $(dirname $agent))
    echo ""
    echo "========================================"
    echo "[matrix] Run $count/$total: fault=$fault agent=$agent_name scenario=$SCENARIO"
    echo "[matrix] $(date)"
    echo "========================================"
    /usr/bin/python3.13 -u code/harness/run_harness.py \
      --scenario "$SCENARIO" \
      --classifier "$agent" \
      --detection-timeout 300 \
      --poll-interval 30 \
      --fault-duration 120 \
      --flag "$fault" \
      --variant cart \
      -o "out/matrix_${fault}_${agent_name}.json" 2>&1 | grep -E '\[harness\]|detected|Written'
    sleep 10
  done
done

echo ""
echo "[matrix] COMPLETE: $(date)"
echo "[matrix] Total runs: $count"
