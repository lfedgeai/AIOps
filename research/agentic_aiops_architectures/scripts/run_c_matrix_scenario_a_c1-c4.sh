#!/bin/bash
# Context C-matrix: scenario A × agents × 3 faults × C1–C4 (single corpus each).
# C0 optional via INCLUDE_C0=1. Does NOT sweep scenarios 0/b/c (see run_full_c_matrix.sh).
# Filename encodes the default restriction: scenario_a + c1-c4.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

set -a && source config/.env && set +a
export CLICKHOUSE_HTTP="${CLICKHOUSE_HTTP:-http://127.0.0.1:38123}"
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-600}"
export MAX_TOOL_ITERATIONS="${MAX_TOOL_ITERATIONS:-20}"
export K8S_FAULT_NAMESPACE=otel-demo
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5050}"
export HARNESS_PYTHON="${HARNESS_PYTHON:-/usr/bin/python3.13}"
export SCENARIO="${SCENARIO:-a}"

"$HARNESS_PYTHON" "$ROOT/scripts/preflight_agent_apis.py" || exit 1

FAULTS="${FAULTS:-scale_zero config_corruption kill_pod}"
LEVELS="${CONTEXT_LEVELS:-1 2 3 4}"
if [[ "${INCLUDE_C0:-0}" =~ ^(1|true|yes)$ ]]; then
  LEVELS="0 $LEVELS"
fi

AGENTS=(
  "nemotron-cascade-2:code/agents/nemotron_agent/agent.py"
  "maas_deepseek:code/agents/deepseek_agent/agent.py"
  "maas_qwen3:code/agents/qwen3_agent/agent.py"
  "maas_qwen38-27b:code/agents/qwen38_27b_agent/agent.py"
  "maas_qwen36-27b:code/agents/qwen36_27b_agent/agent.py"
  "maas_granite:code/agents/granite_agent/agent.py"
  "maas_glm-53-flash:code/agents/glm_53_flash_agent/agent.py"
)

LOG="$ROOT/out/c_matrix_scenario_a_c1-c4_run.log"
mkdir -p "$ROOT/out"

num_f=$(echo $FAULTS | wc -w)
num_l=$(echo $LEVELS | wc -w)
num_a=${#AGENTS[@]}
total=$((num_f * num_l * num_a))

echo "[c-matrix-sa-c1-4] Starting: $(date)" | tee "$LOG"
echo "[c-matrix-sa-c1-4] scenario=$SCENARIO (default a) agents=$num_a faults=$num_f contexts=$num_l total=$total" | tee -a "$LOG"

start_mlflow() {
  if curl -sf http://localhost:5050/ >/dev/null 2>&1; then
    echo "[c-matrix-sa-c1-4] MLflow already running on :5050" | tee -a "$LOG"
    return 0
  fi
  echo "[c-matrix-sa-c1-4] Starting MLflow on :5050" | tee -a "$LOG"
  cd "$ROOT"
  env -i HOME="$HOME" PATH="/usr/local/bin:/usr/bin:/bin" \
    /usr/bin/python3.13 -m mlflow server \
      --backend-store-uri "sqlite:///$ROOT/mlflow_local.db" \
      --default-artifact-root "$ROOT/mlartifacts" \
      --serve-artifacts --host 0.0.0.0 --port 5050 --workers 1 >>"$LOG" 2>&1 &
  for i in $(seq 1 30); do
    if curl -sf http://localhost:5050/ >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "[c-matrix-sa-c1-4] WARNING: MLflow may not be ready" | tee -a "$LOG"
}

refresh_infra() {
  if [[ -n "${OCP_API_URL:-}" && -n "${KUBEADMIN_PASSWORD:-}" ]]; then
    oc login "$OCP_API_URL" \
      --username=kubeadmin --password="$KUBEADMIN_PASSWORD" \
      --insecure-skip-tls-verify >/dev/null 2>&1 || true
  fi
  if ! curl -sf "$CLICKHOUSE_HTTP/" --data "SELECT 1" >/dev/null 2>&1; then
    pkill -f "port-forward.*38123" 2>/dev/null || true
    sleep 2
    oc port-forward -n agentic-aiops svc/clickhouse 38123:8123 >>"$LOG" 2>&1 &
    sleep 6
  fi
  "$HARNESS_PYTHON" "$ROOT/scripts/pre_run_health_gate.py" || exit 1
}

start_mlflow
count=0
failures=0

for agent_entry in "${AGENTS[@]}"; do
  agent_name="${agent_entry%%:*}"
  agent_script="${agent_entry#*:}"
  for level in $LEVELS; do
    for fault in $FAULTS; do
      count=$((count + 1))
      out="out/context_C${level}_s${SCENARIO}_${fault}_${agent_name}.json"
      echo "" | tee -a "$LOG"
      echo "========================================" | tee -a "$LOG"
      echo "[c-matrix-sa-c1-4] Run $count/$total C$level agent=$agent_name fault=$fault scenario=$SCENARIO" | tee -a "$LOG"
      echo "[c-matrix-sa-c1-4] $(date)" | tee -a "$LOG"
      echo "========================================" | tee -a "$LOG"
      refresh_infra
      export CONTEXT_C="$level"
      "$HARNESS_PYTHON" -u code/harness/run_harness.py \
        --agents "$agent_name" \
        --scenario "$SCENARIO" \
        --context-c "$level" \
        --detection-timeout 720 \
        --poll-interval 30 \
        --fault-duration 90 \
        --flag "$fault" \
        --variant cart \
        -o "$out" 2>&1 | tee -a "$LOG" || {
        failures=$((failures + 1))
        echo "[c-matrix-sa-c1-4] RUN FAILED" | tee -a "$LOG"
      }
      sleep 8
    done
  done
done

echo "" | tee -a "$LOG"
echo "[c-matrix-sa-c1-4] COMPLETE: $(date) failures=$failures/$total" | tee -a "$LOG"
"$HARNESS_PYTHON" scripts/analyze_context_matrix.py --out-dir out --report docs/CONTEXT_ENGINEERING_EVAL_REPORT.md | tee -a "$LOG"
