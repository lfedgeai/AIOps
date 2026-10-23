#!/bin/bash
# Full C-matrix: 5 agents × 4 scenarios × 8 faults × C0–C4 (single corpus each)
# Total: 800 runs. Estimated runtime: ~40–65 hours at ~3–5 min/run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

set -a && source config/.env && set +a
export CLICKHOUSE_HTTP="${CLICKHOUSE_HTTP:-http://127.0.0.1:38123}"
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-600}"
export MAX_TOOL_ITERATIONS="${MAX_TOOL_ITERATIONS:-20}"
export K8S_FAULT_NAMESPACE="${K8S_FAULT_NAMESPACE:-otel-demo}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5050}"
export HARNESS_PYTHON="${HARNESS_PYTHON:-/usr/bin/python3.13}"

SCENARIOS="${SCENARIOS:-0 a b c}"
FAULTS="${FAULTS:-scale_zero kill_pod memory_limit network_partition readiness_probe_fail config_corruption dependency_removal replica_overload}"
CONTEXT_LEVELS="${CONTEXT_LEVELS:-0 1 2 3 4}"
DETECTION_TIMEOUT="${DETECTION_TIMEOUT:-480}"
FAULT_DURATION="${FAULT_DURATION:-120}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
VARIANT="${VARIANT:-cart}"

"$HARNESS_PYTHON" "$ROOT/scripts/preflight_agent_apis.py" || exit 1

AGENTS=(
  "prelude_nemotron-cascade-2-30b:code/agents/nemotron_agent/agent.py"
  "rhdp_deepseek-r1-distill-qwen-14b:code/agents/deepseek_agent/agent.py"
  "rhdp_qwen3-14b:code/agents/qwen3_agent/agent.py"
  "rhdp_qwen38-27b:code/agents/qwen38_27b_agent/agent.py"
  "litemaas_qwen3.8-27b:code/agents/qwen36_27b_agent/agent.py"
  "rhdp_granite-3-2-8b-instruct:code/agents/granite_agent/agent.py"
  "prelude_glm-53-flash:code/agents/glm_53_flash_agent/agent.py"
)

OUT_DIR="$ROOT/out/full_c_matrix"
LOG="$OUT_DIR/run.log"
mkdir -p "$OUT_DIR"

num_s=$(echo $SCENARIOS | wc -w)
num_f=$(echo $FAULTS | wc -w)
num_c=$(echo $CONTEXT_LEVELS | wc -w)
num_a=${#AGENTS[@]}
total=$((num_s * num_f * num_c * num_a))

echo "[full-c-matrix] Starting: $(date)" | tee "$LOG"
echo "[full-c-matrix] scenarios=$SCENARIOS" | tee -a "$LOG"
echo "[full-c-matrix] faults=$FAULTS" | tee -a "$LOG"
echo "[full-c-matrix] contexts=$CONTEXT_LEVELS" | tee -a "$LOG"
echo "[full-c-matrix] agents=$num_a AGENT_TIMEOUT=$AGENT_TIMEOUT detection_timeout=$DETECTION_TIMEOUT" | tee -a "$LOG"
echo "[full-c-matrix] total runs=$total output=$OUT_DIR" | tee -a "$LOG"

start_mlflow() {
  if curl -sf http://localhost:5050/ >/dev/null 2>&1; then
    echo "[full-c-matrix] MLflow already running on :5050" | tee -a "$LOG"
    return 0
  fi
  echo "[full-c-matrix] Starting MLflow on :5050" | tee -a "$LOG"
  cd "$ROOT"
  env -i HOME="$HOME" PATH="/usr/local/bin:/usr/bin:/bin" \
    "$HARNESS_PYTHON" -m mlflow server \
      --backend-store-uri "sqlite:///$ROOT/mlflow_local.db" \
      --default-artifact-root "$ROOT/mlartifacts" \
      --serve-artifacts --host 0.0.0.0 --port 5050 --workers 1 >>"$LOG" 2>&1 &
  for _ in $(seq 1 30); do
    if curl -sf http://localhost:5050/ >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "[full-c-matrix] WARNING: MLflow may not be ready" | tee -a "$LOG"
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

for scenario in $SCENARIOS; do
  for fault in $FAULTS; do
    for level in $CONTEXT_LEVELS; do
      for agent_entry in "${AGENTS[@]}"; do
        agent_name="${agent_entry%%:*}"
        agent_script="${agent_entry#*:}"
        count=$((count + 1))
        out="$OUT_DIR/context_C${level}_s${scenario}_${fault}_${agent_name}.json"
        echo "" | tee -a "$LOG"
        echo "========================================" | tee -a "$LOG"
        echo "[full-c-matrix] Run $count/$total C$level scenario=$scenario fault=$fault agent=$agent_name" | tee -a "$LOG"
        echo "[full-c-matrix] $(date)" | tee -a "$LOG"
        echo "========================================" | tee -a "$LOG"
        refresh_infra
        export CONTEXT_C="$level"
        export SCENARIO="$scenario"
        "$HARNESS_PYTHON" -u code/harness/run_harness.py \
          --agents "$agent_name" \
          --scenario "$scenario" \
          --context-c "$level" \
          --detection-timeout "$DETECTION_TIMEOUT" \
          --poll-interval "$POLL_INTERVAL" \
          --fault-duration "$FAULT_DURATION" \
          --flag "$fault" \
          --variant "$VARIANT" \
          -o "$out" 2>&1 | tee -a "$LOG" || {
          failures=$((failures + 1))
          echo "[full-c-matrix] RUN FAILED (exit $?)" | tee -a "$LOG"
        }
        sleep 8
      done
    done
  done
done

echo "" | tee -a "$LOG"
echo "[full-c-matrix] COMPLETE: $(date) failures=$failures/$total" | tee -a "$LOG"
"$HARNESS_PYTHON" scripts/analyze_context_matrix.py \
  --out-dir "$OUT_DIR" \
  --report docs/CONTEXT_ENGINEERING_EVAL_REPORT.md | tee -a "$LOG"
