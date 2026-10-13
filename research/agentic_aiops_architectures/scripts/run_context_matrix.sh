#!/bin/bash
# Shortened context-engineering matrix: Nemotron × 3 faults × 4 scenarios × L0–L2
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

FAULTS="scale_zero config_corruption kill_pod"
SCENARIOS="0 a b c"
LEVELS="0 1 2"
AGENT="code/agents/nemotron_agent/agent.py"
LOG="$ROOT/out/context_matrix_run.log"
mkdir -p "$ROOT/out"

num_f=$(echo $FAULTS | wc -w)
num_s=$(echo $SCENARIOS | wc -w)
num_l=$(echo $LEVELS | wc -w)
total=$((num_f * num_s * num_l))

echo "[context-matrix] Starting: $(date)" | tee "$LOG"
echo "[context-matrix] Total planned runs: $total" | tee -a "$LOG"

start_mlflow() {
  if curl -sf http://localhost:5050/ >/dev/null 2>&1; then
    echo "[context-matrix] MLflow already running on :5050" | tee -a "$LOG"
    return 0
  fi
  echo "[context-matrix] Starting MLflow on :5050" | tee -a "$LOG"
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
  echo "[context-matrix] WARNING: MLflow may not be ready" | tee -a "$LOG"
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
  "${HARNESS_PYTHON:-/usr/bin/python3.13}" "$ROOT/scripts/pre_run_health_gate.py" || exit 1
}

start_mlflow
count=0
failures=0

for level in $LEVELS; do
  for scenario in $SCENARIOS; do
    for fault in $FAULTS; do
      count=$((count + 1))
      out="out/context_L${level}_s${scenario}_${fault}_nemotron.json"
      echo "" | tee -a "$LOG"
      echo "========================================" | tee -a "$LOG"
      echo "[context-matrix] Run $count/$total L$level scenario=$scenario fault=$fault" | tee -a "$LOG"
      echo "[context-matrix] $(date)" | tee -a "$LOG"
      echo "========================================" | tee -a "$LOG"
      refresh_infra
      export CONTEXT_LEVEL="$level"
      "$HARNESS_PYTHON" -u code/harness/run_harness.py \
        --single-agent \
        --classifier "$AGENT" \
        --scenario "$scenario" \
        --context-level "$level" \
        --detection-timeout 720 \
        --poll-interval 30 \
        --fault-duration 90 \
        --flag "$fault" \
        --variant cart \
        -o "$out" 2>&1 | tee -a "$LOG" || {
        failures=$((failures + 1))
        echo "[context-matrix] RUN FAILED" | tee -a "$LOG"
      }
      sleep 8
    done
  done
done

echo "" | tee -a "$LOG"
echo "[context-matrix] COMPLETE: $(date) failures=$failures/$total" | tee -a "$LOG"
"$HARNESS_PYTHON" scripts/analyze_context_matrix.py --out-dir out --report docs/CONTEXT_ENGINEERING_EVAL_REPORT.md | tee -a "$LOG"
