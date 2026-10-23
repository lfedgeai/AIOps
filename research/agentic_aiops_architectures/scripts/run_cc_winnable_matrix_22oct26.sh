#!/bin/bash
# config_corruption winnable test matrix (post patch_deployment implementation).
#
# Purpose: verify the patch_deployment + get_deployment_config tool path makes the
# config_corruption fault winnable, per agent, with a smaller matrix than Option 2.
#
# Differences vs scripts/run_c_matrix_scenario_a_c_permutations.sh (intentional):
#   - FAULTS: config_corruption only
#   - AGENTS: 5 (qwen38-27b skipped: 44% agent_error in the Oct 20-22 batch)
#   - CONTEXT_LEVELS: 8 representative labels (override with CONTEXT_LEVELS)
#   - Output: out/ccw_22oct26/ (separate from out/ batch JSONs; own kickoff file)
#   - No auto report/compare/learnings steps (report is written manually)
# Everything else (prompts, skills, gate, RAG, timeouts, poll/duration) matches the
# Option 2 matrix so cells stay comparable to the Oct 20-22 batch.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

set -a && source config/.env && set +a
export CLICKHOUSE_HTTP="${CLICKHOUSE_HTTP:-http://127.0.0.1:38123}"
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-900}"
export MAX_TOOL_ITERATIONS="${MAX_TOOL_ITERATIONS:-20}"
export K8S_FAULT_NAMESPACE=otel-demo
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5050}"
export HARNESS_PYTHON="${HARNESS_PYTHON:-/usr/bin/python3.13}"
export SCENARIO="${SCENARIO:-a}"
export FORCE_RAG="${FORCE_RAG:-1}"
export AGENT_SKILLS="${AGENT_SKILLS:-1}"
export AGENT_EXECUTE_GATE="${AGENT_EXECUTE_GATE:-1}"
DETECTION_TIMEOUT="${DETECTION_TIMEOUT:-1200}"

OUTDIR="$ROOT/out/ccw_22oct26"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/run.log"
KICKOFF_FILE="$OUTDIR/.kickoff"

# Own resume state (independent of the Option 2 matrix kickoff file)
if [[ "${MATRIX_RESUME:-0}" =~ ^(1|true|yes)$ ]] && [[ -f "$KICKOFF_FILE" ]]; then
  echo "[ccw-matrix] RESUME kickoff=$(cat "$KICKOFF_FILE") (SKIP_EXISTING=1)" | tee -a "$LOG"
  export SKIP_EXISTING=1
else
  date '+%Y-%m-%d %H:%M:%S' >"$KICKOFF_FILE"
fi
KICKOFF_EPOCH="$(date -d "$(cat "$KICKOFF_FILE")" +%s)"

FAULTS="${FAULTS:-config_corruption}"
if [[ -n "${CONTEXT_LEVELS:-}" ]]; then
  LEVELS="$CONTEXT_LEVELS"
else
  LEVELS="0 1 2 3 4 1,2 2,3,4 1,2,3,4"
fi

AGENTS=(
  "rhdp_deepseek-r1-distill-qwen-14b"
  "rhdp_qwen3-14b"
  "litemaas_qwen3.8-27b"
  "rhdp_granite-3-2-8b-instruct"
  "prelude_glm-53-flash"
)

# Fail fast if any matrix LLM endpoint is down
PREFLIGHT_AGENTS="$(IFS=,; echo "${AGENTS[*]}")"
"$HARNESS_PYTHON" "$ROOT/scripts/preflight_agent_apis.py" --agents "$PREFLIGHT_AGENTS" || exit 1

safe_ctx() { echo "$1" | tr ',' '-'; }

num_f=$(echo $FAULTS | wc -w)
num_l=$(echo $LEVELS | wc -w)
num_a=${#AGENTS[@]}
total=$((num_f * num_l * num_a))

echo "[ccw-matrix] Starting: $(date)" | tee "$LOG"
echo "[ccw-matrix] scenario=$SCENARIO agents=$num_a faults=$num_f contexts=$num_l total=$total" | tee -a "$LOG"
echo "[ccw-matrix] contexts: $LEVELS" | tee -a "$LOG"
echo "[ccw-matrix] FORCE_RAG=$FORCE_RAG AGENT_SKILLS=$AGENT_SKILLS AGENT_EXECUTE_GATE=$AGENT_EXECUTE_GATE AGENT_TIMEOUT=$AGENT_TIMEOUT DETECTION_TIMEOUT=$DETECTION_TIMEOUT MLFLOW=$MLFLOW_TRACKING_URI CH=$CLICKHOUSE_HTTP" | tee -a "$LOG"
echo "[ccw-matrix] new tools active: get_deployment_config + patch_deployment (config_corruption winnable path)" | tee -a "$LOG"

start_mlflow() {
  if curl -sf http://127.0.0.1:5050/ >/dev/null 2>&1; then
    echo "[ccw-matrix] MLflow already running on :5050" | tee -a "$LOG"
    return 0
  fi
  echo "[ccw-matrix] Starting MLflow on :5050" | tee -a "$LOG"
  cd "$ROOT"
  env -i HOME="$HOME" PATH="/usr/local/bin:/usr/bin:/bin" \
    MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false \
    /usr/bin/python3.13 -m mlflow server \
      --backend-store-uri "sqlite:///$ROOT/mlflow_local.db" \
      --default-artifact-root "$ROOT/mlartifacts" \
      --serve-artifacts --host 127.0.0.1 --port 5050 --workers 1 >>"$LOG" 2>&1 &
  for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:5050/ >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "[ccw-matrix] WARNING: MLflow may not be ready" | tee -a "$LOG"
}

refresh_infra() {
  if ! oc whoami >/dev/null 2>&1; then
    if [[ -n "${OCP_API_URL:-}" && -n "${KUBEADMIN_PASSWORD:-}" ]]; then
      oc login "$OCP_API_URL" \
        --username=kubeadmin --password="$KUBEADMIN_PASSWORD" \
        --insecure-skip-tls-verify >/dev/null 2>&1 || true
    fi
  fi
  if ! curl -sf "$CLICKHOUSE_HTTP/" --data "SELECT 1" >/dev/null 2>&1; then
    pkill -f "port-forward.*38123" 2>/dev/null || true
    sleep 2
    oc port-forward -n agentic-aiops svc/clickhouse 38123:8123 >>"$LOG" 2>&1 &
    sleep 6
  fi
  oc exec -n otel-demo deploy/valkey-cart -- redis-cli CONFIG SET stop-writes-on-bgsave-error no >/dev/null 2>&1 || true
  local gate_attempt
  for gate_attempt in 1 2 3; do
    if "$HARNESS_PYTHON" "$ROOT/scripts/pre_run_health_gate.py"; then
      break
    fi
    if [[ "$gate_attempt" == "3" ]]; then
      echo "[ccw-matrix] health gate failed 3x — aborting" | tee -a "$LOG"
      exit 1
    fi
    echo "[ccw-matrix] health gate attempt $gate_attempt failed; retrying in 60s" | tee -a "$LOG"
    sleep 60
  done
}

# Preflight corpora (forced RAG needs C1-C4 populated)
echo "[ccw-matrix] Validating C1–C4 corpora..." | tee -a "$LOG"
"$HARNESS_PYTHON" - <<'PY' | tee -a "$LOG"
from code.tools.context_engineering import ensure_context_corpora_ready
ensure_context_corpora_ready(frozenset({1, 2, 3, 4}))
print("[ccw-matrix] corpora OK")
PY

start_mlflow
count=0
failures=0
skipped=0

for agent_name in "${AGENTS[@]}"; do
  for level in $LEVELS; do
    for fault in $FAULTS; do
      count=$((count + 1))
      safe="$(safe_ctx "$level")"
      out="$OUTDIR/context_C${safe}_s${SCENARIO}_${fault}_${agent_name}.json"
      if [[ "${SKIP_EXISTING:-0}" =~ ^(1|true|yes)$ ]] && [[ -f "$out" ]]; then
        out_mtime="$(stat -c %Y "$out" 2>/dev/null || echo 0)"
        if [[ "$out_mtime" -ge "$KICKOFF_EPOCH" ]]; then
          skipped=$((skipped + 1))
          echo "[ccw-matrix] SKIP $count/$total (exists this kickoff) C{$level} agent=$agent_name fault=$fault" | tee -a "$LOG"
          continue
        fi
      fi
      echo "" | tee -a "$LOG"
      echo "========================================" | tee -a "$LOG"
      echo "[ccw-matrix] Run $count/$total C{$level} agent=$agent_name fault=$fault scenario=$SCENARIO" | tee -a "$LOG"
      echo "[ccw-matrix] $(date) -> $out" | tee -a "$LOG"
      echo "========================================" | tee -a "$LOG"
      refresh_infra
      export CONTEXT_C="$level"
      "$HARNESS_PYTHON" -u code/harness/run_harness.py \
        --agents "$agent_name" \
        --scenario "$SCENARIO" \
        --context-c "$level" \
        --detection-timeout "$DETECTION_TIMEOUT" \
        --poll-interval 30 \
        --fault-duration 90 \
        --flag "$fault" \
        --variant cart \
        -o "$out" 2>&1 | tee -a "$LOG" || {
          failures=$((failures + 1))
          echo "[ccw-matrix] RUN FAILED" | tee -a "$LOG"
        }
      sleep 8
    done
  done
done

echo "" | tee -a "$LOG"
echo "[ccw-matrix] COMPLETE: $(date) failures=$failures/$total skipped=$skipped" | tee -a "$LOG"
echo "[ccw-matrix] Results in $OUTDIR/ — analyze with the Oct 22 by-fault table as baseline (config_corruption was 0/96 recovery pre-patch_deployment)."
