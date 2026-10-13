#!/bin/bash
# Context C-matrix Option 2: scenario A × agents × 3 faults × all CONTEXT_C subsets.
# Contexts: C0 + all non-empty subsets of {1,2,3,4} = 16 labels
#   0 | 1 2 3 4 | 1,2 1,3 1,4 2,3 2,4 3,4 | 1,2,3 1,2,4 1,3,4 2,3,4 | 1,2,3,4
# Forced RAG applies whenever any of C1–C4 is present.
# Does NOT sweep scenarios 0/b/c.
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
# Shared skills via code/agents/prompts.py (identical for every agent)
export AGENT_SKILLS="${AGENT_SKILLS:-1}"
# Shared diagnose→execute gate (block final JSON after detected=true until rem write)
export AGENT_EXECUTE_GATE="${AGENT_EXECUTE_GATE:-1}"
# Agent wall-clock: prior Option2 had 32/48 qwen36 timeouts at 600s
DETECTION_TIMEOUT="${DETECTION_TIMEOUT:-1200}"

LOG="$ROOT/out/c_matrix_scenario_a_c_permutations_run.log"
KICKOFF_FILE="$ROOT/out/.matrix_kickoff_aest"
mkdir -p "$ROOT/out"
date '+%Y-%m-%d %H:%M:%S' >"$KICKOFF_FILE"

FAULTS="${FAULTS:-scale_zero config_corruption kill_pod}"

# All CONTEXT_C permutations (comma-joined labels; "0" = no corpora)
if [[ -n "${CONTEXT_LEVELS:-}" ]]; then
  LEVELS="$CONTEXT_LEVELS"
else
  LEVELS="0 1 2 3 4 1,2 1,3 1,4 2,3 2,4 3,4 1,2,3 1,2,4 1,3,4 2,3,4 1,2,3,4"
fi

# Matrix lineup (preflight aborts if any listed endpoint is unhealthy).
# Omitted: nemotron-cascade-2 (prelude timeout/503); maas_llama31-70b (0% detect in Oct matrix).
AGENTS=(
  "maas_deepseek:code/agents/deepseek_agent/agent.py"
  "maas_qwen3:code/agents/qwen3_agent/agent.py"
  "maas_qwen38-27b:code/agents/qwen38_27b_agent/agent.py"
  "maas_qwen36-27b:code/agents/qwen36_27b_agent/agent.py"
  "maas_granite:code/agents/granite_agent/agent.py"
  "maas_glm-53-flash:code/agents/glm_53_flash_agent/agent.py"
)

# Fail fast if any matrix LLM endpoint is down (HTML/auth/connection errors)
PREFLIGHT_AGENTS=""
for agent_entry in "${AGENTS[@]}"; do
  name="${agent_entry%%:*}"
  PREFLIGHT_AGENTS="${PREFLIGHT_AGENTS:+$PREFLIGHT_AGENTS,}$name"
done
"$HARNESS_PYTHON" "$ROOT/scripts/preflight_agent_apis.py" --agents "$PREFLIGHT_AGENTS" || exit 1

# Filename-safe label: commas → hyphens (CONTEXT_C still uses commas)
safe_ctx() {
  echo "$1" | tr ',' '-'
}

num_f=$(echo $FAULTS | wc -w)
num_l=$(echo $LEVELS | wc -w)
num_a=${#AGENTS[@]}
total=$((num_f * num_l * num_a))

echo "[c-matrix-sa-perm] Starting: $(date)" | tee "$LOG"
echo "[c-matrix-sa-perm] scenario=$SCENARIO agents=$num_a faults=$num_f contexts=$num_l total=$total" | tee -a "$LOG"
echo "[c-matrix-sa-perm] contexts: $LEVELS" | tee -a "$LOG"
echo "[c-matrix-sa-perm] FORCE_RAG=$FORCE_RAG AGENT_SKILLS=$AGENT_SKILLS AGENT_EXECUTE_GATE=$AGENT_EXECUTE_GATE AGENT_TIMEOUT=$AGENT_TIMEOUT DETECTION_TIMEOUT=$DETECTION_TIMEOUT MLFLOW=$MLFLOW_TRACKING_URI CH=$CLICKHOUSE_HTTP" | tee -a "$LOG"

start_mlflow() {
  if curl -sf http://127.0.0.1:5050/ >/dev/null 2>&1; then
    echo "[c-matrix-sa-perm] MLflow already running on :5050" | tee -a "$LOG"
    return 0
  fi
  echo "[c-matrix-sa-perm] Starting MLflow on :5050" | tee -a "$LOG"
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
  echo "[c-matrix-sa-perm] WARNING: MLflow may not be ready" | tee -a "$LOG"
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
  # Keep cart writable across long matrix (valkey MISCONF)
  oc exec -n otel-demo deploy/valkey-cart -- redis-cli CONFIG SET stop-writes-on-bgsave-error no >/dev/null 2>&1 || true
  "$HARNESS_PYTHON" "$ROOT/scripts/pre_run_health_gate.py" || exit 1
}

# Preflight corpora (abort matrix early if empty)
echo "[c-matrix-sa-perm] Validating C1–C4 corpora..." | tee -a "$LOG"
"$HARNESS_PYTHON" - <<'PY' | tee -a "$LOG"
from code.tools.context_engineering import ensure_context_corpora_ready
ensure_context_corpora_ready(frozenset({1, 2, 3, 4}))
print("[c-matrix-sa-perm] corpora OK")
PY

start_mlflow
count=0
failures=0

for agent_entry in "${AGENTS[@]}"; do
  agent_name="${agent_entry%%:*}"
  agent_script="${agent_entry#*:}"
  for level in $LEVELS; do
    for fault in $FAULTS; do
      count=$((count + 1))
      safe="$(safe_ctx "$level")"
      out="out/context_C${safe}_s${SCENARIO}_${fault}_${agent_name}.json"
      echo "" | tee -a "$LOG"
      echo "========================================" | tee -a "$LOG"
      echo "[c-matrix-sa-perm] Run $count/$total C{$level} agent=$agent_name fault=$fault scenario=$SCENARIO" | tee -a "$LOG"
      echo "[c-matrix-sa-perm] $(date) -> $out" | tee -a "$LOG"
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
        echo "[c-matrix-sa-perm] RUN FAILED" | tee -a "$LOG"
      }
      sleep 8
    done
  done
done

echo "" | tee -a "$LOG"
echo "[c-matrix-sa-perm] COMPLETE: $(date) failures=$failures/$total" | tee -a "$LOG"

REPORT_DATE="$(date '+%d%b%y' | tr '[:upper:]' '[:lower:]')"
REPORT="docs/EVALUATION_RESULTS_${REPORT_DATE}.md"
echo "[c-matrix-sa-perm] Generating report $REPORT (kickoff=$(cat "$KICKOFF_FILE"))" | tee -a "$LOG"
"$HARNESS_PYTHON" "$ROOT/scripts/analyze_option2_run.py" \
  --report "$REPORT" \
  --kickoff "$(cat "$KICKOFF_FILE")" 2>&1 | tee -a "$LOG"

COMPARE_MD="docs/COMPARE_${REPORT_DATE}_vs_23sep26.md"
echo "[c-matrix-sa-perm] Comparing vs archive_sep22_option2 -> $COMPARE_MD" | tee -a "$LOG"
"$HARNESS_PYTHON" "$ROOT/scripts/compare_option2_kickoffs.py" \
  --baseline-kickoff "2026-09-22 00:02:00" \
  --baseline-dir "$ROOT/out/archive_sep22_option2" \
  --new-kickoff "$(cat "$KICKOFF_FILE")" \
  --markdown "$ROOT/$COMPARE_MD" 2>&1 | tee -a "$LOG" || true

echo "[c-matrix-sa-perm] Updating docs/Learnings_across_runs_report.md" | tee -a "$LOG"
"$HARNESS_PYTHON" "$ROOT/scripts/update_learnings_across_runs_report.py" \
  --kickoff "$(cat "$KICKOFF_FILE")" \
  --baseline-dir "$ROOT/out/archive_sep22_option2" \
  --eval-report "$REPORT" \
  --notes "Experiment vs Sep 22–23 baseline: shared AIOps skills in system prompt (AGENT_SKILLS=1), CHANGELOG removed from C2 docs corpus (INDEX_VERSION=no-changelog-v1), AGENT_TIMEOUT=900 / detection=1200. See $COMPARE_MD and $REPORT." \
  2>&1 | tee -a "$LOG" || true
