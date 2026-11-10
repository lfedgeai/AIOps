#!/usr/bin/env bash
# Matrix runner for the standalone AIOps harness.
#
#   ./scripts/run_matrix.sh full          # agents x faults x scenarios 0/a/b/c
#   ./scripts/run_matrix.sh context       # scenario a x agents x 3 faults x C1-C4
#   ./scripts/run_matrix.sh permutations  # scenario a x agents x 3 faults x all C subsets
#   ./scripts/run_matrix.sh full-c        # full cross-product (very long)
#
# Useful env knobs:
#   HARNESS_AGENTS="rhdp_qwen3-14b,prelude_glm-53-flash"
#   FAULTS="scale_zero kill_pod"
#   CONTEXTS="0 1 2 3 4"
#   SKIP_EXISTING=1
#   HARNESS_EXTRA_ARGS="--no-judge --detection-timeout 900"
set -euo pipefail

. "$(dirname "$0")/lib.sh"
ensure_dirs
check_python_deps

MODE="${1:-full}"
VARIANT="${VARIANT:-cart}"
case "$MODE" in
  full|context|permutations|full-c) ;;
  *) die "usage: $0 {full|context|permutations|full-c}" ;;
esac

if [[ -z "${KUBERNETES_SERVICE_HOST:-}" ]]; then
  oc_login
fi
ensure_clickhouse
ensure_mlflow

AGENT_SPECS=(
  "rhdp_deepseek-r1-distill-qwen-14b:code/agents/deepseek_agent/agent.py"
  "litemaas_qwen3.8-27b:code/agents/qwen36_27b_agent/agent.py"
  "rhdp_granite-3-2-8b-instruct:code/agents/granite_agent/agent.py"
)

selected="${HARNESS_AGENTS:-}"
agents=()
for spec in "${AGENT_SPECS[@]}"; do
  name="${spec%%:*}"
  if [[ -z "$selected" ]] || [[ ",${selected}," == *",${name},"* ]]; then
    agents+=("$spec")
  fi
done
[[ ${#agents[@]} -gt 0 ]] || die "no agents selected (check HARNESS_AGENTS)"

agent_configured() {
  case "$1" in
    rhdp_deepseek-r1-distill-qwen-14b) [[ -n "${DEEPSEEK_API_BASE:-}" && -n "${DEEPSEEK_API_KEY:-}" ]] ;;
    rhdp_qwen3-14b)                    [[ -n "${QWEN3_API_BASE:-}" && -n "${QWEN3_API_KEY:-}" ]] ;;
    rhdp_qwen38-27b)                   [[ -n "${QWEN38_API_BASE:-}" && -n "${QWEN38_API_KEY:-}" ]] ;;
    litemaas_qwen3.8-27b)              [[ -n "${QWEN36_API_BASE:-}" && -n "${QWEN36_API_KEY:-}" ]] ;;
    rhdp_granite-3-2-8b-instruct)      [[ -n "${GRANITE_API_BASE:-}" && -n "${GRANITE_API_KEY:-}" ]] ;;
    prelude_glm-53-flash)              [[ -n "${GLM_API_BASE:-}" && -n "${GLM_API_KEY:-}" ]] ;;
    *) return 0 ;;
  esac
}

if [[ "${HARNESS_ALLOW_MISSING_AGENTS:-0}" =~ ^(1|true|yes)$ ]]; then
  filtered=()
  for spec in "${agents[@]}"; do
    if agent_configured "${spec%%:*}"; then
      filtered+=("$spec")
    else
      warn "skipping unconfigured agent ${spec%%:*}"
    fi
  done
  agents=("${filtered[@]}")
  [[ ${#agents[@]} -gt 0 ]] || die "no configured agents remain"
fi

case "$MODE" in
  full)
    FAULTS="${FAULTS:-scale_zero kill_pod memory_limit network_partition readiness_probe_fail config_corruption dependency_removal replica_overload}"
    SCENARIOS="${SCENARIOS:-0 a b c}"
    CONTEXTS="${CONTEXTS:-0}"
    ;;
  context)
    FAULTS="${FAULTS:-scale_zero config_corruption kill_pod}"
    SCENARIOS="a"
    CONTEXTS="${CONTEXTS:-1 2 3 4}"
    [[ "${INCLUDE_C0:-0}" =~ ^(1|true|yes)$ ]] && CONTEXTS="0 $CONTEXTS"
    ;;
  permutations)
    FAULTS="${FAULTS:-scale_zero config_corruption kill_pod}"
    SCENARIOS="a"
    CONTEXTS="${CONTEXTS:-0 1 2 3 4 1,2 1,3 1,4 2,3 2,4 3,4 1,2,3 1,2,4 1,3,4 2,3,4 1,2,3,4}"
    ;;
  full-c)
    FAULTS="${FAULTS:-scale_zero kill_pod memory_limit network_partition readiness_probe_fail config_corruption dependency_removal replica_overload}"
    SCENARIOS="${SCENARIOS:-0 a b c}"
    CONTEXTS="${CONTEXTS:-0 1 2 3 4 1,2 1,3 1,4 2,3 2,4 3,4 1,2,3 1,2,4 1,3,4 2,3,4 1,2,3,4}"
    ;;
esac

# Preflight the selected agents once. Set HARNESS_ALLOW_MISSING_AGENTS=1 to skip
# agents whose credentials are absent; matrices should normally be strict.
"$PYTHON_BIN" "$ROOT/scripts/preflight_agent_apis.py" --require-all \
  --agents "$(printf '%s\n' "${agents[@]}" | cut -d: -f1 | paste -sd, -)"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$ROOT/out/matrix_${MODE}_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
log "matrix mode=$MODE agents=${#agents[@]} scenarios=[$SCENARIOS] faults=[$FAULTS] contexts=[$CONTEXTS]"
log "log: $LOG"

total=$(( ${#agents[@]} * $(wc -w <<<"$SCENARIOS") * $(wc -w <<<"$FAULTS") * $(wc -w <<<"$CONTEXTS") ))
count=0
fail=0

for scenario in $SCENARIOS; do
  for fault in $FAULTS; do
    for context in $CONTEXTS; do
      for spec in "${agents[@]}"; do
        agent="${spec%%:*}"
        script="${spec#*:}"
        count=$((count + 1))
        safe_ctx="${context//,/_}"
        out="$ROOT/out/${MODE}_s${scenario}_${fault}_c${safe_ctx}_${agent}.json"

        if [[ "${SKIP_EXISTING:-1}" =~ ^(1|true|yes)$ && -s "$out" ]]; then
          log "[$count/$total] skip existing $out"
          continue
        fi

        log "[$count/$total] scenario=$scenario fault=$fault context=$context agent=$agent"
        if [[ -z "${KUBERNETES_SERVICE_HOST:-}" ]]; then
          oc_login
        fi
        set +e
        "$PYTHON_BIN" "$ROOT/scripts/pre_run_health_gate.py" --target "$VARIANT" || {
          warn "health gate failed; skipping cell"
          fail=$((fail + 1))
          set -e
          sleep "${MATRIX_CELL_GAP:-5}"
          continue
        }
        "$PYTHON_BIN" -u "$ROOT/code/harness/run_harness.py" \
          --scenario "$scenario" \
          --classifier "$ROOT/$script" \
          --flag "$fault" \
          --variant "$VARIANT" \
          --context-c "$context" \
          --output "$out" \
          ${HARNESS_EXTRA_ARGS:-}
        rc=$?
        set -e
        if [[ $rc -ne 0 ]]; then
          fail=$((fail + 1))
          warn "cell failed rc=$rc: scenario=$scenario fault=$fault context=$context agent=$agent"
        fi
        sleep "${MATRIX_CELL_GAP:-5}"
      done
    done
  done
done

log "matrix complete: $((count - fail))/$count successful, failures=$fail"
[[ $fail -eq 0 ]] || exit 1
