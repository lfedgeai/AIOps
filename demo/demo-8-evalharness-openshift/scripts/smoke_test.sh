#!/usr/bin/env bash
# One-cell smoke test: scale cart to zero, run one agent, recover.
set -euo pipefail

. "$(dirname "$0")/lib.sh"
ensure_dirs
check_python_deps

AGENT="${HARNESS_SMOKE_AGENT:-rhdp_qwen3-14b}"

if [[ -z "${KUBERNETES_SERVICE_HOST:-}" ]]; then
  oc_login
fi
ensure_clickhouse
ensure_mlflow

log "probing model endpoint(s): $AGENT"
"$PYTHON_BIN" "$ROOT/scripts/preflight_agent_apis.py" --agents "$AGENT" --timeout 120

log "checking cluster baseline (cart in $K8S_FAULT_NAMESPACE)"
"$PYTHON_BIN" "$ROOT/scripts/pre_run_health_gate.py" --target cart --namespace "$K8S_FAULT_NAMESPACE"

log "running smoke cell (scenario a, scale_zero/cart, agent=$AGENT)"
"$PYTHON_BIN" -u "$ROOT/code/harness/run_harness.py" \
  --scenario a \
  --flag scale_zero \
  --variant cart \
  --agents "$AGENT" \
  --detection-timeout 600 \
  --fault-duration 120 \
  --no-judge \
  --output "$ROOT/out/smoke_${AGENT//[^A-Za-z0-9_.-]/_}.json"

log "smoke test finished. Latest output:"
ls -1t "$ROOT/out"/smoke_*.json 2>/dev/null | head -1 || true
