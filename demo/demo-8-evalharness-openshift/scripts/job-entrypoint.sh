#!/usr/bin/env bash
# In-cluster Job entrypoint. Set HARNESS_ARGS to a simple, space-separated
# argument list, e.g.:
#   HARNESS_ARGS="--scenario a --flag scale_zero --variant cart --agents rhdp_qwen3-14b"
set -euo pipefail

. /app/scripts/lib.sh
ensure_dirs

if [[ -n "${HARNESS_ARGS:-}" ]]; then
  read -r -a args <<<"$HARNESS_ARGS"
  exec /app/scripts/run_harness.sh "${args[@]}"
fi

exec /app/scripts/run_harness.sh "$@"
