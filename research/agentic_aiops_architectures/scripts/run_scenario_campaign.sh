#!/bin/bash
# Run all agents across all scenarios (a, b, c) with the same K8s fault.
#
# Usage:
#   ./scripts/run_scenario_campaign.sh
#   ./scripts/run_scenario_campaign.sh --flag kill_pod --variant cart
#   FAULT_FLAG=scale_zero FAULT_VARIANT=cart ./scripts/run_scenario_campaign.sh

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FLAG="${FAULT_FLAG:-scale_zero}"
VARIANT="${FAULT_VARIANT:-cart}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/out}"
STAMP="$(date -u +%Y%m%d%H%M%S)"
OUT="$OUTPUT_DIR/harness_campaign_${FLAG}_${VARIANT}_${STAMP}.json"

mkdir -p "$OUTPUT_DIR"

echo "Campaign: all agents x scenarios a,b,c fault=${FLAG} variant=${VARIANT}"
echo "Output: $OUT"

./scripts/run_harness.sh \
  --all-scenarios \
  --flag "$FLAG" \
  --variant "$VARIANT" \
  --output "$OUT" \
  "$@"

echo "Done. Results: $OUT"
