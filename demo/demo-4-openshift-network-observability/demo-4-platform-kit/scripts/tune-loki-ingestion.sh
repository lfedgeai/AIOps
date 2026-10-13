#!/usr/bin/env bash
# tune-loki-ingestion.sh — raise LokiStack ingestion limits for NetObserv flow volume.
#
# Use on existing clusters when distributors log 429 "ingestion rate limit exceeded"
# during full demo load (~5+ MB/s flow ingest on busy labs).
#
#   ./scripts/tune-loki-ingestion.sh          # apply defaults
#   ./scripts/tune-loki-ingestion.sh status   # show current limits + recent 429s
#
# Env:
#   LOKI_NS=netobserv-loki
#   LOKISTACK_NAME=loki
#   LOKI_INGESTION_RATE=10        # MB/s (soft limit)
#   LOKI_INGESTION_BURST_SIZE=20  # MB per distributor push burst
#
set -euo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-apply}"

LOKI_NS="${LOKI_NS:-netobserv-loki}"
LOKISTACK_NAME="${LOKISTACK_NAME:-loki}"
LOKI_INGESTION_RATE="${LOKI_INGESTION_RATE:-10}"
LOKI_INGESTION_BURST_SIZE="${LOKI_INGESTION_BURST_SIZE:-20}"

KUBECTL="${KUBECTL:-oc}"

c_reset=$'\033[0m'; c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'
c_yellow=$'\033[1;33m'
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

show_status() {
  if ! "$KUBECTL" get lokistack "$LOKISTACK_NAME" -n "$LOKI_NS" >/dev/null 2>&1; then
    die "LokiStack $LOKI_NS/$LOKISTACK_NAME not found"
  fi
  size="$("$KUBECTL" get lokistack "$LOKISTACK_NAME" -n "$LOKI_NS" -o jsonpath='{.spec.size}')"
  rate="$("$KUBECTL" get lokistack "$LOKISTACK_NAME" -n "$LOKI_NS" -o jsonpath='{.spec.limits.global.ingestion.ingestionRate}')"
  burst="$("$KUBECTL" get lokistack "$LOKISTACK_NAME" -n "$LOKI_NS" -o jsonpath='{.spec.limits.global.ingestion.ingestionBurstSize}')"
  phase="$("$KUBECTL" get lokistack "$LOKISTACK_NAME" -n "$LOKI_NS" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')"
  printf 'LokiStack %s/%s  size=%s  Ready=%s\n' "$LOKI_NS" "$LOKISTACK_NAME" "${size:-?}" "${phase:-?}"
  printf '  ingestionRate=%s MB/s  ingestionBurstSize=%s MB\n' "${rate:-<operator default>}" "${burst:-<operator default>}"
  printf '  target: ingestionRate=%s  ingestionBurstSize=%s\n' "$LOKI_INGESTION_RATE" "$LOKI_INGESTION_BURST_SIZE"
  if "$KUBECTL" get pods -n "$LOKI_NS" -l app.kubernetes.io/component=distributor >/dev/null 2>&1; then
    n429="$("$KUBECTL" logs -n "$LOKI_NS" -l app.kubernetes.io/component=distributor --tail=200 2>/dev/null \
      | grep -c '429' || true)"
    printf '  distributor 429 lines (last 200 log lines): %s\n' "$n429"
  fi
}

apply_limits() {
  if ! "$KUBECTL" get lokistack "$LOKISTACK_NAME" -n "$LOKI_NS" >/dev/null 2>&1; then
    die "LokiStack $LOKI_NS/$LOKISTACK_NAME not found — run ./scripts/install-netobserv-aws.sh first"
  fi
  step "Patching LokiStack ingestion limits (rate=${LOKI_INGESTION_RATE} MB/s burst=${LOKI_INGESTION_BURST_SIZE} MB)"
  "$KUBECTL" patch lokistack "$LOKISTACK_NAME" -n "$LOKI_NS" --type=merge -p "$(cat <<EOF
{
  "spec": {
    "limits": {
      "global": {
        "ingestion": {
          "ingestionRate": ${LOKI_INGESTION_RATE},
          "ingestionBurstSize": ${LOKI_INGESTION_BURST_SIZE}
        }
      }
    }
  }
}
EOF
)" >/dev/null
  ok "LokiStack patched — operator will reconcile distributors/ingesters"
  step "Waiting for LokiStack Ready (timeout 300s)"
  if "$KUBECTL" wait --for=condition=Ready "lokistack/${LOKISTACK_NAME}" -n "$LOKI_NS" --timeout=300s >/dev/null 2>&1; then
    ok "LokiStack Ready"
  else
    warn "LokiStack not Ready within 300s — check: oc get pods -n $LOKI_NS"
  fi
  show_status
}

case "$CMD" in
  apply|all) apply_limits ;;
  status) show_status ;;
  *) die "usage: $0 [apply|status]" ;;
esac
