#!/usr/bin/env bash
# Fail-closed verify: traces, metrics, AND logs in ClickHouse for a site_id.
set -euo pipefail

SITE_ID="${SITE_ID:-site-a}"
NAMESPACE="${CLICKHOUSE_NAMESPACE:-demo5-observability}"
LOOKBACK_MINUTES="${LOOKBACK_MINUTES:-30}"
CH_POD=""

die() { printf '[deploy-todo-otel] FAIL: %s\n' "$*" >&2; exit 1; }
log() { printf '[deploy-todo-otel] %s\n' "$*"; }

command -v oc >/dev/null || die "oc not found"
oc whoami >/dev/null 2>&1 || die "oc not logged in"

CH_POD="$(oc -n "${NAMESPACE}" get pod -l app=clickhouse -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
[[ -n "${CH_POD}" ]] || die "no clickhouse pod in ${NAMESPACE}"

query() {
  local sql="$1"
  oc -n "${NAMESPACE}" exec "${CH_POD}" -- clickhouse-client --database=otel -q "${sql}" 2>/dev/null || true
}

# OTEL ClickHouse exporter table names (contrib clickhouseexporter).
check_signal() {
  local label="$1"
  local sql="$2"
  local count
  count="$(query "${sql}" | tr -d '[:space:]')"
  if [[ -z "${count}" || "${count}" == "0" ]]; then
    die "no ${label} for site_id=${SITE_ID} in last ${LOOKBACK_MINUTES}m (got '${count:-empty}')"
  fi
  log "OK ${label}: ${count} rows (site_id=${SITE_ID})"
}

log "Verifying OTEL→ClickHouse for site_id=${SITE_ID} (fail closed on any missing signal)…"

check_signal "traces" \
  "SELECT count() FROM otel_traces WHERE ResourceAttributes['site_id'] = '${SITE_ID}' AND Timestamp >= now() - INTERVAL ${LOOKBACK_MINUTES} MINUTE"

check_signal "metrics" \
  "SELECT count() FROM otel_metrics_sum WHERE ResourceAttributes['site_id'] = '${SITE_ID}' AND TimeUnix >= now() - INTERVAL ${LOOKBACK_MINUTES} MINUTE"

check_signal "logs" \
  "SELECT count() FROM otel_logs WHERE ResourceAttributes['site_id'] = '${SITE_ID}' AND Timestamp >= now() - INTERVAL ${LOOKBACK_MINUTES} MINUTE"

log "PASS: traces + metrics + logs present for site_id=${SITE_ID}"
