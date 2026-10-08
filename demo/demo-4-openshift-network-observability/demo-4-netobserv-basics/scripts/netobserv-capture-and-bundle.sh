#!/usr/bin/env bash
#
# netobserv-capture-and-bundle.sh
# ---------------------------------------------------------------------------
# Runs an on-demand Network Observability CLI flow capture on the DB path,
# then extracts a compact "evidence bundle" (RTT stats, connection counts,
# drops, app error signatures, datasource config) for AI analysis.
#
# Run this WHILE the incident is live (after fault-injection 'slow').
#
# Output:
#   ./nob-capture-<ts>/output/...           raw capture (flow JSON + SQLite db)
#   ./nob-capture-<ts>/evidence.json        structured evidence for the AI step
#   ./nob-capture-<ts>/evidence.md          human-readable summary
#
# Read-only against the cluster (captures + reads; changes nothing).
# ---------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

APP_NS="${APP_NS:-todo-demo}"
DB_PORT="${DB_PORT:-5432}"
DURATION="${DURATION:-180}"          # capture seconds (keep captures short: <10 min)
WORKDIR="${WORKDIR:-./nob-capture-$(date +%Y%m%d-%H%M%S)}"

c_reset=$'\033[0m'; c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'
c_yellow=$'\033[1;33m'; c_red=$'\033[1;31m'
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
die()  { printf '%s[fail]%s %s\n' "$c_red" "$c_reset" "$*" >&2; exit 1; }

# --- preflight -------------------------------------------------------------
command -v oc >/dev/null 2>&1 || die "'oc' not found in PATH."
oc whoami >/dev/null 2>&1 || die "Not logged in. Run 'oc login ...' first."
command -v sqlite3 >/dev/null 2>&1 || die "'sqlite3' not found (needed to read the flow database)."
command -v jq >/dev/null 2>&1 || die "'jq' not found."
if ! oc netobserv version >/dev/null 2>&1; then
  die "The 'oc netobserv' plugin is not installed. Install it with:
     curl -LO https://mirror.openshift.com/pub/cgw/netobserv/latest/oc-netobserv-amd64
     sudo mv ./oc-netobserv-amd64 /usr/local/bin/oc-netobserv && chmod +x /usr/local/bin/oc-netobserv"
fi

mkdir -p "$WORKDIR"; cd "$WORKDIR"

# --- capture ---------------------------------------------------------------
step "Starting background flow capture on TCP:${DB_PORT} (RTT + packet drops), ${DURATION}s"
if oc netobserv flows --background --enable_rtt --enable_pkt_drop \
     --action=Accept --cidr=0.0.0.0/0 --protocol=TCP --port="${DB_PORT}" >capture.log 2>&1; then
  ok "Capture running in background."
else
  warn "Background capture failed to start. Your CLI may predate --background."
  warn "Fallback: in another terminal run, then Ctrl+C after ~${DURATION}s:"
  warn "  oc netobserv flows --enable_rtt --enable_pkt_drop --protocol=TCP --port=${DB_PORT}"
  die  "Re-run this script once you have an ./output directory with a flow *.db file."
fi

step "Collecting for ${DURATION}s (Ctrl+C will stop early)"
sleep "$DURATION"

step "Stopping and copying capture output"
oc netobserv stop    >>capture.log 2>&1 || warn "stop returned non-zero."
oc netobserv copy    >>capture.log 2>&1 || warn "copy returned non-zero."
oc netobserv cleanup >>capture.log 2>&1 || warn "cleanup returned non-zero."

DB="$(find output -name '*.db' 2>/dev/null | head -1 || true)"
[[ -n "$DB" ]] || die "No flow database found under ./output — check capture.log."
ok "Flow database: $DB"

# --- feature extraction (defensive: columns vary by CLI version) -----------
step "Extracting evidence from the flow database"
COLS="$(sqlite3 "$DB" 'PRAGMA table_info(flow);' | cut -d'|' -f2 | tr '\n' ' ')"
has_col() { [[ " $COLS " == *" $1 "* ]]; }

RTT_COL=""
for c in TimeFlowRttNs TimeFlowRTT FlowRtt; do has_col "$c" && { RTT_COL="$c"; break; }; done
DPORT_COL="DstPort"; has_col "$DPORT_COL" || DPORT_COL=""
SPORT_COL="SrcPort"; has_col "$SPORT_COL" || SPORT_COL=""

sql() { sqlite3 -noheader "$DB" "$1" 2>/dev/null; }

# port filter (match either direction on the DB port)
PORTF="1=1"
if [[ -n "$DPORT_COL" && -n "$SPORT_COL" ]]; then
  PORTF="($DPORT_COL=$DB_PORT OR $SPORT_COL=$DB_PORT)"
fi

TOTAL_FLOWS="$(sql "SELECT COUNT(*) FROM flow WHERE $PORTF;")"; TOTAL_FLOWS="${TOTAL_FLOWS:-0}"

RTT_MIN_MS=null; RTT_AVG_MS=null; RTT_MAX_MS=null; RTT_P95_MS=null
if [[ -n "$RTT_COL" ]]; then
  RTT_MIN_MS="$(sql "SELECT ROUND(MIN($RTT_COL)/1000000.0,2) FROM flow WHERE $PORTF AND $RTT_COL>0;")"
  RTT_AVG_MS="$(sql "SELECT ROUND(AVG($RTT_COL)/1000000.0,2) FROM flow WHERE $PORTF AND $RTT_COL>0;")"
  RTT_MAX_MS="$(sql "SELECT ROUND(MAX($RTT_COL)/1000000.0,2) FROM flow WHERE $PORTF AND $RTT_COL>0;")"
  # approximate p95 via ordered offset
  N="$(sql "SELECT COUNT(*) FROM flow WHERE $PORTF AND $RTT_COL>0;")"; N="${N:-0}"
  if [[ "$N" -gt 0 ]]; then
    OFF=$(( N*95/100 )); [[ "$OFF" -ge "$N" ]] && OFF=$((N-1))
    RTT_P95_MS="$(sql "SELECT ROUND($RTT_COL/1000000.0,2) FROM flow WHERE $PORTF AND $RTT_COL>0 ORDER BY $RTT_COL LIMIT 1 OFFSET $OFF;")"
  fi
  : "${RTT_MIN_MS:=null}"; : "${RTT_AVG_MS:=null}"; : "${RTT_MAX_MS:=null}"; : "${RTT_P95_MS:=null}"
fi

# drops (column name varies; try known variants)
DROP_FLOWS=0
for dc in PktDropPackets PktDropBytes PktDropLatestDropCause; do
  if has_col "$dc"; then
    DROP_FLOWS="$(sql "SELECT COUNT(*) FROM flow WHERE $PORTF AND $dc IS NOT NULL AND $dc<>0 AND $dc<>'';")"
    DROP_FLOWS="${DROP_FLOWS:-0}"; break
  fi
done

# top talkers to the DB port
TOP_TALKERS="[]"
if has_col SrcK8S_OwnerName && [[ -n "$DPORT_COL" ]]; then
  TOP_TALKERS="$(sqlite3 -json "$DB" \
    "SELECT SrcK8S_OwnerName AS owner, SrcK8S_Namespace AS ns, COUNT(*) AS flows
     FROM flow WHERE $DPORT_COL=$DB_PORT AND SrcK8S_OwnerName IS NOT NULL
     GROUP BY owner, ns ORDER BY flows DESC LIMIT 5;" 2>/dev/null || echo "[]")"
  [[ -z "$TOP_TALKERS" ]] && TOP_TALKERS="[]"
fi

# --- application-side evidence --------------------------------------------
step "Gathering application error signatures and datasource config"
APP_ERRORS="$(oc logs -n "$APP_NS" deploy/todo --tail=800 2>/dev/null \
  | grep -iE 'acquisition|timeout|agroal|pool|exhaust|SQLState|ERROR|WARN|500' \
  | tail -40 || true)"
DS_CONFIG="$(oc get cm todo-config -n "$APP_NS" -o jsonpath='{.data.application\.properties}' 2>/dev/null || true)"
TODO_RESTARTS="$(oc get pods -n "$APP_NS" -l app=todo -o jsonpath='{.items[0].status.containerStatuses[0].restartCount}' 2>/dev/null || echo 0)"

# --- assemble bundle -------------------------------------------------------
step "Writing evidence bundle"
jq -n \
  --arg app_ns "$APP_NS" \
  --argjson db_port "$DB_PORT" \
  --argjson total_flows "${TOTAL_FLOWS:-0}" \
  --argjson rtt_min "${RTT_MIN_MS:-null}" \
  --argjson rtt_avg "${RTT_AVG_MS:-null}" \
  --argjson rtt_p95 "${RTT_P95_MS:-null}" \
  --argjson rtt_max "${RTT_MAX_MS:-null}" \
  --argjson drop_flows "${DROP_FLOWS:-0}" \
  --argjson top_talkers "${TOP_TALKERS:-[]}" \
  --arg restarts "${TODO_RESTARTS:-0}" \
  --arg app_errors "$APP_ERRORS" \
  --arg ds_config "$DS_CONFIG" \
  '{
    scenario: "todo + PostgreSQL; DB path observed via oc netobserv",
    app_namespace: $app_ns,
    db_port: $db_port,
    network_evidence: {
      total_db_flows: $total_flows,
      rtt_ms: { min: $rtt_min, avg: $rtt_avg, p95: $rtt_p95, max: $rtt_max },
      dropped_flows: $drop_flows,
      top_talkers_to_db: $top_talkers
    },
    application_evidence: {
      todo_restart_count: ($restarts|tonumber? // 0),
      error_log_excerpts: ($app_errors | split("\n") | map(select(length>0))),
      datasource_config: $ds_config
    }
  }' > evidence.json
ok "evidence.json written."

# --- human-readable summary ------------------------------------------------
{
  echo "# Network Observability — Incident Evidence"
  echo
  echo "- App namespace: \`$APP_NS\`   DB port: \`$DB_PORT\`"
  echo "- Flows observed on DB path: **$TOTAL_FLOWS**   Dropped flows: **$DROP_FLOWS**   todo restarts: **$TODO_RESTARTS**"
  echo "- RTT to DB (ms): min=$RTT_MIN_MS avg=$RTT_AVG_MS p95=$RTT_P95_MS max=$RTT_MAX_MS"
  echo
  echo "## Top talkers to the database"
  echo '```json'
  echo "$TOP_TALKERS" | jq . 2>/dev/null || echo "$TOP_TALKERS"
  echo '```'
  echo
  echo "## Datasource config"
  echo '```properties'
  echo "$DS_CONFIG"
  echo '```'
  echo
  echo "## Application error signatures (tail)"
  echo '```'
  echo "$APP_ERRORS"
  echo '```'
} > evidence.md
ok "evidence.md written."

cat <<MSG

${c_green}Evidence bundle ready:${c_reset}
  $(pwd)/evidence.json   (feed this to the AI analysis step)
  $(pwd)/evidence.md     (human-readable)
  $(pwd)/$DB             (raw flow database, queryable with sqlite3)

Optional wire-level capture (manual; run in a separate terminal, Ctrl+C to stop):
  oc netobserv packets --protocol=TCP --port=${DB_PORT}
MSG
