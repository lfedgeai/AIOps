#!/usr/bin/env bash
# Fallback: pull latest evidence.json from capture-proxy into workspace evidence/.
# Prefer MCP netobserv_analyze_evidence (reads /latest in-cluster without syncing).
set -euo pipefail

CAPTURE_URL="${NETOBSERV_CAPTURE_URL:-http://netobserv-capture-proxy.openclaw.svc.cluster.local:8080/latest}"
EVID_DIR=""
for root in /sandbox /opt/openclaw/workspace .; do
  if [[ -d "$root/evidence" ]]; then
    EVID_DIR="$root/evidence"
    break
  fi
done
if [[ -z "$EVID_DIR" ]]; then
  EVID_DIR="$(find /sandbox /opt/openclaw/workspace . -type d -name evidence 2>/dev/null | head -1 || true)"
fi
if [[ -z "$EVID_DIR" ]]; then
  EVID_DIR="./evidence"
  mkdir -p "$EVID_DIR"
fi

if command -v curl >/dev/null 2>&1; then
  curl -sf "$CAPTURE_URL" -o "${EVID_DIR}/latest.json"
elif python3 -c "import urllib.request" 2>/dev/null; then
  python3 - <<PY
import urllib.request
from pathlib import Path
data = urllib.request.urlopen("${CAPTURE_URL}", timeout=120).read()
Path("${EVID_DIR}/latest.json").write_bytes(data)
PY
else
  echo "ERROR: need curl or python3 to fetch capture evidence" >&2
  exit 1
fi

echo "EVIDENCE=${EVID_DIR}/latest.json"
python3 -c "import json; json.load(open('${EVID_DIR}/latest.json'))" && echo "EVIDENCE_VALID=true"
