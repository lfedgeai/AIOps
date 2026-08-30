#!/usr/bin/env bash
# Fallback entrypoint for NetObserv evidence analysis inside OpenShell.
# Prefer MCP netobserv_analyze_evidence (runs summarize-evidence.py in the MCP pod).
# OpenShell nests the workspace under openclaw-openshell-upload-*; always discover paths.
set -euo pipefail
SCRIPT="$(find /sandbox /opt/openclaw/workspace . -name summarize-evidence.py 2>/dev/null | head -1 || true)"
EVIDENCE="${1:-}"
if [[ -z "$EVIDENCE" ]]; then
  EVIDENCE="$(find /sandbox /opt/openclaw/workspace . -path '*/evidence/latest.json' 2>/dev/null | head -1 || true)"
fi
if [[ -z "$SCRIPT" ]]; then
  echo "ERROR: summarize-evidence.py not found" >&2
  find /sandbox -maxdepth 5 -type f -name '*.py' 2>/dev/null | head -40 >&2 || true
  exit 1
fi
if [[ -z "$EVIDENCE" || ! -f "$EVIDENCE" ]]; then
  echo "ERROR: evidence file not found (run netobserv_capture_flows + netobserv_analyze_evidence via MCP, or sync-evidence.sh then retry)" >&2
  exit 1
fi
echo "SCRIPT=$SCRIPT"
echo "EVIDENCE=$EVIDENCE"
exec python3 "$SCRIPT" "$EVIDENCE" "${@:2}"
