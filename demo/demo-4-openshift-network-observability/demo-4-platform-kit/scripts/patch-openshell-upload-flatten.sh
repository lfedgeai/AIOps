#!/usr/bin/env bash
# Patch @openclaw/openshell-sandbox so remote uploads are flattened immediately.
# OpenShell `sandbox upload <tmpdir> <remotePath>` nests the temp directory
# basename (openclaw-openshell-upload-*) under remotePath. OpenClaw then reads
# skills at the flat path /.openclaw/sandbox-skills/skills/... → ENOENT.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
KUBECTL="$(command -v oc || command -v kubectl)"

POD="$("$KUBECTL" -n "$OPENCLAW_NS" get pods \
  -l app.kubernetes.io/name=openclaw \
  --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}')"
[[ -n "$POD" ]] || { echo "no openclaw pod" >&2; exit 1; }

"$KUBECTL" -n "$OPENCLAW_NS" cp \
  "$ROOT/scripts/patch-openshell-upload-flatten.py" \
  "${POD}:/tmp/patch-openshell-upload-flatten.py"

"$KUBECTL" -n "$OPENCLAW_NS" exec "$POD" -c openclaw -- \
  python3 /tmp/patch-openshell-upload-flatten.py
