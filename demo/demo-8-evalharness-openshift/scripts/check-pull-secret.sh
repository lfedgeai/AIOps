#!/usr/bin/env bash
# Check that the cluster global pull secret has Red Hat registry credentials.
# RHOAI operator images live in registry.redhat.io; without auth the operator
# pods fail with ImagePullBackOff/unauthorized.
#
# This script prints registry hostnames only — never credentials.
set -euo pipefail

. "$(dirname "$0")/lib.sh"
oc_login

secret_json="$("$(kube)" get secret "$PULL_SECRET_NAME" -n "$PULL_SECRET_NAMESPACE" -o json 2>/dev/null || true)"
if [[ -z "$secret_json" ]]; then
  warn "cannot read ${PULL_SECRET_NAMESPACE}/${PULL_SECRET_NAME} (cluster-admin required)"
  exit 2
fi

"$PYTHON_BIN" - "$secret_json" <<'PY'
import base64
import json
import sys

doc = json.loads(sys.argv[1])
data = doc.get("data") or {}
encoded = data.get(".dockerconfigjson")
if not encoded:
    print(f"ERROR: secret has no .dockerconfigjson key", file=sys.stderr)
    raise SystemExit(2)

cfg = json.loads(base64.b64decode(encoded))
auths = cfg.get("auths") or {}
registries = sorted({a.replace("https://", "").replace("http://", "").rstrip("/") for a in auths})
print("registries in cluster pull secret: " + (", ".join(registries) if registries else "(none)"))

required = ["registry.redhat.io"]
recommended = ["registry.connect.redhat.com"]
missing = [r for r in required if r not in registries]
if missing:
    print(
        "MISSING Red Hat registry auth for: " + ", ".join(missing) + "\n"
        "RHOAI operator images are hosted at registry.redhat.io. Merge your\n"
        "Red Hat pull secret into openshift-config/pull-secret before installing RHOAI.",
        file=sys.stderr,
    )
    raise SystemExit(1)

missing_optional = [r for r in recommended if r not in registries]
if missing_optional:
    print("NOTE: " + ", ".join(missing_optional) + " not present (not always required for RHOAI)")
print("OK: Red Hat registry credentials are present")
PY
