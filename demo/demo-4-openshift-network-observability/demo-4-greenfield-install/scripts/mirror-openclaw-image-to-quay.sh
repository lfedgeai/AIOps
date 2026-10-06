#!/usr/bin/env bash
# Copy the lab-pinned OpenClaw image from a *working* cluster worker (CRI-O cache)
# into your Quay namespace when ryan_nix's digest was deleted upstream.
#
# Run on a bastion with `oc` access to a cluster where OpenClaw is still Running
# with the lab-pinned digest. Requires podman/skopeo on a worker (via oc debug).
#
# Usage:
#   export QUAY_ORG=your-org
#   export QUAY_REPO=openclaw-openshift
#   export QUAY_TAG=openclaw-v2026.6.11
#   ./scripts/mirror-openclaw-image-to-quay.sh
#
# Then on your greenfield bastion:
#   export OPENCLAW_GATEWAY_IMAGE=quay.io/your-org/openclaw-openshift:openclaw-v2026.6.11
#   export OPENCLAW_SANDBOX_IMAGE=quay.io/your-org/openclaw-openshift:openclaw-v2026.6.11
#   cd ~/AIOps/demo/demo-4-greenfield-install && ./scripts/phase2-openshell.sh deploy
set -euo pipefail

SOURCE_DIGEST="${SOURCE_DIGEST:-sha256:a91dbc1cc1879a46137da1b73e5c270ea3094cdc3753f5ce8636168c9e1d8d0c}"
SOURCE_REF="quay.io/ryan_nix/openclaw-openshift@${SOURCE_DIGEST}"
QUAY_ORG="${QUAY_ORG:-your-org}"
QUAY_REPO="${QUAY_REPO:-openclaw-openshift}"
QUAY_TAG="${QUAY_TAG:-openclaw-v2026.6.11}"
DEST_IMAGE="quay.io/${QUAY_ORG}/${QUAY_REPO}:${QUAY_TAG}"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
KUBECTL="$(command -v oc || command -v kubectl)"

c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }

[[ -n "$KUBECTL" ]] || { echo "oc/kubectl required" >&2; exit 1; }

step "Locate Running openclaw pod with ${SOURCE_DIGEST:0:19}…"
POD="$("$KUBECTL" -n "$OPENCLAW_NS" get pod -l app.kubernetes.io/name=openclaw \
  -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{"\n"}{end}' | head -1)"
[[ -n "$POD" ]] || { echo "No Running openclaw pod in ${OPENCLAW_NS}" >&2; exit 1; }

IMAGE_ID="$("$KUBECTL" -n "$OPENCLAW_NS" get pod "$POD" \
  -o jsonpath='{.status.containerStatuses[0].imageID}')"
[[ "$IMAGE_ID" == *"${SOURCE_DIGEST}"* ]] || {
  warn "Pod $POD imageID does not match expected digest: $IMAGE_ID"
  warn "Set SOURCE_DIGEST or fix the reference cluster before mirroring."
  exit 1
}
NODE="$("$KUBECTL" -n "$OPENCLAW_NS" get pod "$POD" -o jsonpath='{.spec.nodeName}')"
ok "Found $POD on node $NODE"

step "Export from worker CRI-O cache and push → ${DEST_IMAGE}"
# CRI-O image ID on disk (short id from crictl images)
SHORT_ID="$("$KUBECTL" debug "node/${NODE}" --to-namespace="${OPENCLAW_NS}" --quiet -- \
  chroot /host bash -lc "crictl images -o json | python3 -c \"
import json,sys
d=json.load(sys.stdin)
for img in d.get('images',[]):
  for ref in img.get('repoDigests',[]) + img.get('repoTags',[]):
    if '${SOURCE_DIGEST}' in ref or 'openclaw-openshift' in ref:
      print(img['id'].split(':')[-1][:12])
      sys.exit(0)
sys.exit(1)
\"" 2>/dev/null | tail -1)"

[[ -n "$SHORT_ID" ]] || { echo "Could not resolve CRI-O image id on node" >&2; exit 1; }
ok "CRI-O short id: ${SHORT_ID}"

AUTH_JSON="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/containers/auth.json"
if [[ -f "$AUTH_JSON" ]]; then
  _auth="$(jq -r '.auths["quay.io"].auth // empty' "$AUTH_JSON")"
  if [[ -n "$_auth" ]]; then
    _creds="$(printf '%s' "$_auth" | base64 -d 2>/dev/null || true)"
    QUAY_USER="${_creds%%:*}"
    QUAY_PASS="${_creds#*:}"
  fi
fi
QUAY_USER="${QUAY_USER:-$(podman login --get-login quay.io 2>/dev/null || true)}"
[[ -n "$QUAY_USER" && -n "$QUAY_PASS" ]] || {
  echo "Run: podman login quay.io (as ${QUAY_ORG}) on this bastion first" >&2
  exit 1
}

"$KUBECTL" debug "node/${NODE}" --to-namespace="${OPENCLAW_NS}" --quiet -- \
  chroot /host bash -lc "
set -euo pipefail
IMG=\$(crictl images -o json | python3 -c \"
import json,sys
d=json.load(sys.stdin)
for img in d.get('images',[]):
  if img['id'].endswith('${SHORT_ID}') or '${SHORT_ID}' in img['id']:
    print(img['id'])
    sys.exit(0)
sys.exit(1)
\")
echo \"Using CRI-O image: \$IMG\"
command -v skopeo >/dev/null
skopeo copy --override-os linux \
  \"containers-storage:\${IMG}\" \
  \"docker://${DEST_IMAGE}\" \
  --dest-creds '${QUAY_USER}:${QUAY_PASS}'
"

ok "Pushed ${DEST_IMAGE}"
step "Verify pull on this bastion"
podman pull "${DEST_IMAGE}"
podman run --rm --entrypoint node "${DEST_IMAGE}" /app/openclaw.mjs --version

cat <<EOF

Mirror complete. On the **greenfield** bastion:

  export OPENCLAW_GATEWAY_IMAGE=${DEST_IMAGE}
  export OPENCLAW_SANDBOX_IMAGE=${DEST_IMAGE}
  cd ~/AIOps/demo/demo-4-greenfield-install
  ./scripts/phase2-openshell.sh deploy

(Optional) Pin by digest after push:
  podman inspect ${DEST_IMAGE} --format '{{.Digest}}'
EOF
