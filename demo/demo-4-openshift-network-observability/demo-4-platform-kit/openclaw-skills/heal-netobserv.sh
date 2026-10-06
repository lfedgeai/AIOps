#!/usr/bin/env bash
# Stable heal/probe entrypoint for OpenShell.
# Prefer the in-cluster heal proxy (OpenShell blocks :6443 and cannot tunnel to
# kubernetes.default.svc). Fall back to local python+kubeconfig only outside OpenShell.
set -euo pipefail
CMD="${1:-probe}"
case "$CMD" in
  heal|probe|status) ;;
  *)
    echo "Usage: $0 {heal|probe|status}" >&2
    exit 1
    ;;
esac

PROXY_URLS=(
  "${NETOBSERV_HEAL_PROXY:-}"
  "http://netobserv-heal-proxy.openclaw.svc.cluster.local:8080"
  "http://netobserv-heal-proxy.openclaw.svc:8080"
)

call_proxy() {
  local base="$1"
  [[ -n "$base" ]] || return 1
  local url="${base%/}/${CMD}"
  echo "PROXY=${url}"
  # Prefer curl; fall back to python urllib.
  if command -v curl >/dev/null 2>&1; then
    curl -sS --connect-timeout 5 --max-time 240 -f "$url"
    return $?
  fi
  python3 - "$url" <<'PY'
import sys, urllib.request
url = sys.argv[1]
with urllib.request.urlopen(url, timeout=240) as r:
    sys.stdout.write(r.read().decode("utf-8", errors="replace"))
PY
}

for base in "${PROXY_URLS[@]}"; do
  [[ -n "$base" ]] || continue
  if call_proxy "$base"; then
    exit 0
  fi
  echo "[warn] heal proxy unreachable at $base; trying next/fallback" >&2
done

# Fallback: direct local script (works via kubectl exec, not via OpenShell policy)
pick_newest() {
  local pattern="$1"
  find /sandbox /opt/openclaw/workspace . \
    -path "$pattern" ! -path '*/.openclaw/*' 2>/dev/null \
    | while read -r p; do
        [[ -f "$p" ]] || continue
        printf '%s\t%s\n' "$(stat -c %Y "$p" 2>/dev/null || stat -f %m "$p" 2>/dev/null || echo 0)" "$p"
      done \
    | sort -nr \
    | head -1 \
    | cut -f2-
}

HEAL="$(pick_newest '*/skills/netobserv-heal/scripts/netobserv-cluster-heal.py')"
if [[ -z "$HEAL" ]]; then
  HEAL="$(find /sandbox /opt/openclaw/workspace . -name netobserv-cluster-heal.py ! -path '*/.openclaw/*' 2>/dev/null | head -1 || true)"
fi
KC="$(pick_newest '*/.kube/config')"
if [[ -z "$KC" ]]; then
  KC="$(find /sandbox /opt/openclaw/workspace . -path '*/.kube/config' ! -path '*/.openclaw/*' 2>/dev/null | head -1 || true)"
fi
if [[ -z "$HEAL" ]]; then
  echo "ERROR: netobserv-cluster-heal.py not found and heal proxy unreachable" >&2
  exit 1
fi
if [[ -n "$KC" ]]; then
  export KUBECONFIG="$KC"
fi
echo "HEAL=$HEAL (direct fallback)"
echo "KUBECONFIG=${KUBECONFIG:-}"
exec python3 "$HEAL" "$CMD"
