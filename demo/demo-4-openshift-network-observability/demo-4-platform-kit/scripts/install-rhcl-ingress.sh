#!/usr/bin/env bash
# Install Red Hat Connectivity Link (Kuadrant) and Phase A′ ingress for OpenClaw.
#
# Option A (default): dedicated openclaw-gateway + Authorino AuthPolicies (OpenShift OAuth).
# Manifests include Kuadrant OIDCPolicy to seed policies; patch-openclaw-oidc-openshift.sh
# applies OpenShift fixes and detaches standalone AuthPolicies.
# Does NOT attach OpenClaw to the RHOAI data-science-gateway (avoids double auth / wrong cookies).
#
# Usage:
#   ./scripts/install-rhcl-ingress.sh status
#   ./scripts/install-rhcl-ingress.sh install          # RHCL operator + Kuadrant CR
#   ./scripts/install-rhcl-ingress.sh ingress         # dedicated gateway + OIDC (OpenShift OAuth patch)
#   ./scripts/install-rhcl-ingress.sh all             # install + ingress
#
# Env:
#   OPENCLAW_RHCL_HOST           override hostname (default openclaw-rhcl.apps.<ingress>)
#   OPENCLAW_NS=openclaw
#   PATCH_OPENCLAW_ORIGIN=1      run ensure-openclaw-ui-origin.sh after ingress (default 1)
#   OAUTH_CLIENT_SECRET=         reuse secret (default: keep existing OAuthClient or generate)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFESTS="$ROOT/manifests/platform-merge/rhcl"
INGRESS_MANIFESTS="$MANIFESTS/ingress"
KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-all}"
RHCL_CSV="${RHCL_CSV:-rhcl-operator.v1.4.2}"
WAIT_RHCL_SEC="${WAIT_RHCL_SEC:-900}"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
PATCH_OPENCLAW_ORIGIN="${PATCH_OPENCLAW_ORIGIN:-1}"
OAUTH_CLIENT_NAME="${OAUTH_CLIENT_NAME:-openclaw-rhcl}"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -n "$KUBECTL" ]] || die "oc/kubectl required"

resolve_ingress_domain() {
  "$KUBECTL" get ingresscontroller default -n openshift-ingress-operator \
    -o jsonpath='{.status.domain}' 2>/dev/null || true
}

resolve_openclaw_rhcl_host() {
  if [[ -n "${OPENCLAW_RHCL_HOST:-}" ]]; then
    echo "$OPENCLAW_RHCL_HOST"
    return
  fi
  local domain
  domain="$(resolve_ingress_domain)"
  [[ -n "$domain" ]] || die "could not resolve cluster ingress domain"
  echo "openclaw-rhcl.${domain}"
}

resolve_oauth_client_secret() {
  if [[ -n "${OAUTH_CLIENT_SECRET:-}" ]]; then
    echo "$OAUTH_CLIENT_SECRET"
    return
  fi
  local existing
  existing="$("$KUBECTL" get oauthclient "$OAUTH_CLIENT_NAME" -o jsonpath='{.secret}' 2>/dev/null || true)"
  if [[ -n "$existing" ]]; then
    echo "$existing"
    return
  fi
  openssl rand -base64 24 | tr -d "=+/" | cut -c1-32
}

cleanup_legacy_shared_ingress() {
  step "Remove legacy OpenClaw HTTPRoute on RHOAI data-science-gateway (if present)"
  "$KUBECTL" delete httproute openclaw-ui -n openshift-ingress --ignore-not-found
  "$KUBECTL" delete route openclaw-rhcl -n openshift-ingress --ignore-not-found
  "$KUBECTL" delete authpolicy openclaw-ui-allow -n openshift-ingress --ignore-not-found
  "$KUBECTL" delete ratelimitpolicy openclaw-ui-demo-limit -n openshift-ingress --ignore-not-found
  "$KUBECTL" delete referencegrant allow-httproute-from-openshift-ingress -n "$OPENCLAW_NS" --ignore-not-found 2>/dev/null || true
}

print_status() {
  step "RHCL control plane"
  "$KUBECTL" get sub,csv -n kuadrant-system 2>/dev/null || true
  "$KUBECTL" get kuadrant -n kuadrant-system 2>/dev/null || true
  "$KUBECTL" get pods -n kuadrant-system -l app.kubernetes.io/name=authorino 2>/dev/null || true

  step "Dedicated OpenClaw RHCL gateway (Option A)"
  "$KUBECTL" get gateway openclaw-gateway -n "$OPENCLAW_NS" 2>/dev/null || \
    warn "Gateway openclaw-gateway not applied — run: $0 ingress"
  "$KUBECTL" get httproute openclaw-ui,openclaw-ui-static,openclaw-ui-oidc-callback -n "$OPENCLAW_NS" 2>/dev/null || true
  "$KUBECTL" get route openclaw-rhcl -n "$OPENCLAW_NS" 2>/dev/null || true
  "$KUBECTL" get authpolicy,oidcpolicy,ratelimitpolicy -n "$OPENCLAW_NS" 2>/dev/null | grep -E 'openclaw|NAME' || true
  "$KUBECTL" get oauthclient "$OAUTH_CLIENT_NAME" 2>/dev/null || \
    warn "OAuthClient $OAUTH_CLIENT_NAME missing"

  step "RHOAI data-science-gateway (MLflow only — OpenClaw not attached)"
  "$KUBECTL" get gateway data-science-gateway -n openshift-ingress 2>/dev/null || \
    warn "data-science-gateway not found (install RHOAI for MLflow)"

  local host legacy domain
  host="$(resolve_openclaw_rhcl_host 2>/dev/null || true)"
  domain="$(resolve_ingress_domain 2>/dev/null || true)"
  legacy="$("$KUBECTL" -n "$OPENCLAW_NS" get route openclaw -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  [[ -n "$host" ]] && ok "RHCL OpenClaw URL (Authorino OIDC): https://${host}/"
  [[ -n "$legacy" ]] && ok "Legacy OpenClaw Route (lab token, no RHCL): https://${legacy}/"
  [[ -n "$domain" ]] && ok "RHOAI MLflow: https://rh-ai.${domain}/mlflow"
}

install_rhcl() {
  step "Apply RHCL operator subscription"
  "$KUBECTL" apply -f "$MANIFESTS/00-namespace-kuadrant-system.yaml"
  "$KUBECTL" apply -f "$MANIFESTS/01-rhcl-operatorgroup.yaml"
  "$KUBECTL" apply -f "$MANIFESTS/02-rhcl-subscription.yaml"

  step "Wait RHCL CSV (timeout ${WAIT_RHCL_SEC}s)"
  local end=$((SECONDS + WAIT_RHCL_SEC))
  while (( SECONDS < end )); do
    local phase
    phase="$("$KUBECTL" get csv "$RHCL_CSV" -n kuadrant-system -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    if [[ "$phase" == "Succeeded" ]]; then
      ok "CSV $RHCL_CSV Succeeded"
      break
    fi
    sleep 15
  done

  step "Apply Kuadrant CR"
  "$KUBECTL" apply -f "$MANIFESTS/03-kuadrant-cr.yaml"
  "$KUBECTL" wait kuadrant/kuadrant --for=condition=Ready=true -n kuadrant-system --timeout=600s 2>/dev/null || \
    warn "Kuadrant Ready condition not met within 600s — check oc get kuadrant -n kuadrant-system -o yaml"
}

install_ingress() {
  local host domain oauth_secret tmpdir api_server oauth_jwks_url
  host="$(resolve_openclaw_rhcl_host)"
  domain="$(resolve_ingress_domain)"
  [[ -n "$domain" ]] || die "could not resolve cluster ingress domain"
  oauth_secret="$(resolve_oauth_client_secret)"
  api_server="$("$KUBECTL" whoami --show-server 2>/dev/null || true)"
  [[ -n "$api_server" ]] || die "could not resolve API server URL for JWKS"
  oauth_jwks_url="${api_server%/}/openid/v1/jwks"

  "$KUBECTL" get svc openclaw -n "$OPENCLAW_NS" >/dev/null 2>&1 || \
    die "service/openclaw not found in $OPENCLAW_NS"

  cleanup_legacy_shared_ingress

  step "Apply dedicated OpenClaw gateway + Authorino OIDC manifests (host=$host)"
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' RETURN

  for f in "$INGRESS_MANIFESTS"/*.yaml; do
    [[ "$(basename "$f")" == "kustomization.yaml" ]] && continue
    # Merged into patch-openclaw-oidc-openshift.sh on openclaw-ui-oidc (conflicts if both target openclaw-ui).
    [[ "$(basename "$f")" == "09-authpolicy-openclaw-ui-authorize.yaml" ]] && continue
    sed -e "s|PLACEHOLDER_OPENCLAW_RHCL_HOST|${host}|g" \
        -e "s|PLACEHOLDER_INGRESS_DOMAIN|${domain}|g" \
        -e "s|PLACEHOLDER_OAUTH_CLIENT_SECRET|${oauth_secret}|g" \
        -e "s|PLACEHOLDER_OAUTH_JWKS_URL|${oauth_jwks_url}|g" \
        "$f" >"$tmpdir/$(basename "$f")"
  done

  "$KUBECTL" apply -f "$tmpdir"

  step "Patch Authorino AuthPolicies for OpenShift OAuth (scope, token exchange, opaque session)"
  sleep 5
  OAUTH_CLIENT_SECRET="$oauth_secret" OAUTH_JWKS_URL="$oauth_jwks_url" \
    "$ROOT/scripts/patch-openclaw-oidc-openshift.sh" || \
    warn "OpenShift OIDC patch failed — check scripts/patch-openclaw-oidc-openshift.sh"

  step "Wait Gateway + HTTPRoute programming"
  local end=$((SECONDS + 180)) gw_ok="" hr_ok=""
  while (( SECONDS < end )); do
    gw_ok="$("$KUBECTL" get gateway openclaw-gateway -n "$OPENCLAW_NS" \
      -o jsonpath='{.status.listeners[?(@.name=="http")].conditions[?(@.type=="Programmed")].status}' 2>/dev/null || true)"
    hr_ok="$("$KUBECTL" get httproute openclaw-ui -n "$OPENCLAW_NS" \
      -o jsonpath='{.status.parents[0].conditions[?(@.type=="Accepted")].status}' 2>/dev/null || true)"
    [[ "$gw_ok" == "True" && "$hr_ok" == "True" ]] && break
    sleep 5
  done
  [[ "$gw_ok" == "True" ]] && ok "Gateway openclaw-gateway http listener programmed" || \
    warn "Gateway not programmed — oc describe gateway openclaw-gateway -n $OPENCLAW_NS"
  [[ "$hr_ok" == "True" ]] && ok "HTTPRoute openclaw-ui accepted" || \
    warn "HTTPRoute not Accepted — oc describe httproute openclaw-ui -n $OPENCLAW_NS"

  step "Wait Authorino AuthPolicies enforced (post OpenShift patch)"
  end=$((SECONDS + 120))
  local ap_enforced=""
  while (( SECONDS < end )); do
    ap_enforced="$("$KUBECTL" get authpolicy openclaw-ui-oidc,openclaw-ui-oidc-callback -n "$OPENCLAW_NS" \
      -o jsonpath='{range .items[*]}{.metadata.name}{"="}{.status.conditions[?(@.type=="Enforced")].status}{"\n"}{end}' 2>/dev/null || true)"
    if echo "$ap_enforced" | grep -q 'openclaw-ui-oidc=True' && \
       echo "$ap_enforced" | grep -q 'openclaw-ui-oidc-callback=True'; then
      break
    fi
    sleep 5
  done
  echo "$ap_enforced" | grep -q 'openclaw-ui-oidc=True' && \
    echo "$ap_enforced" | grep -q 'openclaw-ui-oidc-callback=True' && \
    ok "Authorino AuthPolicies enforced on UI + callback routes" || \
    warn "AuthPolicies not Enforced yet — oc get authpolicy -n $OPENCLAW_NS"

  if [[ "$PATCH_OPENCLAW_ORIGIN" == "1" && -x "$ROOT/scripts/ensure-openclaw-ui-origin.sh" ]]; then
    step "Allow RHCL hostname in OpenClaw controlUi.allowedOrigins"
    ORIGIN="https://${host}" "$ROOT/scripts/ensure-openclaw-ui-origin.sh" || \
      warn "ensure-openclaw-ui-origin failed — add https://${host} manually"
  fi

  ok "OpenClaw RHCL ingress: https://${host}/"
  cat <<EOF

Presenter notes (Option A — dedicated gateway + Authorino):
  - RHCL URL: https://${host}/  (302 → OpenShift OAuth client ${OAUTH_CLIENT_NAME}, then Control UI)
  - Zero-trust: Gateway deny-all + patched Authorino AuthPolicies + RateLimitPolicy
  - OpenShift OAuth patch: scripts/patch-openclaw-oidc-openshift.sh (also run automatically above)
  - Callback HTTPRoute must be openclaw-ui-oidc-callback (matches AuthPolicy target)
  - Control UI logo: HTTPRoute openclaw-ui-static serves /apple-touch-icon.png (no OAuth redirect)
  - Legacy lab path (gateway token): oc get route openclaw -n ${OPENCLAW_NS}
  - MLflow stays on RHOAI gateway: https://rh-ai.${domain}/mlflow

Verify: $0 status
EOF
}

case "$CMD" in
  status) print_status ;;
  install) install_rhcl; print_status ;;
  ingress) install_ingress; print_status ;;
  all)
    install_rhcl
    install_ingress
    print_status
    ;;
  *)
    die "usage: $0 [status|install|ingress|all]"
    ;;
esac
