#!/usr/bin/env bash
# Install Ansible Automation Platform 2.7 operator + instance on OpenShift.
#
# Usage:
#   ./scripts/install-aap.sh status
#   ./scripts/install-aap.sh install
#   ./scripts/install-aap.sh uninstall
#
# Env:
#   AAP_CHANNEL=stable-2.7       operator subscription channel (default stable-2.7)
#   AAP_NS=ansible-automation-platform
#   AAP_INSTANCE=netobserv-aap
#   AAP_MINIMAL=1              use controller-only CR (default 1 — lighter for lab)
#   AAP_STORAGE_CLASS=         required when AAP_MINIMAL=0 (Hub RWX)
#   WAIT_AAP_SEC=3600
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFESTS="$ROOT/manifests/aap"
KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-status}"

AAP_NS="${AAP_NS:-ansible-automation-platform}"
AAP_CHANNEL="${AAP_CHANNEL:-stable-2.7}"
AAP_INSTANCE="${AAP_INSTANCE:-netobserv-aap}"
AAP_MINIMAL="${AAP_MINIMAL:-1}"
AAP_STORAGE_CLASS="${AAP_STORAGE_CLASS:-}"
WAIT_AAP_SEC="${WAIT_AAP_SEC:-3600}"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -n "$KUBECTL" ]] || die "oc/kubectl required"

aap_gateway_url() {
  local host
  host="$("$KUBECTL" -n "$AAP_NS" get route -l app.kubernetes.io/name="$AAP_INSTANCE" -o jsonpath='{.items[0].spec.host}' 2>/dev/null || true)"
  if [[ -z "$host" ]]; then
    host="$("$KUBECTL" -n "$AAP_NS" get route -o jsonpath='{.items[0].spec.host}' 2>/dev/null || true)"
  fi
  [[ -n "$host" ]] && printf 'https://%s' "$host"
}

aap_admin_password() {
  local secret pass
  for secret in \
    "netobserv-aap-admin-password" \
    "${AAP_INSTANCE}-admin-password" \
    "${AAP_INSTANCE}-gateway-admin-password" \
    "gateway-admin-password" \
    "${AAP_INSTANCE}-controller-admin-password"; do
    pass="$("$KUBECTL" -n "$AAP_NS" get secret "$secret" -o jsonpath='{.data.password}' 2>/dev/null | base64 -d 2>/dev/null || true)"
    if [[ -n "$pass" ]]; then
      printf '%s' "$pass"
      return 0
    fi
  done
  return 1
}

ensure_demo_admin_secret() {
  if "$KUBECTL" -n "$AAP_NS" get secret netobserv-aap-admin-password >/dev/null 2>&1; then
    return 0
  fi
  local pass
  pass="$(aap_admin_password)" || return 0
  "$KUBECTL" -n "$AAP_NS" create secret generic netobserv-aap-admin-password \
    --from-literal=password="$pass" \
    --dry-run=client -o yaml | "$KUBECTL" apply -f -
  ok "secret/netobserv-aap-admin-password (demo alias for wire scripts)"
}

aap_admin_password_hint() {
  if "$KUBECTL" -n "$AAP_NS" get secret netobserv-aap-admin-password >/dev/null 2>&1; then
    ok "Admin password: secret/netobserv-aap-admin-password (oc get secret -n $AAP_NS)"
    return 0
  fi
  local secret pass
  for secret in \
    "${AAP_INSTANCE}-admin-password" \
    "${AAP_INSTANCE}-gateway-admin-password" \
    "gateway-admin-password" \
    "${AAP_INSTANCE}-controller-admin-password"; do
    pass="$("$KUBECTL" -n "$AAP_NS" get secret "$secret" -o jsonpath='{.data.password}' 2>/dev/null | base64 -d 2>/dev/null || true)"
    if [[ -n "$pass" ]]; then
      printf 'secret/%s password available (not printed — use oc get secret)\n' "$secret"
      return 0
    fi
  done
  warn "Admin password secret not found yet — check AAP operator docs / gateway UI"
}

wait_aap_csv() {
  step "Wait AAP operator CSV (timeout ${WAIT_AAP_SEC}s)"
  local end=$((SECONDS + WAIT_AAP_SEC))
  while (( SECONDS < end )); do
    local phase csv
    csv="$("$KUBECTL" get csv -n "$AAP_NS" -o json 2>/dev/null | python3 -c "
import json,sys
for item in json.load(sys.stdin).get('items',[]):
    name=item.get('metadata',{}).get('name','')
    if 'aap-operator' in name or 'ansible-automation-platform' in name:
        print(name); break
" 2>/dev/null || true)"
    if [[ -n "$csv" ]]; then
      phase="$("$KUBECTL" get csv "$csv" -n "$AAP_NS" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
      if [[ "$phase" == "Succeeded" ]]; then
        ok "CSV $csv Succeeded"
        return 0
      fi
      printf '  waiting csv=%s phase=%s\n' "$csv" "${phase:-Pending}"
    else
      printf '  waiting for AAP operator CSV...\n'
    fi
    sleep 20
  done
  warn "AAP operator CSV not Succeeded within ${WAIT_AAP_SEC}s"
  return 1
}

aap_instance_ready() {
  local successful failure
  successful="$("$KUBECTL" get ansibleautomationplatform "$AAP_INSTANCE" -n "$AAP_NS" \
    -o jsonpath='{.status.conditions[?(@.type=="Successful")].status}' 2>/dev/null || true)"
  failure="$("$KUBECTL" get ansibleautomationplatform "$AAP_INSTANCE" -n "$AAP_NS" \
    -o jsonpath='{.status.conditions[?(@.type=="Failure")].status}' 2>/dev/null || true)"
  [[ "$successful" == "True" && "$failure" != "True" ]]
}

wait_aap_instance() {
  step "Wait AnsibleAutomationPlatform/$AAP_INSTANCE"
  local end=$((SECONDS + WAIT_AAP_SEC))
  while (( SECONDS < end )); do
    local successful failure running
    successful="$("$KUBECTL" get ansibleautomationplatform "$AAP_INSTANCE" -n "$AAP_NS" \
      -o jsonpath='{.status.conditions[?(@.type=="Successful")].status}' 2>/dev/null || true)"
    failure="$("$KUBECTL" get ansibleautomationplatform "$AAP_INSTANCE" -n "$AAP_NS" \
      -o jsonpath='{.status.conditions[?(@.type=="Failure")].status}' 2>/dev/null || true)"
    running="$("$KUBECTL" get ansibleautomationplatform "$AAP_INSTANCE" -n "$AAP_NS" \
      -o jsonpath='{.status.conditions[?(@.type=="Running")].status}' 2>/dev/null || true)"
    if aap_instance_ready; then
      ok "AnsibleAutomationPlatform/$AAP_INSTANCE Successful (AAP 2.7)"
      return 0
    fi
    if [[ "$failure" == "True" ]]; then
      warn "AAP instance Failure=True — oc describe ansibleautomationplatform $AAP_INSTANCE -n $AAP_NS"
      return 1
    fi
    printf '  waiting AAP instance Successful=%s Running=%s\n' "${successful:-?}" "${running:-?}"
    sleep 30
  done
  warn "AAP instance not Successful within ${WAIT_AAP_SEC}s"
  "$KUBECTL" get ansibleautomationplatform "$AAP_INSTANCE" -n "$AAP_NS" -o yaml 2>/dev/null | tail -30 || true
  return 1
}

cmd_status() {
  step "AAP operator"
  "$KUBECTL" get sub,csv -n "$AAP_NS" 2>/dev/null || warn "namespace $AAP_NS not found"
  step "AAP instance"
  "$KUBECTL" get ansibleautomationplatform -n "$AAP_NS" 2>/dev/null || true
  "$KUBECTL" get pods -n "$AAP_NS" 2>/dev/null | head -20 || true
  local url
  url="$(aap_gateway_url || true)"
  if [[ -n "$url" ]]; then
    ok "Gateway URL: $url"
  else
    warn "No AAP route yet"
  fi
  aap_admin_password_hint || true
  step "NetObserv heal RBAC (AAP execution SA)"
  "$KUBECTL" get sa aap-netobserv-heal -n openclaw 2>/dev/null || warn "aap-netobserv-heal SA not applied"
}

cmd_install() {
  step "Apply AAP operator subscription (channel=$AAP_CHANNEL)"
  sed "s/channel: stable-2.7/channel: ${AAP_CHANNEL}/" \
    "$MANIFESTS/01-operator-subscription.yaml" | "$KUBECTL" apply -f -
  wait_aap_csv || true

  step "Apply AAP execution RBAC (aap-netobserv-heal)"
  "$KUBECTL" apply -f "$MANIFESTS/03-aap-netobserv-heal-rbac.yaml"

  if [[ "$AAP_MINIMAL" == "1" ]]; then
    step "Deploy minimal AAP instance (controller + gateway; hub/eda disabled)"
    "$KUBECTL" apply -f "$MANIFESTS/02-aap-instance-minimal.yaml"
  else
    [[ -n "$AAP_STORAGE_CLASS" ]] || die "AAP_MINIMAL=0 requires AAP_STORAGE_CLASS (RWX for Hub)"
    step "Deploy AAP instance with Hub (storage class=$AAP_STORAGE_CLASS)"
    sed "s/REPLACE_RWX_STORAGE_CLASS/${AAP_STORAGE_CLASS}/g" \
      "$MANIFESTS/02-aap-instance-hub.yaml" | "$KUBECTL" apply -f -
  fi

  wait_aap_instance || true
  ensure_demo_admin_secret || true
  cmd_status
  cat <<EOF

Next steps:
  1. Open the Gateway URL above and apply your AAP subscription license.
  2. ./scripts/wire-openclaw-aap.sh all
  3. ENABLE_ANSIBLE_MCP=1 ./scripts/seed-openclaw-netobserv-skills.sh
  4. ./scripts/netobserv-e2e-openclaw-test.sh aap-check

See docs/AAP-HEAL-DESIGN.md and docs/AAP-PRESENTER-GUIDE.md
EOF
}

cmd_uninstall() {
  warn "Removing AAP instance and operator from $AAP_NS (RBAC in openclaw kept)"
  "$KUBECTL" delete ansibleautomationplatform "$AAP_INSTANCE" -n "$AAP_NS" --ignore-not-found
  "$KUBECTL" delete -f "$MANIFESTS/01-operator-subscription.yaml" --ignore-not-found
  ok "AAP uninstall requested (namespace may remain until manually deleted)"
}

case "$CMD" in
  status) cmd_status ;;
  install) cmd_install ;;
  uninstall) cmd_uninstall ;;
  *) die "usage: $0 {status|install|uninstall}" ;;
esac
