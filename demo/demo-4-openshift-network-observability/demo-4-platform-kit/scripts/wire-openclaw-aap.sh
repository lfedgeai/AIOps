#!/usr/bin/env bash
# Wire Ansible Automation Platform + ansible-automation MCP for governed tasks.
#
# Usage:
#   AAP_ADMIN_PASSWORD=   optional — auto-read from netobserv-aap-admin-password secret when unset
#   ./scripts/wire-openclaw-aap.sh all
#   ./scripts/wire-openclaw-aap.sh credentials   # refresh openshift SA token + re-bootstrap templates
#   ./scripts/wire-openclaw-aap.sh bootstrap     # Gitea seed + org, project, templates, launcher token
#   ./scripts/wire-openclaw-aap.sh gitea         # install Gitea + push playbooks only
#   ./scripts/wire-openclaw-aap.sh ansible-mcp   # deploy ansible-mcp + patch OpenClaw config
#   ./scripts/wire-openclaw-aap.sh status
#
# Env:
#   AAP_NS=ansible-automation-platform
#   AAP_CONTROLLER_URL=https://<gateway-host>
#   AAP_ADMIN_PASSWORD=
#   OPENCLAW_NS=openclaw
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-status}"

AAP_NS="${AAP_NS:-ansible-automation-platform}"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
GITEA_NS="${GITEA_NS:-gitea}"
GITEA_ADMIN_USER="${GITEA_ADMIN_USER:-netobserv}"
GITEA_REPO_NAME="${GITEA_REPO_NAME:-netobserv-heal}"
GITEA_SECRET="${GITEA_SECRET:-gitea-admin-credentials}"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -n "$KUBECTL" ]] || die "oc/kubectl required"

load_aap_admin_password() {
  [[ -n "${AAP_ADMIN_PASSWORD:-}" ]] && return 0
  if "$KUBECTL" -n "$AAP_NS" get secret netobserv-aap-admin-password >/dev/null 2>&1; then
    AAP_ADMIN_PASSWORD="$("$KUBECTL" -n "$AAP_NS" get secret netobserv-aap-admin-password \
      -o jsonpath='{.data.password}' | base64 -d)"
    export AAP_ADMIN_PASSWORD
    return 0
  fi
  # Fall back to operator-created gateway secrets (install-aap.sh also mirrors to netobserv-aap-admin-password)
  local secret pass
  for secret in \
    "${AAP_INSTANCE:-netobserv-aap}-gateway-admin-password" \
    "${AAP_INSTANCE:-netobserv-aap}-admin-password" \
    "gateway-admin-password"; do
    pass="$("$KUBECTL" -n "$AAP_NS" get secret "$secret" -o jsonpath='{.data.password}' 2>/dev/null | base64 -d 2>/dev/null || true)"
    if [[ -n "$pass" ]]; then
      AAP_ADMIN_PASSWORD="$pass"
      export AAP_ADMIN_PASSWORD
      return 0
    fi
  done
}

check_aap_job_templates() {
  step "AAP job templates (confirm gate + openshift credential)"
  if ! "$KUBECTL" -n "$OPENCLAW_NS" get secret openclaw-aap-launcher >/dev/null 2>&1; then
    warn "openclaw-aap-launcher missing — skip template check"
    return 0
  fi
  local token url
  token="$("$KUBECTL" -n "$OPENCLAW_NS" get secret openclaw-aap-launcher -o jsonpath='{.data.token}' | base64 -d)"
  url="$("$KUBECTL" -n "$OPENCLAW_NS" get secret openclaw-aap-launcher -o jsonpath='{.data.controller_url}' | base64 -d)"
  if ! python3 - "$url" "$token" <<'PY'
import json, ssl, sys, urllib.parse, urllib.request

base, token = sys.argv[1:3]
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def get(path):
    req = urllib.request.Request(
        f"{base.rstrip('/')}/api/controller/v2{path}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        return json.loads(resp.read().decode())

failed = False
for name in ("netobserv-heal-db-path", "netobserv-restore-policy"):
    data = get(f"/job_templates/?name={urllib.parse.quote(name)}")
    rows = data.get("results") or []
    if not rows:
        print(f"[fail] job template missing: {name}")
        failed = True
        continue
    jt = rows[0]
    jt_id = jt["id"]
    ask = jt.get("ask_variables_on_launch")
    extra_raw = jt.get("extra_vars") or ""
    extra = {}
    if extra_raw:
        try:
            extra = json.loads(extra_raw) if isinstance(extra_raw, str) else extra_raw
        except json.JSONDecodeError:
            extra = {}
    creds = get(f"/job_templates/{jt_id}/credentials/")
    cred_count = int(creds.get("count") or 0)
    if not ask:
        print(f"[fail] {name}: ask_variables_on_launch must be true (MCP passes confirmed at launch)")
        failed = True
    if extra.get("confirmed") is False:
        print(f"[fail] {name}: extra_vars pins confirmed=false — re-run wire-openclaw-aap.sh bootstrap")
        failed = True
    if cred_count < 1:
        print(f"[fail] {name}: netobserv-openshift credential not attached — re-run bootstrap")
        failed = True
    if ask and extra.get("confirmed") is not False and cred_count >= 1:
        print(f"[ ok ] {name}: launch vars + openshift credential")
if failed:
    sys.exit(1)
PY
  then
    warn "AAP job template check failed — export AAP_ADMIN_PASSWORD=... && ./scripts/wire-openclaw-aap.sh bootstrap"
    return 1
  fi
  return 0
}

discover_aap_url() {
  if [[ -n "${AAP_GATEWAY_URL:-}" ]]; then
    printf '%s' "${AAP_GATEWAY_URL%/}"
    return
  fi
  if [[ -n "${AAP_CONTROLLER_URL:-}" ]]; then
    printf '%s' "${AAP_CONTROLLER_URL%/}"
    return
  fi
  local host
  host="$("$KUBECTL" -n "$AAP_NS" get route -o jsonpath='{.items[0].spec.host}' 2>/dev/null || true)"
  [[ -n "$host" ]] && printf 'https://%s' "$host"
}

load_gitea_env() {
  export GITEA_GIT_USER="${GITEA_GIT_USER:-$GITEA_ADMIN_USER}"
  export GITEA_GIT_BRANCH="${GITEA_GIT_BRANCH:-main}"
  export GITEA_REPO_URL="${GITEA_REPO_URL:-http://gitea.${GITEA_NS}.svc.cluster.local:3000/${GITEA_ADMIN_USER}/${GITEA_REPO_NAME}.git}"
  if [[ -z "${GITEA_GIT_PASSWORD:-}" ]] && "$KUBECTL" -n "$GITEA_NS" get secret "$GITEA_SECRET" >/dev/null 2>&1; then
    export GITEA_GIT_PASSWORD="$("$KUBECTL" -n "$GITEA_NS" get secret "$GITEA_SECRET" \
      -o jsonpath='{.data.password}' | base64 -d)"
  fi
  [[ -n "${GITEA_GIT_PASSWORD:-}" ]] || die "GITEA git password missing — run: ./scripts/install-gitea.sh install"
}

cmd_status() {
  step "Gitea"
  "$ROOT/scripts/install-gitea.sh" status 2>/dev/null || warn "Gitea not installed"
  step "AAP"
  "$ROOT/scripts/install-aap.sh" status
  step "Ansible MCP"
  "$KUBECTL" -n "$OPENCLAW_NS" get deploy ansible-mcp 2>/dev/null || warn "ansible-mcp not deployed"
  "$KUBECTL" -n "$OPENCLAW_NS" get secret openclaw-aap-launcher 2>/dev/null && ok "openclaw-aap-launcher secret present" \
    || warn "openclaw-aap-launcher secret missing — run bootstrap"
  check_aap_job_templates || true
}

cmd_gitea() {
  if ! "$KUBECTL" -n "$GITEA_NS" get statefulset gitea >/dev/null 2>&1; then
    step "Install Gitea (kwkoo/gitea-openshift)"
    "$ROOT/scripts/install-gitea.sh" install
  else
    ok "Gitea already installed"
  fi
  step "Push playbooks to Gitea"
  chmod +x "$ROOT/scripts/seed-gitea-netobserv-playbooks.sh"
  "$ROOT/scripts/seed-gitea-netobserv-playbooks.sh" seed
}

cmd_bootstrap() {
  load_aap_admin_password
  [[ -n "${AAP_ADMIN_PASSWORD:-}" ]] || die "AAP_ADMIN_PASSWORD required for bootstrap (or install netobserv-aap-admin-password secret)"
  cmd_gitea
  load_gitea_env
  local url
  url="$(discover_aap_url)"
  [[ -n "$url" ]] || die "AAP route not found — run install-aap.sh install first"
  export AAP_CONTROLLER_URL="$url"
  export KUBECTL
  step "Bootstrap AAP org, git project, credentials, job templates"
  local out
  out="$(python3 "$ROOT/scripts/aap-bootstrap.py")"
  local token gateway
  token="$(printf '%s' "$out" | python3 -c "import json,sys; print(json.load(sys.stdin)['launcher_token'])")"
  gateway="$(printf '%s' "$out" | python3 -c "import json,sys; print(json.load(sys.stdin)['gateway_url'])")"
  step "Store launcher token in openclaw namespace"
  "$KUBECTL" -n "$OPENCLAW_NS" create secret generic openclaw-aap-launcher \
    --from-literal=token="$token" \
    --from-literal=controller_url="$gateway" \
    --dry-run=client -o yaml | "$KUBECTL" apply -f -
  ok "openclaw-aap-launcher secret applied"
  printf '%s\n' "$out" | python3 -m json.tool 2>/dev/null || printf '%s\n' "$out"
}

cmd_ansible_mcp() {
  [[ "$("$KUBECTL" -n "$OPENCLAW_NS" get secret openclaw-aap-launcher -o name 2>/dev/null)" ]] \
    || die "openclaw-aap-launcher missing — run: $0 bootstrap"
  step "Deploy ansible-automation MCP server"
  "$KUBECTL" -n "$OPENCLAW_NS" create configmap ansible-mcp-scripts \
    --from-file=server.py="$ROOT/openclaw-skills/ansible-mcp-server/server.py" \
    --from-file=aap_client.py="$ROOT/openclaw-skills/ansible-mcp-server/aap_client.py" \
    --from-file=requirements.txt="$ROOT/openclaw-skills/ansible-mcp-server/requirements.txt" \
    --dry-run=client -o yaml | "$KUBECTL" apply -f -
  "$KUBECTL" apply -f "$ROOT/openclaw-skills/manifests/ansible-mcp.yaml"
  "$KUBECTL" -n "$OPENCLAW_NS" rollout restart deploy/ansible-mcp 2>/dev/null || true
  "$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy/ansible-mcp --timeout=180s
  ok "ansible-mcp deployed"
  step "Enable ansible-automation MCP in OpenClaw gateway config"
  ENABLE_ANSIBLE_MCP=1 CONFIG_ONLY=1 OPENCLAW_NS="$OPENCLAW_NS" "$ROOT/scripts/seed-openclaw-netobserv-skills.sh" \
    || warn "seed config-only failed — run seed-openclaw-netobserv-skills.sh with ENABLE_ANSIBLE_MCP=1"
}

cmd_credentials() {
  load_aap_admin_password
  [[ -n "${AAP_ADMIN_PASSWORD:-}" ]] || die "AAP_ADMIN_PASSWORD required to refresh credential"
  cmd_bootstrap
}

cmd_all() {
  "$KUBECTL" apply -f "$ROOT/manifests/aap/03-aap-netobserv-heal-rbac.yaml"
  cmd_bootstrap
  cmd_ansible_mcp
  ok "AAP wire complete — verify: ./scripts/netobserv-e2e-openclaw-test.sh aap-check"
}

case "$CMD" in
  status) cmd_status ;;
  gitea) cmd_gitea ;;
  bootstrap) cmd_bootstrap ;;
  credentials) cmd_credentials ;;
  ansible-mcp) cmd_ansible_mcp ;;
  mcp) cmd_ansible_mcp ;;  # legacy alias
  all) cmd_all ;;
  playbooks)
    warn "playbooks subcommand removed — use: $0 gitea"
    cmd_gitea
    ;;
  *) die "usage: $0 {status|gitea|bootstrap|credentials|ansible-mcp|all}" ;;
esac
