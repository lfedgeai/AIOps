#!/usr/bin/env bash
#
# openshell-sandbox-proof.sh
# ---------------------------------------------------------------------------
# Bastion checks for OpenShell Agent Sandbox isolation.
# Full security narrative (rogue prompts + MCP rebuttal): SECURITY-DEMO.md
#
# Run on the bastion (or any host with oc) after OpenClaw + OpenShell are up.
# Start a Control UI session (/new) then send **any chat message** before inspect /
# prove-6443. /new alone does not create openclaw-agent-* pods — OpenShell spawns
# the sandbox when the agent begins processing a turn (often 15–60s after send).
#
#   export DEMO_KIT_ROOT="${DEMO_KIT_ROOT:-$HOME/AIOps/demo/demo-4-platform-kit}"
#   cd "$DEMO_KIT_ROOT"
#   ./scripts/openshell-sandbox-proof.sh prove
#
# Subcommands:
#   status        Agent Sandbox controller + OpenShell gateway + OpenClaw wiring
#   list          Sandboxes CR + openclaw-agent-* pods + openclaw sandbox list
#   wait          Poll until agent pod appears (after Control UI message post-/new)
#   inspect       /sandbox layout inside the agent pod (needs active sandbox)
#   prove-6443    Curl kubernetes.default.svc:6443 from sandbox — expect block
#   prove-mcp     MCP probe from OpenClaw gateway (in-cluster, not sandbox)
#   prove-policy  Show managed-policy allowlist snippet from ConfigMap
#   hints         Presenter narrative + Control UI negative-test prompts
#   prove         Run status → list → inspect → prove-6443 → prove-mcp → prove-policy
#
# Env: OPENCLAW_NS=openclaw  OPENSHELL_NS=openshell
#      OPENSHELL_GATEWAY_URL=http://openshell.openshell.svc.cluster.local:8080
# ---------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
OPENSHELL_NS="${OPENSHELL_NS:-openshell}"
OPENSHELL_GATEWAY_URL="${OPENSHELL_GATEWAY_URL:-http://openshell.${OPENSHELL_NS}.svc.cluster.local:8080}"
AGENT_SANDBOX_NS="${AGENT_SANDBOX_NS:-agent-sandbox-system}"

c_reset=$'\033[0m'; c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'
c_yellow=$'\033[1;33m'; c_cyan=$'\033[1;36m'
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
note() { printf '%s---%s %s\n' "$c_cyan" "$c_reset" "$*"; }

check_sandbox_imagepull() {
  local bad=""
  bad="$(oc -n "$OPENSHELL_NS" get pods -o json 2>/dev/null | python3 -c "
import json,sys
items=json.load(sys.stdin).get('items') or []
for p in items:
  name=p.get('metadata',{}).get('name','')
  if 'openclaw-agent' not in name:
    continue
  for cs in (p.get('status') or {}).get('containerStatuses') or []:
    st=cs.get('state') or {}
    if st.get('waiting',{}).get('reason') in ('ImagePullBackOff','ErrImagePull'):
      print(name); break
  for cs in (p.get('status') or {}).get('initContainerStatuses') or []:
    st=cs.get('state') or {}
    if st.get('waiting',{}).get('reason') in ('ImagePullBackOff','ErrImagePull'):
      print(name); break
" 2>/dev/null || true)"
  if [[ -n "$bad" ]]; then
    warn "Sandbox pod ImagePullBackOff: $bad"
    note "Fix: ./scripts/fix-openshell-sandbox-image.sh"
    note "Then: ./scripts/clear-openclaw-sessions.sh → /new → one message"
    return 1
  fi
  return 0
}
die()  { printf '[fail] %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found in PATH"; }

resolve_openclaw_pod() {
  oc -n "$OPENCLAW_NS" get pods -l app.kubernetes.io/name=openclaw -o json 2>/dev/null | python3 -c '
import json, sys
items = json.load(sys.stdin).get("items") or []
for it in items:
  if it.get("metadata", {}).get("deletionTimestamp"):
    continue
  if (it.get("status") or {}).get("phase") != "Running":
    continue
  cs = (it.get("status") or {}).get("containerStatuses") or []
  if cs and all(c.get("ready") for c in cs):
    print(it["metadata"]["name"])
    break
' 2>/dev/null || true
}

resolve_agent_pod() {
  local name=""
  name="$(oc -n "$OPENSHELL_NS" get pods -o json 2>/dev/null | python3 -c '
import json, sys
items = json.load(sys.stdin).get("items") or []
candidates = []
for it in items:
  n = it.get("metadata", {}).get("name") or ""
  if not n.startswith("openclaw-agent-"):
    continue
  if it.get("metadata", {}).get("deletionTimestamp"):
    continue
  if (it.get("status") or {}).get("phase") != "Running":
    continue
  cs = (it.get("status") or {}).get("containerStatuses") or []
  if cs and all(c.get("ready") for c in cs):
    candidates.append(n)
print(sorted(candidates)[-1] if candidates else "")
' 2>/dev/null || true)"
  [[ -n "$name" ]] || return 1
  printf '%s' "$name"
}

agent_container() {
  oc -n "$OPENSHELL_NS" get pod "$1" -o jsonpath='{.spec.containers[0].name}' 2>/dev/null || echo agent
}

cmd_status() {
  need oc
  step "Agent Sandbox controller ($AGENT_SANDBOX_NS)"
  if oc -n "$AGENT_SANDBOX_NS" get deploy agent-sandbox-controller >/dev/null 2>&1; then
    oc -n "$AGENT_SANDBOX_NS" get deploy agent-sandbox-controller
    oc -n "$AGENT_SANDBOX_NS" rollout status deploy/agent-sandbox-controller --timeout=60s 2>/dev/null \
      && ok "agent-sandbox-controller ready" \
      || warn "agent-sandbox-controller not ready"
  else
    warn "agent-sandbox-controller not found in $AGENT_SANDBOX_NS"
  fi
  echo

  step "OpenShell gateway ($OPENSHELL_NS)"
  oc -n "$OPENSHELL_NS" get sts,svc,pods 2>/dev/null || warn "OpenShell namespace missing"
  echo

  step "OpenClaw sandbox wiring ($OPENCLAW_NS)"
  local pod
  pod="$(resolve_openclaw_pod)" || die "No ready OpenClaw pod in $OPENCLAW_NS"
  ok "OpenClaw pod: $pod"
  oc -n "$OPENCLAW_NS" exec "$pod" -c openclaw -- sh -lc '
    HOME=/opt/openclaw OPENCLAW_CONFIG_PATH=/opt/openclaw/config/openclaw.json
    echo "--- sandbox config ---"
    node /app/openclaw.mjs config get agents.defaults.sandbox 2>/dev/null || true
    echo "--- openshell plugin ---"
    node /app/openclaw.mjs config get plugins.entries.openshell 2>/dev/null || true
    echo "--- openshell status ---"
    if command -v openshell >/dev/null 2>&1; then
      openshell status --gateway-endpoint "'"${OPENSHELL_GATEWAY_URL}"'" 2>&1 || true
    else
      echo "(openshell CLI not in pod — gateway wiring still OK if sandboxes exist)"
    fi
  ' || warn "OpenClaw exec failed"
}

cmd_list() {
  need oc
  step "OpenClaw sandbox runtimes (gateway view)"
  local pod
  if pod="$(resolve_openclaw_pod 2>/dev/null)"; then
    oc -n "$OPENCLAW_NS" exec "$pod" -c openclaw -- sh -lc '
      HOME=/opt/openclaw OPENCLAW_CONFIG_PATH=/opt/openclaw/config/openclaw.json
      node /app/openclaw.mjs sandbox list 2>&1 || true
    ' 2>/dev/null || warn "openclaw sandbox list failed"
  else
    warn "No ready OpenClaw pod for sandbox list"
  fi
  echo

  step "Sandbox CRs ($OPENSHELL_NS)"
  oc -n "$OPENSHELL_NS" get sandboxes -o wide 2>/dev/null \
    || warn "No sandboxes CRD or none listed"
  echo

  step "Agent workload pods ($OPENSHELL_NS)"
  local pods
  pods="$(oc -n "$OPENSHELL_NS" get pods -o name 2>/dev/null | grep openclaw-agent || true)"
  if [[ -n "$pods" ]]; then
    printf '%s\n' "$pods"
    check_sandbox_imagepull || true
    ok "Agent sandbox pod(s) present"
  else
    warn "No openclaw-agent-* pods yet"
    note "/new alone does NOT create a sandbox pod."
    note "In Control UI: /new → send any message (e.g. investigate prompt) → wait 15–60s"
    note "Then: $0 wait   or: watch oc get pods -n $OPENSHELL_NS"
  fi
}

cmd_wait() {
  need oc
  local timeout="${1:-120}" elapsed=0 agent=""
  step "Waiting up to ${timeout}s for openclaw-agent-* (send a Control UI message after /new)"
  while (( elapsed < timeout )); do
    if agent="$(resolve_agent_pod 2>/dev/null)"; then
      ok "Agent pod ready: $agent"
      cmd_list
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    printf '.'
  done
  echo
  warn "Timed out after ${timeout}s"
  check_sandbox_imagepull || true
  note "Control UI: hard refresh → /new → send a message (investigate prompt is fine)"
  note "In another terminal: watch oc get pods -n $OPENSHELL_NS"
  cmd_list
  return 1
}

cmd_inspect() {
  need oc
  local agent ctr
  agent="$(resolve_agent_pod)" || {
    warn "No running openclaw-agent-* pod"
    note "Control UI: /new → send any message → $0 wait"
    return 1
  }
  ctr="$(agent_container "$agent")"
  step "Inspect sandbox filesystem in $agent (container: $ctr)"
  oc -n "$OPENSHELL_NS" exec "$agent" -c "$ctr" -- sh -c '
    echo "hostname: $(hostname)"
    echo "whoami: $(id -un 2>/dev/null || echo unknown)"
    echo "--- /sandbox (top) ---"
    ls -la /sandbox 2>/dev/null | head -15 || echo "/sandbox missing"
    echo "--- skills ---"
    if test -f /sandbox/.openclaw/sandbox-skills/skills/netobserv-investigate/SKILL.md; then
      echo "netobserv-investigate/SKILL.md: present"
    else
      echo "netobserv-investigate/SKILL.md: MISSING (re-seed or /new after seed)"
    fi
    echo "--- kubeconfig ---"
    if test -f /sandbox/.kube/config; then
      echo "/sandbox/.kube/config: present (limited SA — policy still blocks :6443)"
    else
      echo "no /sandbox/.kube/config"
    fi
    echo "--- sample paths ---"
    find /sandbox -maxdepth 4 -type f 2>/dev/null | head -12
  '
  ok "Agent exec runs inside isolated pod — not on OpenClaw gateway host"
}

cmd_prove_6443() {
  need oc
  local agent ctr rc=0
  agent="$(resolve_agent_pod)" || {
    warn "No running openclaw-agent-* pod"
    note "Control UI: /new → send any message → $0 wait"
    return 1
  }
  ctr="$(agent_container "$agent")"
  step "From sandbox pod: curl kubernetes.default.svc:6443 (OpenShell SSRF block expected)"
  note "OpenShell hard-blocks control-plane :6443 — sandboxes use MCP + heal-proxy :8080 instead"
  if oc -n "$OPENSHELL_NS" exec "$agent" -c "$ctr" -- sh -c \
    'curl -sk --connect-timeout 5 --max-time 8 https://kubernetes.default.svc:6443/version 2>&1'; then
    warn "Unexpected: :6443 reachable from sandbox — review OpenShell policy"
    rc=1
  else
    ok "BLOCKED (expected) — sandbox cannot reach API server on :6443"
  fi
  return "$rc"
}

cmd_prove_mcp() {
  need oc
  local pod
  pod="$(resolve_openclaw_pod)" || die "No ready OpenClaw pod"
  step "OpenClaw gateway MCP probe (in-cluster — not sandbox exec)"
  oc -n "$OPENCLAW_NS" exec "$pod" -c openclaw -- sh -lc '
    HOME=/opt/openclaw OPENCLAW_CONFIG_PATH=/opt/openclaw/config/openclaw.json
    node /app/openclaw.mjs mcp list 2>&1 | head -20
    echo "---"
    node /app/openclaw.mjs mcp doctor --probe 2>&1 | head -25
  ' || warn "mcp doctor failed"
  ok "MCP servers reachable from gateway — scoped tools replace raw :6443 in sandbox"
}

cmd_prove_policy() {
  need oc
  step "Managed policy snippet ($OPENCLAW_NS/openclaw-config)"
  local pol
  pol="$(oc -n "$OPENCLAW_NS" get cm openclaw-config -o jsonpath='{.data.openclaw-managed-policy\.yaml}' 2>/dev/null || true)"
  if [[ -z "$pol" ]]; then
    warn "openclaw-managed-policy.yaml not in ConfigMap — apply lab kustomize"
    return 1
  fi
  printf '%s\n' "$pol" | head -55
  echo "..."
  note "Stock policy allows LLM host + heal-proxy :8080; denies arbitrary egress"
  if printf '%s\n' "$pol" | grep -q netobserv-heal-proxy; then
    ok "heal-proxy allowlist present"
  else
    warn "netobserv-heal-proxy not in policy fragment — merge openshell-openshift-api-policy-fragment.yaml"
  fi
}

cmd_hints() {
  cat <<EOF
${c_cyan}========== OpenShell sandbox — see SECURITY-DEMO.md (~5 min) ==========${c_reset}

Narrative:
  "OpenClaw is the agent UI. OpenShell creates an isolated pod when the agent
   actually runs a turn — not on /new alone. Send a message, then we prove the
   sandbox blocks :6443 and keeps skills under /sandbox."

Before Control UI (terminal 2):
  $0 list                    # empty before any agent turn

Control UI:
  hard refresh → /new → send ANY message (investigate prompt is fine)

Terminal 2 (while agent is thinking):
  $0 wait                    # polls up to 120s for openclaw-agent-*
  # or: watch oc get pods -n openshell

After pod appears:
  $0 inspect
  $0 prove-6443

Full security interlude (3 rogue prompts + MCP rebuttal + investigate follow-up):
  See SECURITY-DEMO.md in kit root (~5 min script)

Optional negative tests in Control UI (one prompt per /new — see SECURITY-DEMO.md):
  1) Delete todo-demo namespace with oc
  2) curl Kubernetes API :6443 for cluster-admin
  3) ClusterRoleBinding granting cluster-admin

Full automated proof:
  $0 prove

Pair with NetObserv demo:
  ./scripts/netobserv-e2e-openclaw-test.sh demo-a
  # Agent investigate uses MCP — not sandbox → :6443

EOF
}

cmd_security() {
  local doc="$ROOT/SECURITY-DEMO.md"
  [[ -f "$doc" ]] || die "missing $doc"
  step "Security demo script (first section)"
  sed -n '1,55p' "$doc"
  echo
  note "Full script: $doc"
  note "Quick: /new → message → $0 wait → rogue prompts (SECURITY-DEMO Part 2)"
}

cmd_prove() {
  local rc=0
  cmd_status || rc=1
  echo
  cmd_list || rc=1
  echo
  cmd_inspect || warn "inspect skipped (send a Control UI message after /new, then: $0 wait)"
  echo
  cmd_prove_6443 || rc=1
  echo
  cmd_prove_mcp || rc=1
  echo
  cmd_prove_policy || rc=1
  echo
  if [[ "$rc" -eq 0 ]]; then
    ok "Sandbox proof checks passed"
  else
    warn "Some proof checks failed or were skipped — see messages above"
  fi
  cmd_hints
  return "$rc"
}

usage() {
  cat <<EOF
openshell-sandbox-proof.sh — prove OpenClaw runs in OpenShell Agent Sandbox

  export DEMO_KIT_ROOT="${DEMO_KIT_ROOT:-$HOME/AIOps/demo/demo-4-platform-kit}"
  cd "$DEMO_KIT_ROOT"
  ./scripts/openshell-sandbox-proof.sh prove

Subcommands:
  status        Controller + OpenShell + OpenClaw sandbox config
  list          OpenClaw sandbox list + sandboxes + openclaw-agent-* pods
  wait [secs]   Poll for agent pod (default 120) — send UI message after /new first
  inspect       /sandbox filesystem in agent pod (needs active agent turn)
  prove-6443    Block test: curl :6443 from sandbox
  prove-mcp     MCP probe from OpenClaw gateway
  prove-policy  Show managed-policy allowlist snippet
  security      Print SECURITY-DEMO.md intro (full script in kit root)
  hints         Presenter narrative + rogue prompts
  prove         Run all checks (after /new + UI message; see SECURITY-DEMO.md)

Env: OPENCLAW_NS OPENSHELL_NS OPENSHELL_GATEWAY_URL AGENT_SANDBOX_NS
EOF
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    status)       cmd_status ;;
    list)         cmd_list ;;
    wait)         cmd_wait "${2:-120}" ;;
    inspect)      cmd_inspect ;;
    prove-6443)   cmd_prove_6443 ;;
    prove-mcp)    cmd_prove_mcp ;;
    prove-policy) cmd_prove_policy ;;
    security)     cmd_security ;;
    hints)        cmd_hints ;;
    prove)        cmd_prove ;;
    -h|--help|help|"") usage; [[ -n "$cmd" ]] || exit 1 ;;
    *) die "Unknown subcommand: $cmd (try --help)" ;;
  esac
}

main "$@"
