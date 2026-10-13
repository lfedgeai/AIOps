#!/usr/bin/env bash
# Greenfield install orchestrator — delegates to demo kit scripts.
#
# Run from: demo-4-greenfield-install/  (NOT from the platform-kit folder)
#
# Usage:
#   ./scripts/greenfield-install.sh plan
#   ./scripts/greenfield-install.sh all
#   ./scripts/greenfield-install.sh netobserv|openshell|rhoai|…|verify|status
#
# Env:
#   DEMO_KIT_ROOT             path to demo-4-platform-kit (default: sibling ../demo-4-platform-kit)
#   SLACK_CHANNEL_ID          required for slack, event, spiffe phases
#   SKIP_NETOBSERV / SKIP_OPENCLAW / SKIP_RHCL / AUTO_CONTINUE / LAB_CFG
set -euo pipefail

GF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GF_ROOT
[[ -f "$GF_ROOT/config/env.local" ]] && source "$GF_ROOT/config/env.local"
# shellcheck source=scripts/resolve-demo-kit.sh
source "$GF_ROOT/scripts/resolve-demo-kit.sh"
resolve_demo_kit

KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-plan}"

# Site secrets (YAML or JSON) — see scripts/site-config.sh
# shellcheck source=scripts/site-config.sh
source "$GF_ROOT/scripts/site-config.sh"
SITE_CONFIG="$(site_config_path)"
export SITE_SECRETS_FILE="$SITE_CONFIG"
GREENFIELD_CONFIG_MODE="${GREENFIELD_CONFIG_MODE:-hybrid}"
SKIP_SITE_CONFIG="${SKIP_SITE_CONFIG:-0}"

SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
LAB_CFG="${LAB_CFG:-$HOME/labs/openshell-on-openshift-lab/manifests/openclaw/config.yaml}"
AUTO_CONTINUE="${AUTO_CONTINUE:-0}"
SKIP_NETOBSERV="${SKIP_NETOBSERV:-0}"
SKIP_OPENCLAW="${SKIP_OPENCLAW:-0}"
SKIP_RHCL="${SKIP_RHCL:-0}"
SKIP_RHCL_INGRESS="${SKIP_RHCL_INGRESS:-0}"

KIT="$DEMO_KIT_ROOT/scripts"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_cyan=$'\033[1;36m'; c_reset=$'\033[0m'
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -n "$KUBECTL" ]] || die "oc/kubectl required"
chmod +x "$KIT/"*.sh 2>/dev/null || true

pause_manual() {
  [[ "$AUTO_CONTINUE" == "1" ]] && return 0
  printf '\n%s── MANUAL ──%s %s\n' "$c_cyan" "$c_reset" "$*"
  read -r -p "Press Enter when complete (Ctrl-C to abort)... " _ || true
}

require_slack_id() {
  [[ -n "$SLACK_CHANNEL_ID" ]] || die "SLACK_CHANNEL_ID required — set in $SITE_CONFIG or config prompt"
}

load_site_config() {
  local phase="${1:-all}"
  [[ "$SKIP_SITE_CONFIG" == "1" ]] && return 0
  site_config_ensure "$phase"
}

cmd_config() {
  local sub="${1:-help}"
  case "$sub" in
    init) site_config_init ;;
    prompt|wizard)
      shift
      site_config_prompt "$@"
      ;;
    validate) site_config_validate "${2:-all}" ;;
    show) site_config_show ;;
    apply) site_config_apply "${2:-all}" ;;
    path) site_config_path ;;
    help|-h|--help)
      cat <<EOF
${c_cyan}Site secrets${c_reset} — $SITE_CONFIG (gitignored)

  ./scripts/greenfield-install.sh config init       # copy template YAML
  ./scripts/greenfield-install.sh config prompt     # interactive wizard (saves YAML)
  ./scripts/greenfield-install.sh config prompt --minimal   # Phases 1–2 only
  ./scripts/greenfield-install.sh config prompt --full      # + Slack & AAP (skippable)
  ./scripts/greenfield-install.sh config validate   # check required fields
  ./scripts/greenfield-install.sh config show        # masked summary
  ./scripts/greenfield-install.sh config apply llm   # push LLM secret + patch lab config

Modes (env GREENFIELD_CONFIG_MODE):
  hybrid  load YAML; prompt for missing (default)
  file    require complete YAML — no prompts
  prompt  always run wizard if file missing

Skip enforcement: SKIP_SITE_CONFIG=1
EOF
      ;;
    *)
      echo "usage: $0 config {init|prompt|validate|show|apply|path}" >&2
      exit 1
      ;;
  esac
}

cmd_plan() {
  cat <<EOF
${c_cyan}Greenfield install${c_reset} — demo kit: $DEMO_KIT_ROOT
Docs: $GF_ROOT/INSTALL.md

  1  netobserv    NetObserv + Loki (AWS) + todo app
  2  openshell    OpenShell/OpenClaw + LLM (automated deploy)
  3  rhoai        RHOAI + MLflow Traces
  4  grafana      Grafana + OTel federation
  5  guardrails   TrustyAI
  6  aap          AAP + Gitea + ansible-mcp (license manual)
  7  agent        seed skills + ENABLE_ANSIBLE_MCP=1
  8  slack        Slack Socket Mode
  9  event        Event-AIOps
 10  spiffe       ZTWI / SPIFFE mTLS
 11  rhcl         RHCL OAuth (optional)
 12  verify       preflight + feature checks

  ./scripts/greenfield-install.sh config init     # site secrets YAML
  ./scripts/greenfield-install.sh config prompt   # wizard (or edit site-secrets.local.yaml)
  ./scripts/greenfield-install.sh all
  ./scripts/greenfield-install.sh status
EOF
}

cmd_prereq() {
  load_site_config prereq || true
  step "Phase 0 — prerequisites"
  chmod +x "$GF_ROOT/scripts/install-helm3.sh" "$GF_ROOT/scripts/cluster-login.sh" 2>/dev/null || true
  "$GF_ROOT/scripts/install-helm3.sh" install
  "$GF_ROOT/scripts/cluster-login.sh" check || {
    warn "Not logged in to OpenShift — see $GF_ROOT/docs/CLUSTER-LOGIN.md"
    "$GF_ROOT/scripts/cluster-login.sh" help
    pause_manual "oc login complete (./scripts/cluster-login.sh check passes)"
    "$GF_ROOT/scripts/cluster-login.sh" check
  }
  cat <<EOF
  [x] Helm 3 — installed above if missing
  [ ] OpenShift login — ./scripts/cluster-login.sh check
  [ ] OpenShift 4.21+ cluster-admin
  [ ] Bastion: oc, jq, curl, podman, git, python3, python3-pyyaml
  [ ] Site secrets: $SITE_CONFIG — ./scripts/greenfield-install.sh config prompt

  Cluster login guide: $GF_ROOT/docs/CLUSTER-LOGIN.md

  "$KIT/supply-chain-check.sh"
EOF
  "$KIT/supply-chain-check.sh" || warn "supply-chain-check drift"
  pause_manual "Complete prerequisites"
}

cmd_netobserv() {
  [[ "$SKIP_NETOBSERV" == "1" ]] && { warn "SKIP_NETOBSERV=1"; return 0; }
  load_site_config netobserv
  "$GF_ROOT/scripts/cluster-login.sh" check || die "oc login required — see docs/CLUSTER-LOGIN.md"
  step "Phase 1 — NetObserv + todo"
  export STORAGE_CLASS="${STORAGE_CLASS:-gp3-csi}"
  pause_manual "oc logged in (cluster-login.sh check) + AWS values loaded from site config"
  "$KIT/install-netobserv-aws.sh"
  "$KIT/deploy-netobserv-todo-app.sh"
  if "$KUBECTL" get lokistack loki -n "${LOKI_NS:-netobserv-loki}" >/dev/null 2>&1; then
    "$KIT/tune-loki-ingestion.sh" apply || warn "tune-loki-ingestion failed"
  fi
  ok "Phase 1 complete"
}

cmd_openshell() {
  [[ "$SKIP_OPENCLAW" == "1" ]] && { warn "SKIP_OPENCLAW=1"; return 0; }
  load_site_config openshell
  step "Phase 2 — OpenShell + OpenClaw"
  "$GF_ROOT/scripts/install-helm3.sh" install
  "$KIT/clone-openshell-lab.sh"
  site_config_apply openshell || warn "site-config apply openshell had issues"
  "$KIT/supply-chain-check.sh" || warn "supply-chain-check drift"
  chmod +x "$GF_ROOT/scripts/phase2-openshell.sh" 2>/dev/null || true
  "$GF_ROOT/scripts/phase2-openshell.sh" deploy \
    || die "Phase 2 deploy failed — see $GF_ROOT/docs/PHASE-2-OPENSHELL.md"
  "$GF_ROOT/scripts/phase2-openshell.sh" harden \
    || die "Phase 2 harden failed"
  "$GF_ROOT/scripts/phase2-openshell.sh" check \
    || die "Phase 2 check failed — see $GF_ROOT/docs/PHASE-2-OPENSHELL.md"
  if [[ "${SKIP_PHASE2_PROBE:-0}" != "1" ]]; then
    "$GF_ROOT/scripts/phase2-openshell.sh" probe \
      || die "LLM probe failed — see docs/PHASE-2-LLM-PROVIDERS.md"
  fi
  ok "Phase 2 complete — Control UI: ./scripts/phase2-openshell.sh ui"
}

cmd_rhoai() {
  load_site_config openshell
  "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw >/dev/null 2>&1 \
    || die "Phase 2 required — openclaw missing"
  step "Phase 3 — RHOAI + MLflow"
  chmod +x "$GF_ROOT/scripts/phase3-rhoai.sh" 2>/dev/null || true
  "$GF_ROOT/scripts/phase3-rhoai.sh" plan
  printf '\n'
  warn "Install may take 20–40 min — see $GF_ROOT/docs/PHASE-3-RHOAI.md"
  "$KIT/install-rhoai-platform-minimal.sh" install
  MLFLOW_BACKEND=rhoai MLFLOW_REMOVE_STANDALONE="${MLFLOW_REMOVE_STANDALONE:-1}" \
    "$KIT/wire-openclaw-mlflow.sh" \
    || die "wire-openclaw-mlflow.sh failed — re-run: MLFLOW_BACKEND=rhoai $KIT/wire-openclaw-mlflow.sh"
  "$GF_ROOT/scripts/phase3-rhoai.sh" check || warn "Phase 3 checks incomplete"
  ok "Phase 3 — run: $GF_ROOT/scripts/phase3-rhoai.sh verify"
}

cmd_grafana() {
  load_site_config openshell
  "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw >/dev/null 2>&1 \
    || die "Phase 2 required — openclaw missing"
  step "Phase 4 — Grafana + OTel"
  chmod +x "$GF_ROOT/scripts/phase4-grafana.sh" 2>/dev/null || true
  "$GF_ROOT/scripts/phase4-grafana.sh" plan
  printf '\n'
  warn "Install may take 20–40 min — see $GF_ROOT/docs/PHASE-4-GRAFANA.md"
  "$GF_ROOT/scripts/phase4-grafana.sh" deploy \
    || die "Phase 4 deploy failed — see $GF_ROOT/docs/PHASE-4-GRAFANA.md"
  "$GF_ROOT/scripts/phase4-grafana.sh" check \
    || die "Phase 4 check failed"
  ok "Phase 4 complete — run: $GF_ROOT/scripts/phase4-grafana.sh verify"
}

cmd_guardrails() {
  load_site_config openshell
  step "Phase 5 — TrustyAI Guardrails"
  chmod +x "$GF_ROOT/scripts/phase5-guardrails.sh" 2>/dev/null || true
  "$GF_ROOT/scripts/phase5-guardrails.sh" plan
  printf '\n'
  warn "TrustyAI CRD can take 15–30 min — see $GF_ROOT/docs/PHASE-5-GUARDRAILS.md"
  "$GF_ROOT/scripts/phase5-guardrails.sh" deploy \
    || die "Phase 5 deploy failed — see $GF_ROOT/docs/PHASE-5-GUARDRAILS.md"
  "$GF_ROOT/scripts/phase5-guardrails.sh" check || warn "Phase 5 checks incomplete"
  ok "Phase 5 — run: $GF_ROOT/scripts/phase5-guardrails.sh verify"
}

cmd_aap() {
  step "Phase 6 — AAP (install only; license manual in Gateway UI)"
  chmod +x "$GF_ROOT/scripts/phase6-aap.sh" 2>/dev/null || true
  "$GF_ROOT/scripts/phase6-aap.sh" plan
  printf '\n'
  warn "Stops after AAP install — you apply the subscription in Gateway UI, then: phase6-aap.sh wire"
  "$GF_ROOT/scripts/phase6-aap.sh" deploy \
    || die "Phase 6 deploy failed — see $GF_ROOT/docs/PHASE-6-AAP.md"
  ok "Phase 6 install done — apply license in Gateway UI, then: ./scripts/phase6-aap.sh wire"
}

cmd_agent() {
  step "Phase 7 — NetObserv agent kit seed"
  chmod +x "$GF_ROOT/scripts/phase7-agent.sh" 2>/dev/null || true
  "$GF_ROOT/scripts/phase7-agent.sh" plan
  printf '\n'
  "$GF_ROOT/scripts/phase7-agent.sh" deploy \
    || die "Phase 7 deploy failed — see $GF_ROOT/docs/PHASE-7-AGENT.md"
  "$GF_ROOT/scripts/phase7-agent.sh" check || warn "Phase 7 checks incomplete"
  ok "Phase 7 — run: $GF_ROOT/scripts/phase7-agent.sh verify"
}

cmd_slack() {
  step "Phase 8 — Slack Socket Mode"
  chmod +x "$GF_ROOT/scripts/phase8-slack.sh" 2>/dev/null || true
  "$GF_ROOT/scripts/phase8-slack.sh" plan
  printf '\n'
  "$GF_ROOT/scripts/phase8-slack.sh" deploy \
    || die "Phase 8 deploy failed — see $GF_ROOT/docs/PHASE-8-SLACK.md"
  "$GF_ROOT/scripts/phase8-slack.sh" check || warn "Phase 8 checks incomplete"
  ok "Phase 8 — run: $GF_ROOT/scripts/phase8-slack.sh verify"
}

cmd_event() {
  step "Phase 9 — event-AIOps"
  chmod +x "$GF_ROOT/scripts/phase9-event.sh" 2>/dev/null || true
  "$GF_ROOT/scripts/phase9-event.sh" plan
  printf '\n'
  "$GF_ROOT/scripts/phase9-event.sh" deploy \
    || die "Phase 9 deploy failed — see $GF_ROOT/docs/PHASE-9-EVENT.md"
  "$GF_ROOT/scripts/phase9-event.sh" check || warn "Phase 9 checks incomplete"
  ok "Phase 9 — run: $GF_ROOT/scripts/phase9-event.sh verify"
}

cmd_spiffe() {
  step "Phase 10 — SPIFFE / ZTWI mTLS"
  chmod +x "$GF_ROOT/scripts/phase10-spiffe.sh" 2>/dev/null || true
  "$GF_ROOT/scripts/phase10-spiffe.sh" plan
  printf '\n'
  "$GF_ROOT/scripts/phase10-spiffe.sh" deploy \
    || die "Phase 10 deploy failed — see $GF_ROOT/docs/PHASE-10-SPIFFE.md"
  "$GF_ROOT/scripts/phase10-spiffe.sh" check || warn "Phase 10 checks incomplete"
  ok "Phase 10 — run: $GF_ROOT/scripts/phase10-spiffe.sh verify"
}

cmd_rhcl() {
  [[ "$SKIP_RHCL" == "1" ]] && { warn "SKIP_RHCL=1"; return 0; }
  step "Phase 11 — RHCL OAuth"
  chmod +x "$GF_ROOT/scripts/phase11-rhcl.sh" 2>/dev/null || true
  "$GF_ROOT/scripts/phase11-rhcl.sh" plan
  printf '\n'
  "$GF_ROOT/scripts/phase11-rhcl.sh" deploy \
    || die "Phase 11 deploy failed — see $GF_ROOT/docs/PHASE-11-RHCL.md"
  "$GF_ROOT/scripts/phase11-rhcl.sh" check || warn "Phase 11 checks incomplete"
  ok "Phase 11 — run: $GF_ROOT/scripts/phase11-rhcl.sh verify"
}

cmd_verify() {
  step "Phase 12 — verify"
  SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}" "$KIT/demo-cluster-preflight.sh" check || warn "preflight issues"
  "$KIT/netobserv-e2e-openclaw-test.sh" aap-check || warn "aap-check"
  "$KIT/netobserv-e2e-openclaw-test.sh" mlflow-check || warn "mlflow-check"
  "$KUBECTL" -n "$OPENCLAW_NS" get secret openclaw-slack-tokens >/dev/null 2>&1 && \
    "$KIT/netobserv-e2e-openclaw-test.sh" slack-check || true
  "$KUBECTL" get crd clusterspiffeids.spire.spiffe.io >/dev/null 2>&1 && \
    SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}" "$KIT/netobserv-e2e-openclaw-test.sh" spiffe-check || true
  ok "Greenfield verify done — trial: $KIT/netobserv-e2e-openclaw-test.sh demo-a-fast"
}

cmd_status() {
  step "Greenfield status (demo kit: $DEMO_KIT_ROOT)"
  "$KIT/install-netobserv-aws.sh" status 2>/dev/null || true
  "$KIT/install-rhoai-platform-minimal.sh" status 2>/dev/null || true
  "$KIT/install-aap.sh" status 2>/dev/null || true
  "$KIT/wire-openclaw-aap.sh" status 2>/dev/null || true
  "$KIT/install-ztwi-spire.sh" status 2>/dev/null || true
  SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}" "$KIT/demo-cluster-preflight.sh" check 2>/dev/null || true
}

probe_phase() {
  local phase="$1"
  case "$phase" in
    netobserv)
      "$KUBECTL" get flowcollector cluster -n "${NETOBSERV_NS:-netobserv}" >/dev/null 2>&1 \
        && "$KUBECTL" get deploy/postgresql -n todo-demo >/dev/null 2>&1
      ;;
    openshell)
      "$KUBECTL" -n "${OPENSHELL_NS:-openshell}" get statefulset/openshell >/dev/null 2>&1 \
        && "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw >/dev/null 2>&1
      ;;
    rhoai)
      "$KUBECTL" -n "${RHOAI_NS:-redhat-ods-applications}" get deploy/mlflow >/dev/null 2>&1
      ;;
    grafana)
      "$KUBECTL" -n "${GRAFANA_NS:-netobserv-demo}" get deploy/network-aiops-deployment >/dev/null 2>&1 \
        && "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw-otel-prometheus >/dev/null 2>&1
      ;;
    guardrails)
      "$KUBECTL" -n "${GUARDRAILS_NS:-netobserv-guardrails}" get guardrailsorchestrator guardrails-orchestrator >/dev/null 2>&1
      ;;
    aap)
      "$KUBECTL" -n "${AAP_NS:-ansible-automation-platform}" get ansibleautomationplatform >/dev/null 2>&1 \
        && "$KUBECTL" -n "$OPENCLAW_NS" get secret/openclaw-aap-launcher >/dev/null 2>&1
      ;;
    agent)
      "$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-mcp deploy/netobserv-heal-proxy deploy/netobserv-capture-proxy >/dev/null 2>&1
      ;;
    slack)
      "$KUBECTL" -n "$OPENCLAW_NS" get secret/openclaw-slack-tokens >/dev/null 2>&1 \
        && "$KUBECTL" -n "$OPENCLAW_NS" get configmap openclaw-config \
          -o jsonpath='{.data.openclaw\.json}' 2>/dev/null \
          | python3 -c 'import json,sys; d=json.load(sys.stdin); print("slack" in ((d.get("plugins") or {}).get("allow") or []))' 2>/dev/null \
          | grep -q True
      ;;
    event)
      "$KUBECTL" -n "$OPENCLAW_NS" get secret/openclaw-hooks-token deploy/netobserv-grafana-bridge >/dev/null 2>&1
      ;;
    spiffe)
      "$KUBECTL" get clusterspiffeid/netobserv-grafana-bridge >/dev/null 2>&1 \
        && [[ "$("$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-grafana-bridge \
          -o jsonpath='{.spec.template.spec.containers[?(@.name=="bridge")].env[?(@.name=="SPIFFE_MTLS")].value}' 2>/dev/null)" == "1" ]]
      ;;
    rhcl)
      "$KUBECTL" -n "$OPENCLAW_NS" get gateway/openclaw-gateway >/dev/null 2>&1
      ;;
    *) return 1 ;;
  esac
}

cmd_phases() {
  local who p done=0 total=0 next=""
  who="$("$KUBECTL" whoami 2>/dev/null || echo '(not logged in)')"
  step "Greenfield phase progress — $who"
  printf '\n'
  for p in netobserv openshell rhoai grafana guardrails aap agent slack event spiffe rhcl; do
    total=$((total + 1))
    if probe_phase "$p"; then
      printf '  %s✓%s  %-12s done\n' "$c_green" "$c_reset" "$p"
      done=$((done + 1))
    else
      printf '  ○  %-12s pending\n' "$p"
      [[ -z "$next" ]] && next="$p"
    fi
  done
  printf '\n'
  printf 'Progress: %s/%s phases detected on cluster\n' "$done" "$total"
  if [[ -n "$next" ]]; then
    printf 'Suggested next: ./scripts/greenfield-install.sh %s\n' "$next"
  else
    ok "All phases detected — run: ./scripts/greenfield-install.sh verify"
  fi
  printf '\nNote: on a cluster where phases are already installed, most will show ✓; on a new cluster most will be pending.\n'
}

cmd_all() {
  load_site_config all
  cmd_prereq
  cmd_netobserv
  cmd_openshell
  cmd_rhoai
  cmd_grafana
  cmd_guardrails
  cmd_aap
  cmd_agent
  cmd_slack
  cmd_event
  cmd_spiffe
  cmd_rhcl
  cmd_verify
  ok "Greenfield install complete"
}

case "$CMD" in
  plan|help|-h|--help) cmd_plan ;;
  config|secrets) cmd_config "${2:-help}" ;;
  prereq|0) cmd_prereq ;;
  netobserv|1) cmd_netobserv ;;
  openshell|openclaw|2) cmd_openshell ;;
  rhoai|3) cmd_rhoai ;;
  grafana|4) cmd_grafana ;;
  guardrails|trustyai|5) cmd_guardrails ;;
  aap|6) cmd_aap ;;
  agent|seed|7) cmd_agent ;;
  slack|8) cmd_slack ;;
  event|event-aiops|9) cmd_event ;;
  spiffe|ztwi|10) cmd_spiffe ;;
  rhcl|11) cmd_rhcl ;;
  verify|12) cmd_verify ;;
  status) cmd_status ;;
  phases|progress) cmd_phases ;;
  preflight|preflight-0-3) "$GF_ROOT/scripts/preflight-phases-0-3.sh" "${2:-}" ;;
  all) cmd_all ;;
  *)
    echo "usage: $0 {plan|config|preflight|phases|all|status|prereq|netobserv|openshell|rhoai|grafana|guardrails|aap|agent|slack|event|spiffe|rhcl|verify}" >&2
    exit 1
    ;;
esac
