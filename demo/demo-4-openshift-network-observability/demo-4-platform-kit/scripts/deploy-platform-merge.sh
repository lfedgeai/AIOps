#!/usr/bin/env bash
# Deploy merged platform: RHOAI (dashboard + MLflow) + RHCL + OpenClaw wire.
#
# Usage:
#   ./scripts/deploy-platform-merge.sh              # full (rhoai → rhcl operator + ingress → wire)
#   ./scripts/deploy-platform-merge.sh rhoai        # Phase A only
#   ./scripts/deploy-platform-merge.sh rhcl         # Phase A′ (RHCL operator + OpenClaw OAuth ingress)
#   ./scripts/deploy-platform-merge.sh wire         # rewire OpenClaw MCP → RHOAI MLflow
#   ./scripts/deploy-platform-merge.sh grafana     # Grafana + Gateway Diagnostics dashboard
#   ./scripts/deploy-platform-merge.sh otel        # Gateway Diagnostics OTel stack
#   ./scripts/deploy-platform-merge.sh guardrails  # TrustyAI + OpenClaw wire (Layer 0)
#   ./scripts/deploy-platform-merge.sh aap         # AAP operator + OpenClaw heal wire
#   ./scripts/deploy-platform-merge.sh status
#
# Env:
#   MLFLOW_REMOVE_STANDALONE=1   delete openclaw/mlflow after wire (default 1)
#   SKIP_RHCL=1                  skip all RHCL on full run (operator + ingress)
#   SKIP_RHCL_INGRESS=1          on `all`: install RHCL operator only, skip Phase A′ OAuth ingress
#   RESEED_OPENCLAW=1            re-seed skills after wire (default 0)
#
# Note: default `all` = Phase A + A′ + wire — Phase C Grafana is `./scripts/deploy-platform-merge.sh grafana`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-all}"
MLFLOW_REMOVE_STANDALONE="${MLFLOW_REMOVE_STANDALONE:-1}"
SKIP_RHCL="${SKIP_RHCL:-0}"
SKIP_RHCL_INGRESS="${SKIP_RHCL_INGRESS:-0}"
RESEED_OPENCLAW="${RESEED_OPENCLAW:-0}"

c_green=$'\033[1;32m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }

chmod +x "$ROOT/scripts/install-rhoai-platform-minimal.sh" \
         "$ROOT/scripts/install-rhcl-ingress.sh" \
         "$ROOT/scripts/patch-openclaw-oidc-openshift.sh" \
         "$ROOT/scripts/wire-openclaw-mlflow.sh" \
         "$ROOT/scripts/install-grafana-network-aiops.sh" \
         "$ROOT/scripts/install-openclaw-otel-grafana.sh" \
         "$ROOT/scripts/install-trustyai-guardrails.sh" \
         "$ROOT/scripts/wire-openclaw-trustyai-guardrails.sh" \
         "$ROOT/scripts/install-aap.sh" \
         "$ROOT/scripts/wire-openclaw-aap.sh" \
         "$ROOT/scripts/teardown-custom-input-guard.sh"

case "$CMD" in
  status)
    "$ROOT/scripts/install-rhoai-platform-minimal.sh" status
    "$ROOT/scripts/install-rhcl-ingress.sh" status
    "$ROOT/scripts/install-grafana-network-aiops.sh" status || true
    "$ROOT/scripts/install-openclaw-otel-grafana.sh" status || true
    ;;
  rhoai)
    "$ROOT/scripts/install-rhoai-platform-minimal.sh" install
    ;;
  rhcl)
    "$ROOT/scripts/install-rhcl-ingress.sh" all
    ;;
  grafana)
    "$ROOT/scripts/install-grafana-network-aiops.sh" all
    "$ROOT/scripts/install-openclaw-otel-grafana.sh" all || \
      "$ROOT/scripts/install-grafana-network-aiops.sh" fix-auth
    ok "Grafana ready — demo-a-fast syncs metrics automatically"
    ;;
  otel)
    "$ROOT/scripts/install-openclaw-otel-grafana.sh" all
    ok "Gateway Diagnostics OTel ready — run demo-a-fast (includes metrics sync)"
    ;;
  guardrails)
    "$ROOT/scripts/teardown-custom-input-guard.sh" || true
    "$ROOT/scripts/install-trustyai-guardrails.sh" install
    "$ROOT/scripts/wire-openclaw-trustyai-guardrails.sh" all
    ok "TrustyAI guardrails wired — verify: ./scripts/netobserv-e2e-openclaw-test.sh trustyai-guard-check"
    ;;
  aap)
    "$ROOT/scripts/install-aap.sh" install
    ok "AAP installed — apply license in Gateway UI, then:"
    ok "  AAP_ADMIN_PASSWORD=... ./scripts/wire-openclaw-aap.sh all"
    ok "  ENABLE_ANSIBLE_MCP=1 ./scripts/seed-openclaw-netobserv-skills.sh"
    ok "  ./scripts/netobserv-e2e-openclaw-test.sh aap-check"
    ;;
  wire)
    MLFLOW_BACKEND=rhoai MLFLOW_REMOVE_STANDALONE="$MLFLOW_REMOVE_STANDALONE" \
      "$ROOT/scripts/wire-openclaw-mlflow.sh"
    if [[ "$RESEED_OPENCLAW" == "1" ]]; then
      step "Re-seed OpenClaw (ENABLE_RHOAI_PLATFORM=1)"
      ENABLE_RHOAI_PLATFORM=1 ENABLE_OPENCLAW_MLFLOW=0 \
        "$ROOT/scripts/seed-openclaw-netobserv-skills.sh"
    fi
    ;;
  all)
    "$ROOT/scripts/install-rhoai-platform-minimal.sh" install
    if [[ "$SKIP_RHCL" != "1" ]]; then
      if [[ "$SKIP_RHCL_INGRESS" == "1" ]]; then
        step "RHCL operator only (SKIP_RHCL_INGRESS=1 — no OpenClaw OAuth ingress)"
        "$ROOT/scripts/install-rhcl-ingress.sh" install || true
      else
        step "RHCL operator + OpenClaw OAuth ingress (Phase A′)"
        "$ROOT/scripts/install-rhcl-ingress.sh" all || true
      fi
    fi
    MLFLOW_BACKEND=rhoai MLFLOW_REMOVE_STANDALONE="$MLFLOW_REMOVE_STANDALONE" \
      "$ROOT/scripts/wire-openclaw-mlflow.sh"
    ok "Platform merge complete — verify MLflow: ./scripts/netobserv-e2e-openclaw-test.sh mlflow-check"
    if [[ "$SKIP_RHCL" != "1" && "$SKIP_RHCL_INGRESS" != "1" ]]; then
      ok "RHCL OpenClaw OAuth — verify: ./scripts/install-rhcl-ingress.sh status"
    fi
    ;;
  *)
    echo "usage: $0 [all|rhoai|rhcl|wire|grafana|otel|guardrails|aap|status]" >&2
    exit 1
    ;;
esac
