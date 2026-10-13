#!/usr/bin/env bash
# Greenfield site secrets — YAML file + interactive prompts.
#
# Usage (via greenfield-install.sh config … or directly):
#   ./scripts/site-config.sh init
#   ./scripts/site-config.sh prompt
#   ./scripts/site-config.sh validate [phase]
#   ./scripts/site-config.sh show
#   ./scripts/site-config.sh load          # eval "$(./scripts/site-config.sh load)"
#   ./scripts/site-config.sh apply [phase] # llm | slack | aap | all
set -euo pipefail

GF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${GF_ROOT}/scripts/site_config.py"
CMD="${1:-help}"
ARG="${2:-}"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }

[[ -x "$PY" ]] || chmod +x "$PY" 2>/dev/null || true

site_config_path() {
  if [[ -n "${SITE_SECRETS_FILE:-}" ]]; then
    printf '%s' "$SITE_SECRETS_FILE"
    return
  fi
  if [[ -f "$GF_ROOT/config/site-secrets.local.yaml" ]]; then
    printf '%s' "$GF_ROOT/config/site-secrets.local.yaml"
  elif [[ -f "$GF_ROOT/config/site-secrets.local.json" ]]; then
    printf '%s' "$GF_ROOT/config/site-secrets.local.json"
  else
    printf '%s' "$GF_ROOT/config/site-secrets.local.yaml"
  fi
}

SITE_CONFIG="$(site_config_path)"

sc() {
  SITE_CONFIG="$(site_config_path)"
  python3 "$PY" --file "$SITE_CONFIG" "$@"
}

site_config_init() {
  sc init
  SITE_CONFIG="$(site_config_path)"
  chmod 600 "$SITE_CONFIG" 2>/dev/null || true
}

site_config_prompt() {
  sc prompt "${@}"
  chmod 600 "$SITE_CONFIG" 2>/dev/null || true
}

site_config_validate() {
  local phase="${1:-all}"
  sc validate --phase "$phase"
}

site_config_show() {
  sc show
}

site_config_load() {
  sc export-shell
}

site_config_apply() {
  local phase="${1:-all}"
  sc apply --phase "$phase" --lab-dir "${OPENCLAW_LAB_DIR:-$HOME/labs/openshell-on-openshift-lab}"
}

# Load YAML into current shell; prompt if missing and GREENFIELD_CONFIG_MODE allows.
site_config_ensure() {
  local phase="${1:-all}"
  local mode="${GREENFIELD_CONFIG_MODE:-hybrid}"

  if [[ ! -f "$(site_config_path)" ]]; then
    case "$mode" in
      file)
        warn "Missing $SITE_CONFIG"
        printf '  Run: ./scripts/greenfield-install.sh config init\n' >&2
        printf '  Then: ./scripts/greenfield-install.sh config prompt\n' >&2
        return 1
        ;;
      prompt|hybrid)
        warn "No site config — starting interactive wizard"
        site_config_init
        site_config_prompt
        ;;
      *)
        warn "Unknown GREENFIELD_CONFIG_MODE=$mode"
        return 1
        ;;
    esac
  fi

  if [[ "$mode" == "hybrid" ]] && ! site_config_validate "$phase" 2>/dev/null; then
    warn "Site config incomplete for phase '$phase' — prompting for missing values"
    site_config_prompt
  fi

  site_config_validate "$phase"
  SITE_CONFIG="$(site_config_path)"
  eval "$(site_config_load)"
  export SITE_SECRETS_FILE="$SITE_CONFIG"
  ok "site config loaded ($SITE_CONFIG)"
}

# Only run CLI when executed directly (not when sourced by greenfield-install.sh)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
case "$CMD" in
  init) site_config_init ;;
  prompt) shift; site_config_prompt "$@" ;;
  validate) site_config_validate "${ARG:-all}" ;;
  show) site_config_show ;;
  load|export) site_config_load ;;
  apply) site_config_apply "${ARG:-all}" ;;
  ensure) site_config_ensure "${ARG:-all}" ;;
  path) site_config_path ;;
  help|-h|--help)
    cat <<EOF
Usage: $0 {init|prompt|validate|show|load|apply|ensure|path}

  SITE_SECRETS_FILE   default: config/site-secrets.local.yaml
  GREENFIELD_CONFIG_MODE   file | prompt | hybrid (default)

Wizard:  ./scripts/greenfield-install.sh config prompt
         ./scripts/greenfield-install.sh config prompt --minimal   # Phases 1–2
         ./scripts/greenfield-install.sh config prompt --full       # + Slack/AAP (default)
EOF
    ;;
  *)
    echo "usage: $0 {init|prompt|validate|show|load|apply|ensure}" >&2
    exit 1
    ;;
esac
fi
