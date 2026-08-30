#!/usr/bin/env bash
# Resolve DEMO_KIT_ROOT for greenfield scripts (source or run).
# Prefers: env → sibling demo-4-platform-kit.
set -euo pipefail

_gf_root() {
  if [[ -n "${GF_ROOT:-}" ]]; then
    printf '%s' "$GF_ROOT"
  else
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
  fi
}

resolve_demo_kit() {
  local root
  root="$(_gf_root)"
  if [[ -n "${DEMO_KIT_ROOT:-}" ]]; then
    DEMO_KIT_ROOT="$(cd "$DEMO_KIT_ROOT" && pwd)"
  elif [[ -d "$root/../demo-4-platform-kit/scripts" ]]; then
    DEMO_KIT_ROOT="$(cd "$root/../demo-4-platform-kit" && pwd)"
  else
    printf 'error: platform kit not found.\n' >&2
    printf '  export DEMO_KIT_ROOT=/path/to/demo-4-platform-kit\n' >&2
    printf '  (expected sibling: ../demo-4-platform-kit)\n' >&2
    return 1
  fi
  export DEMO_KIT_ROOT
  [[ -x "$DEMO_KIT_ROOT/scripts/install-netobserv-aws.sh" ]] || {
    printf 'error: invalid DEMO_KIT_ROOT=%s\n' "$DEMO_KIT_ROOT" >&2
    return 1
  }
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  resolve_demo_kit
  printf '%s\n' "$DEMO_KIT_ROOT"
fi
