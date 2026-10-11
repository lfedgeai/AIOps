#!/bin/bash
# Ensure local MLflow tracking server on :5050 (sqlite + ./mlartifacts).
# Runs as a systemd --user service so it survives Cursor/agent shell exit.
# Do NOT use the OpenShift agentic-aiops/mlflow service for harness runs.
#
# Usage: source this file, or: ./scripts/ensure_local_mlflow.sh
# Env: MLFLOW_PORT (default 5050), ROOT (repo root)

_ensure_local_mlflow() {
  local root="${ROOT:-}"
  if [[ -z "$root" ]]; then
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
  local port="${MLFLOW_PORT:-5050}"
  local uri="${MLFLOW_TRACKING_URI:-http://127.0.0.1:${port}}"
  local unit="mlflow-local.service"

  # Refuse OpenShift / in-cluster MLflow URLs unless explicitly overridden
  if [[ "${ALLOW_CLUSTER_MLFLOW:-0}" != "1" ]] && [[ "$uri" == *agentic-aiops* || "$uri" == *apps.*localdomain* || "$uri" == *"svc.cluster.local"* ]]; then
    echo "Refusing cluster MLflow URI: $uri" >&2
    echo "Harness uses local MLflow only. Unset MLFLOW_TRACKING_URI or set http://127.0.0.1:${port}" >&2
    echo "(Set ALLOW_CLUSTER_MLFLOW=1 only for emergency overrides.)" >&2
    return 1
  fi

  export MLFLOW_TRACKING_URI="http://127.0.0.1:${port}"
  export MLFLOW_TRACKING_INSECURE_TLS="${MLFLOW_TRACKING_INSECURE_TLS:-false}"

  if curl -sf --max-time 3 "http://127.0.0.1:${port}/" >/dev/null 2>&1 \
    || curl -sf --max-time 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "Local MLflow already running at http://127.0.0.1:${port}/"
    return 0
  fi

  mkdir -p "$root/mlartifacts" "$root/out" "$HOME/.config/systemd/user"

  # Persist via systemd --user (survives agent shell teardown; disables broken Huey jobs)
  cat >"$HOME/.config/systemd/user/${unit}" <<EOF
[Unit]
Description=Local MLflow tracking server (AIOps harness)
After=network.target

[Service]
Type=simple
WorkingDirectory=${root}
Environment=MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false
Environment=MLFLOW_TRACKING_URI=http://127.0.0.1:${port}
ExecStart=/usr/bin/python3.13 -m mlflow server \\
  --backend-store-uri sqlite:///${root}/mlflow_local.db \\
  --default-artifact-root ${root}/mlartifacts \\
  --serve-artifacts \\
  --host 0.0.0.0 \\
  --port ${port} \\
  --workers 1 \\
  --allowed-hosts * \\
  --cors-allowed-origins *
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF

  echo "Starting local MLflow via systemd --user (${unit}) on :${port}..."
  systemctl --user daemon-reload
  # Enable lingering so user services keep running after logout (best-effort)
  loginctl enable-linger "$(id -un)" 2>/dev/null || true
  systemctl --user enable --now "$unit" 2>/dev/null || systemctl --user restart "$unit"

  local i
  for i in $(seq 1 40); do
    if curl -sf --max-time 2 "http://127.0.0.1:${port}/" >/dev/null 2>&1 \
      || curl -sf --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      systemctl --user is-active "$unit" >/dev/null && \
        echo "Local MLflow ready (systemd --user ${unit}):" || \
        echo "Local MLflow ready:"
      echo "  UI:  http://127.0.0.1:${port}/"
      echo "  LAN: http://$(hostname -I 2>/dev/null | awk '{print $1}'):${port}/"
      echo "  Logs: journalctl --user -u ${unit} -f"
      return 0
    fi
    sleep 1
  done

  echo "ERROR: local MLflow failed to start. Status:" >&2
  systemctl --user status "$unit" --no-pager >&2 || true
  journalctl --user -u "$unit" -n 40 --no-pager >&2 || true
  return 1
}

# When executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -euo pipefail
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  mkdir -p "$ROOT/out"
  _ensure_local_mlflow
fi
