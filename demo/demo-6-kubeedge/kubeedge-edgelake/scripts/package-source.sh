#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m compileall -q src/otel-edgelake-adapter src/edgelake-dashboard
echo 'Python syntax check passed.'
echo 'Build adapter: podman build -t otel-edgelake-adapter:demo src/otel-edgelake-adapter'
echo 'Build dashboard: podman build -t edgelake-dashboard:demo src/edgelake-dashboard'
