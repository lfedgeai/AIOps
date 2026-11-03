#!/usr/bin/env bash
# Tier 2 setup: Python env, OpenTelemetry Demo at the pinned tag, images. Add --start to run it.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q -r requirements.txt
[ -f .env ] || { cp .env.example .env; echo "created .env: add your JEV_API_KEY"; }
exec tier2/testbed/setup.sh "$@"
