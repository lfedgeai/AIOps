#!/usr/bin/env bash
# Tier 3 setup: Python env, pre-check, kind cluster, demo chart, health check.
# Needs Docker, kind, kubectl, helm and 12 GB free memory; Tier 2 must be stopped.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q -r requirements.txt
[ -f .env ] || { cp .env.example .env; echo "created .env: add your JEV_API_KEY"; }
tier3/testbed/create.sh
PYTHON=.venv/bin/python tier3/testbed/install.sh
echo "Waiting 3 minutes for span metrics before the health check..."
sleep 180
exec tier3/testbed/health-check.sh
