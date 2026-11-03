#!/usr/bin/env bash
# Check the Tier 3 testbed. Exits non-zero and names the failing check.
cd "$(dirname "$0")"
exec .venv/bin/python -m tier3 health
