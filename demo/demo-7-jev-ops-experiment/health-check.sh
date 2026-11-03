#!/usr/bin/env bash
# Check the Tier 2 testbed. Exits non-zero and names the failing check.
cd "$(dirname "$0")"
exec .venv/bin/python -m tier2 health
