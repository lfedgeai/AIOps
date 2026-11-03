#!/usr/bin/env bash
# Check the Tier 3 cluster and demo. Exits non-zero and names the failing check.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
exec .venv/bin/python -m tier3 health
