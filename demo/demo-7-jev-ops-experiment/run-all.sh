#!/usr/bin/env bash
# Whole experiment in one go: venv -> install -> tests -> key check -> batch run -> report.
#
#   ./run-all.sh --check      set up and run the tests only (no API key, no API calls)
#   ./run-all.sh              70 runs (10 per scenario)
#   ./run-all.sh 7            7 runs (one per scenario)
#   ./run-all.sh 70 --no-logs metrics only, to compare with a run that has logs
#   ./run-all.sh 50 --seed 1234 --scenario contradictory   extra args go to run.py
#   ./run-all.sh 70 --seed 1000 --engine claude-sonnet     baseline LLM instead of Jev
#                             (engines: jev, claude-haiku, claude-sonnet, claude-opus, ollama:<model>)
#
# Environment: SKIP_TESTS=1 skips pytest. PYTHON=/path/to/python picks the interpreter.
set -euo pipefail
cd "$(dirname "$0")"

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then CHECK_ONLY=1; shift; fi

RUNS="${1:-70}"
[[ $# -gt 0 ]] && shift
[[ "$RUNS" =~ ^[0-9]+$ && "$RUNS" -ge 1 ]] || { echo "usage: $0 [--check] [runs] [run.py args...]" >&2; exit 1; }

ENGINE=jev
prev=""
for arg in "$@"; do
  [[ "$prev" == "--engine" ]] && ENGINE="$arg"
  [[ "$arg" == --engine=* ]] && ENGINE="${arg#--engine=}"
  prev="$arg"
done

step() { printf '\n== %s ==\n' "$1"; }
die() { printf 'error: %s\n' "$1" >&2; exit "${2:-1}"; }

step "1/5 Python environment"
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
  done
fi
[[ -n "$PY" ]] || die "Python 3.10 or newer is required and was not found. Install it, or set PYTHON=/path/to/python."
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || die "Python 3.10 or newer is required; $PY is $("$PY" -c 'import platform; print(platform.python_version())')."

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating .venv with $PY"
  "$PY" -m venv .venv || die "could not create a virtual environment. On Debian/Ubuntu: sudo apt install python3-venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --disable-pip-version-check -r requirements.txt
echo "OK: $(python --version), dependencies installed"

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  step "2/5 Unit tests (offline, no API key needed)"
  python -m pytest -q
else
  step "2/5 Unit tests (skipped)"
fi

if [[ ! -f .env && -z "${JEV_API_KEY:-}" ]]; then
  cp .env.example .env
  echo "Created .env from .env.example."
fi
KEY_OK=0
if [[ "$ENGINE" != "jev" && "$CHECK_ONLY" != "1" ]]; then
  # Baseline engines check their own configuration before the first call.
  step "3/5 Engine: $ENGINE (no Jev key needed)"
  KEY_OK=1
else
step "3/5 Jev configuration"
python - <<'PY' && KEY_OK=1 || true
import sys
from jev.client import ConfigError, load_config
try:
    c = load_config()
except ConfigError as e:
    sys.exit(f"{e}")
print("OK: key found, model", c.model)
PY
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
  [[ "$KEY_OK" == "1" ]] || echo "(No key yet. That is fine for --check.)"
  printf '\nSetup check passed. Next: add your key to .env, then ./run-all.sh 7   (7 API calls)\n'
  exit 0
fi
[[ "$KEY_OK" == "1" ]] || exit 2

step "4/5 Experiment: $ENGINE ($RUNS runs = $RUNS model calls)"
python run.py --runs "$RUNS" "$@"

step "5/5 Report"
python report.py
