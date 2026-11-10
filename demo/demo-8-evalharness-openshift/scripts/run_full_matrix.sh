#!/usr/bin/env bash
# Compatibility wrapper: full matrix (agents x faults x scenarios 0/a/b/c).
exec "$(dirname "$0")/run_matrix.sh" full
