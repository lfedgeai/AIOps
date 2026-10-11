#!/bin/bash
# Compatibility shim — prefer the explicit name:
#   scripts/run_c_matrix_scenario_a_c1-c4.sh
# Default matrix: scenario A × corpora C1–C4 (C0 via INCLUDE_C0=1).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_c_matrix_scenario_a_c1-c4.sh" "$@"
