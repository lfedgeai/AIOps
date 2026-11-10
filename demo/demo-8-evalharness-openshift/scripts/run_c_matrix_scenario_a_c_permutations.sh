#!/usr/bin/env bash
# Compatibility wrapper: scenario A x agents x 3 faults x all CONTEXT_C subsets.
exec "$(dirname "$0")/run_matrix.sh" permutations
