#!/usr/bin/env bash
# Compatibility wrapper: scenario A x agents x 3 faults x C1-C4.
exec "$(dirname "$0")/run_matrix.sh" context
