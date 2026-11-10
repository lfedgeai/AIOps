#!/usr/bin/env bash
# Compatibility wrapper: scenario A x agents x 3 faults x C1-C4.
# INCLUDE_C0=1 adds the no-context baseline.
exec "$(dirname "$0")/run_matrix.sh" context
