#!/usr/bin/env python3
"""Pre-run health gate: cluster baseline clean + ClickHouse reachable.

Exit 0 when the target deployment (default cart) has no lingering harness faults
and telemetry backend responds. Exit 1 otherwise (matrix should abort the cell).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.harness.cluster_health import (
    check_clickhouse_reachable,
    enforce_cluster_baseline,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default=os.environ.get("HARNESS_FAULT_VARIANT", "cart"))
    ap.add_argument("--namespace", default=os.environ.get("K8S_FAULT_NAMESPACE", "otel-demo"))
    ap.add_argument("--no-recover", action="store_true", help="Check only; do not auto-recover")
    ap.add_argument("--wait-timeout", type=float, default=120.0)
    ap.add_argument("--skip-clickhouse", action="store_true")
    args = ap.parse_args()

    ok, issues = enforce_cluster_baseline(
        args.target,
        args.namespace,
        auto_recover=not args.no_recover,
        wait_ready=True,
        wait_timeout_sec=args.wait_timeout,
    )
    if not ok:
        print(f"[health-gate] FAIL cluster baseline ({args.target}/{args.namespace}):", flush=True)
        for issue in issues:
            print(f"  - {issue}", flush=True)
        return 1

    print(f"[health-gate] OK cluster baseline {args.target}/{args.namespace}", flush=True)

    if not args.skip_clickhouse:
        if not check_clickhouse_reachable():
            ch = os.environ.get("CLICKHOUSE_HTTP", "http://127.0.0.1:38123")
            print(f"[health-gate] FAIL ClickHouse unreachable at {ch}", flush=True)
            return 1
        print("[health-gate] OK ClickHouse", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
