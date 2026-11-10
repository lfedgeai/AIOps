#!/usr/bin/env python3
"""Set feature flag via OpenShift ConfigMap patch (workaround for flagd-ui write 404).

Use when FLAGD HTTP API write returns 404. Patches flagd-config ConfigMap
and restarts the flagd deployment to pick up the new config.

  python scripts/set-flag-openshift.py cartFailure on
  python scripts/set-flag-openshift.py cartFailure off

Env:
  FLAGD_CONFIGMAP_NAMESPACE: namespace of flagd-config (default: otel-demo)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: set-flag-openshift.py <flag_name> <variant>", file=sys.stderr)
        return 1
    flag_name, variant = sys.argv[1], sys.argv[2]
    ns = os.environ.get("FLAGD_CONFIGMAP_NAMESPACE", "otel-demo")
    cm_name = "flagd-config"
    key = "demo.flagd.json"

    result = subprocess.run(
        ["oc", "get", "configmap", cm_name, "-n", ns, "-o", "json"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"[set-flag] Failed to get ConfigMap: {result.stderr}", file=sys.stderr)
        return 1

    data = json.loads(result.stdout)
    flagd_json = json.loads(data["data"][key])
    flags = flagd_json.get("flags", {})

    if flag_name not in flags:
        print(f"[set-flag] Flag {flag_name} not found; available: {list(flags.keys())}", file=sys.stderr)
        return 1

    flags[flag_name]["defaultVariant"] = variant
    flagd_json["flags"] = flags
    patched = json.dumps(flagd_json, indent=2)

    path = "/tmp/flagd-config-patched.json"
    with open(path, "w") as f:
        f.write(patched)

    yaml_result = subprocess.run(
        ["oc", "create", "configmap", cm_name, f"--from-file={key}={path}", "-n", ns, "--dry-run=client", "-o", "yaml"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if yaml_result.returncode != 0:
        print(f"[set-flag] Failed to generate ConfigMap: {yaml_result.stderr}", file=sys.stderr)
        return 1
    result = subprocess.run(
        ["oc", "apply", "-f", "-"],
        input=yaml_result.stdout,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"[set-flag] Failed to apply: {result.stderr}", file=sys.stderr)
        return 1

    subprocess.run(
        ["oc", "rollout", "restart", "deployment/flagd", "-n", ns],
        capture_output=True,
        check=True,
        cwd=str(ROOT),
    )
    subprocess.run(
        ["oc", "rollout", "status", "deployment/flagd", "-n", ns, "--timeout=120s"],
        capture_output=True,
        cwd=str(ROOT),
    )
    print(f"[set-flag] Set {flag_name}={variant} in {ns}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
