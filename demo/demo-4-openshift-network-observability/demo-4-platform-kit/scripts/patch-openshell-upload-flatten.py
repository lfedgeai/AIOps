#!/usr/bin/env python3
"""Idempotently patch openshell-sandbox uploadPathToRemote to flatten nested uploads."""
from __future__ import annotations

import sys
from pathlib import Path

MARKER = "NETOBSERV_FLATTEN_UPLOAD"
FLATTEN_CALL = (
    "\t\t\t// "
    + MARKER
    + ": OpenShell CLI nests openclaw-openshell-upload-* under remotePath.\n"
    "\t\t\tawait this.runRemoteShellScriptInternal({\n"
    '\t\t\t\tscript: \'remote="$1"; for d in "$remote"/openclaw-openshell-upload-*; '
    'do [ -d "$d" ] || continue; cp -a "$d"/. "$remote"/; rm -rf "$d"; done\',\n'
    "\t\t\t\targs: [remotePath]\n"
    "\t\t\t});\n"
)

NEEDLE = (
    "\t\t\tif (result.code !== 0) "
    'throw new Error(result.stderr.trim() || "openshell sandbox upload failed");\n'
    "\t\t});\n"
    "\t}\n"
    "\tasync maybeSeedRemoteWorkspace()"
)

REPLACEMENT = (
    "\t\t\tif (result.code !== 0) "
    'throw new Error(result.stderr.trim() || "openshell sandbox upload failed");\n'
    + FLATTEN_CALL
    + "\t\t});\n"
    "\t}\n"
    "\tasync maybeSeedRemoteWorkspace()"
)


def patch_file(path: Path) -> str:
    text = path.read_text()
    if MARKER in text:
        return "already-patched"
    if NEEDLE not in text:
        return "needle-missing"
    path.write_text(text.replace(NEEDLE, REPLACEMENT, 1))
    return "patched"


def main() -> int:
    roots = [
        Path("/opt/openclaw/config/npm/projects"),
    ]
    targets: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        targets.extend(root.rglob("**/openshell-sandbox/dist/index.js"))

    targets = sorted({t.resolve() for t in targets if t.is_file()})
    if not targets:
        print("ERROR: no openshell-sandbox dist/index.js found", file=sys.stderr)
        return 1

    ok = False
    for t in targets:
        result = patch_file(t)
        print(f"{result}: {t}")
        if result in ("patched", "already-patched"):
            ok = True
        elif result == "needle-missing":
            print(f"ERROR: patch needle not found in {t}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
