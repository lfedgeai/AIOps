#!/usr/bin/env python3
"""Minimal __ENV_VAR__ renderer for repo manifests (no external deps)."""
from __future__ import annotations

import os
import re
import sys

PATTERN = re.compile(r"__([A-Z0-9_]+)__")


def repl(match: re.Match[str]) -> str:
    key = match.group(1)
    if key not in os.environ:
        print(f"[render] warning: {key} is not set; leaving empty", file=sys.stderr)
    return os.environ.get(key, "")


def main() -> int:
    sys.stdout.write(PATTERN.sub(repl, sys.stdin.read()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
