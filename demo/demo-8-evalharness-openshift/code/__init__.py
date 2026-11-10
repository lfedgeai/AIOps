"""Agentic AIOps code package: harness, agents, tools."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def __getattr__(name: str):
    """
    MLflow (and other tools) expect ``code.InteractiveConsole`` from the **stdlib** ``code`` module.
    This package shadows that module name; delegate ``InteractiveConsole`` to the stdlib file.
    """
    if name == "InteractiveConsole":
        pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
        for base in (sys.base_prefix, getattr(sys, "exec_prefix", "")):
            if not base:
                continue
            root = Path(base)
            for sub in ("lib", "lib64"):
                stdlib_code_py = root / sub / f"python{pyver}" / "code.py"
                if stdlib_code_py.is_file():
                    spec = importlib.util.spec_from_file_location("_stdlib_code_for_interactive", stdlib_code_py)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        return getattr(mod, "InteractiveConsole")
        raise AttributeError("InteractiveConsole: stdlib code.py not found")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
