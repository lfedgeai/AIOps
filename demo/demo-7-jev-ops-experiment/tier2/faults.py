"""Fault injection through the demo's flagd flags (via the flagd-ui API).

Verified against OpenTelemetry Demo 3.1.0: flagd hot-reloads the flag file within
about a second of a write. Two flag shapes exist in demo.flagd.json:
  - plain flags: enabled by setting `defaultVariant`;
  - flags with a `targeting.if` rule (productCatalogFailure): enabled by setting
    the matched branch `targeting.if[1]`; `defaultVariant` is overridden by targeting.
The flagd-ui write replaces the whole config, so every change is read-modify-write.
"""
from __future__ import annotations

import copy
import json
import urllib.request
from typing import Protocol

from tier2.scenarios import FaultSpec

OFF = "off"


class FaultError(RuntimeError):
    pass


class FlagStore(Protocol):
    def read(self) -> dict: ...
    def write(self, config: dict) -> None: ...


class HttpFlagStore:
    """flagd-ui API behind Envoy: GET /feature/api/read, POST /feature/api/write."""

    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 10.0):
        self.base, self.timeout = base_url.rstrip("/"), timeout

    def read(self) -> dict:
        with urllib.request.urlopen(f"{self.base}/feature/api/read", timeout=self.timeout) as r:
            return json.load(r)

    def write(self, config: dict) -> None:
        req = urllib.request.Request(
            f"{self.base}/feature/api/write", data=json.dumps({"data": config}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=self.timeout).read()


def _has_targeting(flag: dict) -> bool:
    return isinstance(flag.get("targeting"), dict) and isinstance(flag["targeting"].get("if"), list) \
        and len(flag["targeting"]["if"]) == 3


def effective_variant(flag: dict) -> str:
    """The variant a client gets when the rule matches (targeting flags) or by default."""
    return flag["targeting"]["if"][1] if _has_targeting(flag) else flag["defaultVariant"]


def _set_variant(flag: dict, variant: str) -> None:
    if _has_targeting(flag):
        flag["targeting"]["if"][1] = variant
    else:
        flag["defaultVariant"] = variant


class FaultInjector:
    def __init__(self, store: FlagStore):
        self.store = store
        self._previous: dict[str, dict] = {}  # flag name -> original flag entry

    def active(self) -> dict[str, str]:
        """Flags not in their `off` variant: {flag: variant}."""
        flags = self.store.read()["flags"]
        return {n: effective_variant(f) for n, f in flags.items() if effective_variant(f) != OFF}

    def inject(self, fault: FaultSpec) -> None:
        cfg = self.store.read()
        flags = cfg["flags"]
        if fault.flag not in flags:
            raise FaultError(f"unknown flag {fault.flag!r}")
        if fault.variant not in flags[fault.flag]["variants"]:
            raise FaultError(f"flag {fault.flag!r} has no variant {fault.variant!r}")
        self._previous.setdefault(fault.flag, copy.deepcopy(flags[fault.flag]))
        _set_variant(flags[fault.flag], fault.variant)
        self.store.write(cfg)
        got = effective_variant(self.store.read()["flags"][fault.flag])
        if got != fault.variant:
            raise FaultError(f"read-back mismatch for {fault.flag}: wanted {fault.variant!r}, got {got!r}")

    def reset(self) -> list[str]:
        """Restore every flag this injector changed to exactly its previous entry."""
        if not self._previous:
            return []
        cfg = self.store.read()
        for name, prev in self._previous.items():
            cfg["flags"][name] = copy.deepcopy(prev)
        self.store.write(cfg)
        restored = list(self._previous)
        self._previous.clear()
        return restored


def reset_faults(store: FlagStore) -> list[str]:
    """Idempotent global reset: every flag back to `off`. Returns the flags changed."""
    cfg = store.read()
    changed = []
    for name, flag in cfg["flags"].items():
        if effective_variant(flag) != OFF:
            _set_variant(flag, OFF)
            changed.append(name)
    if changed:
        store.write(cfg)
    return changed
