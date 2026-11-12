"""Slot contract — pluggable pipeline stage interface.

Purpose
-------
Every stage (telemetry, enrichment, detect, RCA, remediate, verify) implements
``Slot.run(ctx) -> StageResult`` and reads/writes keys on the shared ``SlotContext``.

How it connects
---------------
- Factories in each ``slots/*.py`` module are called from ``assembly.Assembly``.
- ``Orchestrator`` invokes ``run`` in order and records ``StageResult``s.
- Cross-cutting config (models, prompts) is pulled from stores inside slot code,
  not hard-wired into this base class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from runtime_harness.models import StageResult


class SlotContext(dict):
    """Mutable bag passed through the pipeline stages."""


class Slot(ABC):
    name: str = "slot"
    plugin: str = "base"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def run(self, ctx: SlotContext) -> StageResult:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"slot": self.name, "plugin": self.plugin, "config_keys": sorted(self.config.keys())}
