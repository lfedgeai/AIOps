"""Enrichment slot — C1–C4 BM25 corpora (real retrieval).

Purpose
-------
Attach non-telemetry context before detection: docs (C2), architecture (C3),
dependencies (C4), source (C1) — retrieved with BM25 over the git-pinned
demo corpus, chunked 900/120 (eval parity; TF-IDF upgraded to BM25).
The active sets are controlled live from the UI (C1–C4 buttons →
``PUT /context``).

How it connects
---------------
- Runs first in ``orchestrator``; writes ``ctx["enrichment"]``.
- Consumed by detect / RCA prompts (and available context for tool loops).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from runtime_harness.models import Stage, StageResult
from runtime_harness.rag import CONTEXT_GLOBS, get_active_sets, retrieve
from runtime_harness.slots.base import Slot, SlotContext


class ContextCEnrichment(Slot):
    """C1–C4 BM25 retrieval — UI-toggleable context sets."""

    name = "enrichment"
    plugin = "context_c"

    def run(self, ctx: SlotContext) -> StageResult:
        started = datetime.now(timezone.utc)
        sets = get_active_sets()
        focus = ctx.get("target") or "(from telemetry adoption)"
        query = f"{focus} service failure error architecture dependencies configuration"
        retrieved = retrieve(sets, query, k=5)

        docs: list[str] = []
        for cs in sets:
            label = CONTEXT_GLOBS[cs]["label"]
            for chunk in retrieved.get(cs, [])[:3]:
                docs.append(f"[C{cs}/{label}] {chunk[:600]}")

        ctx["enrichment"] = {
            "context_sets": sets,
            "docs": docs or ["no corpus content retrieved"],
            "query": query,
        }
        finished = datetime.now(timezone.utc)
        return StageResult(
            stage=Stage.ENRICH,
            ok=True,
            summary=f"BM25 C{'+'.join(sets)} for '{focus}' ({len(docs)} chunks)",
            detail=ctx["enrichment"],
            started_at=started,
            finished_at=finished,
            duration_s=(finished - started).total_seconds(),
        )


def build_enrichment(config: dict[str, Any]) -> Slot:
    plugin = config.get("plugin", "context_c")
    if plugin != "context_c":
        raise ValueError(f"Unknown enrichment plugin: {plugin}")
    return ContextCEnrichment(config)
