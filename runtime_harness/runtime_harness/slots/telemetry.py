"""Telemetry slot — adopt applications and collect OTEL signals.

Purpose
-------
Discover workloads from ClickHouse ``ServiceName`` (self-adoption) and pull
logs/metrics/traces for the focused service into ``ctx["signals"]``.

How it connects
---------------
- First data stage after enrichment in ``orchestrator``.
- Uses ``telemetry/clickhouse.py`` (same queries as the MCP server).
- Detect / RCA / remediate consume the signals bag; no hard-coded app registry.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from runtime_harness.models import Stage, StageResult
from runtime_harness.slots.base import Slot, SlotContext
from runtime_harness.telemetry.clickhouse import (
    ClickHouseClient,
    clickhouse_http_url_from_config,
    list_services,
    search_logs,
    search_metrics,
    search_traces,
)


class ClickHouseOtelTelemetry(Slot):
    """Ingestion / adoption slot — OTEL → ClickHouse."""

    name = "telemetry"
    plugin = "clickhouse_otel"

    def run(self, ctx: SlotContext) -> StageResult:
        started = datetime.now(timezone.utc)
        url = clickhouse_http_url_from_config(self.config)
        database = str(self.config.get("database", "otel"))
        client = ClickHouseClient(base_url=url, database=database)

        inventory_raw = list_services(client=client)
        try:
            inventory = json.loads(inventory_raw)
        except json.JSONDecodeError:
            inventory = {"source": "unknown", "services": [], "raw": inventory_raw}

        adopted = [
            s.get("service") for s in inventory.get("services") or [] if s.get("service")
        ]
        # Empty live inventory → keep stub demo services so local cycles still adopt.
        if not adopted and inventory.get("source") == "clickhouse":
            inventory = {
                "source": "stub",
                "services": [
                    {"service": "cart", "signal_count": 0},
                    {"service": "checkout", "signal_count": 0},
                    {"service": "payment", "signal_count": 0},
                ],
                "note": "ClickHouse reachable but no ServiceName rows yet — stub adopt list",
            }
            adopted = [s["service"] for s in inventory["services"]]

        # Focus: operator hint if set; else first telemetry-adopted service.
        focus = ctx.get("target") or None
        if not focus and adopted:
            focus = adopted[0]
        if focus:
            ctx["target"] = focus

        logs = search_logs(service_name=focus, limit=10, client=client) if focus else "[]"
        traces = search_traces(service_name=focus, limit=10, client=client) if focus else "[]"
        metrics = search_metrics(service_name=focus, limit=10, client=client) if focus else "[]"
        # Cross-service evidence: errors mentioning the focus in OTHER services
        # (callers report a dead service — the eval agents search this way too).
        related = search_logs(query=focus or "", limit=10, client=client) if focus else "[]"

        source = inventory.get("source", "stub")
        signals = {
            "source": source,
            "namespace": ctx.get("namespace", "otel-demo"),
            "target": focus,
            "adopted_services": adopted,
            "window_s": 120,
            "injected_fault": ctx.get("injected_fault"),
            "logs": logs,
            "traces": traces,
            "metrics": metrics,
            "related_error_logs": related,
        }
        ctx["signals"] = signals
        ctx["adopted_services"] = adopted
        finished = datetime.now(timezone.utc)
        n = len(adopted)
        return StageResult(
            stage=Stage.TELEMETRY,
            ok=True,
            summary=(
                f"Adopted {n} service(s) via {source}"
                + (f"; focus={focus}" if focus else "")
            ),
            detail={
                "source": source,
                "adopted_services": adopted,
                "focus": focus,
                "inventory": inventory,
            },
            started_at=started,
            finished_at=finished,
            duration_s=(finished - started).total_seconds(),
        )


def build_telemetry(config: dict[str, Any]) -> Slot:
    plugin = config.get("plugin", "clickhouse_otel")
    if plugin != "clickhouse_otel":
        raise ValueError(f"Unknown telemetry plugin: {plugin}")
    return ClickHouseOtelTelemetry(config)
