"""ClickHouse / OTEL telemetry access (Oct-12 leading path).

Used by the telemetry slot for app self-adoption and by the MCP server for
``list_services`` / ``search_logs`` / ``search_metrics`` / ``search_traces``.
"""

from runtime_harness.telemetry.clickhouse import (
    ClickHouseClient,
    list_services,
    search_logs,
    search_metrics,
    search_traces,
)

__all__ = [
    "ClickHouseClient",
    "list_services",
    "search_logs",
    "search_metrics",
    "search_traces",
]
