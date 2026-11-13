"""MCP server exposing OTEL logs / metrics / traces tools.

Started with ``aiops-runtime mcp``. Shares ``telemetry.clickhouse`` with the
in-process telemetry slot so agents and the runtime see the same signal path.
"""

from runtime_harness.mcp.server import build_mcp, main

__all__ = ["build_mcp", "main"]
