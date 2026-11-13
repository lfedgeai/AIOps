"""CLI entry point — ``aiops-runtime`` commands.

Purpose
-------
Operator-facing commands: show assembly, run a cycle, serve the status UI,
or start the telemetry MCP server.

How it connects
---------------
- Loads YAML via ``assembly.load_assembly_config`` → ``Assembly``.
- ``serve`` launches ``api:app`` (prompts + models + cycle UI).
- ``mcp`` delegates to ``mcp.server.main`` (logs/metrics/traces tools).
"""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from runtime_harness.assembly import Assembly, load_assembly_config
from runtime_harness.orchestrator import Orchestrator, bootstrap_status
from runtime_harness.status_store import get_store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aiops-runtime",
        description="LF Edge AIOps Runtime Harness (pluggable Oct-12 assembly)",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to assembly YAML (default: config/default_assembly.yaml)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show-assembly", help="Print assembled components")
    p_status = sub.add_parser("status", help="Print current runtime status JSON")
    p_cycle = sub.add_parser("cycle", help="Run one detect→RCA→remediate→verify cycle")
    p_cycle.add_argument(
        "--target",
        default="",
        help="Optional focus service; omit to adopt from telemetry inventory",
    )
    p_cycle.add_argument("--namespace", default="otel-demo")
    p_cycle.add_argument("--context-c", default=None)
    p_serve = sub.add_parser("serve", help="Start status API + UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_mcp = sub.add_parser(
        "mcp",
        help="Start MCP server (list_services, search_logs/metrics/traces)",
    )
    p_mcp.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
    )
    p_mcp.add_argument("--host", default="127.0.0.1")
    p_mcp.add_argument("--port", type=int, default=8090)

    args = parser.parse_args(argv)
    console = Console()
    assembly = Assembly(load_assembly_config(args.config))
    bootstrap_status(assembly)

    if args.cmd == "show-assembly":
        table = Table(title=f"Assembly: {assembly.name}")
        table.add_column("Slot")
        table.add_column("Plugin")
        table.add_column("Agent")
        table.add_column("Role")
        for ref in assembly.component_refs():
            table.add_row(ref.slot, ref.plugin, ref.agent_id or "—", ref.role or "—")
        console.print(table)
        ev = assembly.evidence_report()
        if ev:
            console.print(f"[dim]Evidence: {ev}[/dim]")
        return 0

    if args.cmd == "status":
        console.print_json(get_store().get().model_dump_json())
        return 0

    if args.cmd == "cycle":
        orch = Orchestrator(assembly)
        status = orch.run_cycle(
            target=args.target,
            namespace=args.namespace,
            context_c=args.context_c,
        )
        console.print(f"[bold]{status.health.value}[/bold] — {status.message}")
        console.print_json(status.model_dump_json())
        return 0

    if args.cmd == "serve":
        import os

        import uvicorn

        if args.config:
            os.environ["RUNTIME_HARNESS_CONFIG"] = str(args.config)

        console.print(
            f"Status UI: http://{args.host}:{args.port}/  "
            f"(JSON: /status, /prompts, /models, cycle: POST /cycle)"
        )
        uvicorn.run(
            "runtime_harness.api:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
        return 0

    if args.cmd == "mcp":
        from runtime_harness.mcp.server import main as mcp_main

        return mcp_main(
            [
                "--transport",
                args.transport,
                "--host",
                args.host,
                "--port",
                str(args.port),
            ]
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
