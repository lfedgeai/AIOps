#!/usr/bin/env python3
"""
Ansible Automation Platform MCP server (in-cluster).

Generic automation surface: launch pre-approved job templates, poll status.
Investigation MCPs (NetObserv, OpenShift, …) stay read-focused; remediation and
other governed tasks go through this server.
"""

from __future__ import annotations

import json
import os

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover
    raise SystemExit("mcp package missing — pip install requirements.txt") from e

from aap_client import (
    format_launch_result,
    job_status,
    launch_job_template,
    list_job_templates,
)

mcp = FastMCP(
    "ansible-automation",
    host=os.environ.get("MCP_HOST", "0.0.0.0"),
    port=int(os.environ.get("MCP_PORT", "8080")),
    instructions=(
        "Governed Ansible Automation Platform (AAP) job launcher. "
        "Use only after the user explicitly confirms the action in the current message "
        "(pass confirmed=true). "
        "Do not launch jobs during investigation or diagnosis turns. "
        "NetObserv demo templates: netobserv-heal-db-path (latency/chaos path), "
        "netobserv-restore-policy (NetworkPolicy / microsegmentation). "
        "Pass extra_vars as JSON when the playbook expects them (e.g. {\"confirmed\": true}). "
        "Quote AAP_JOB_ID and verdict lines from the tool result to the user."
    ),
)


def _parse_extra_vars(extra_vars_json: str = "") -> dict:
    raw = (extra_vars_json or "").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("extra_vars_json must be a JSON object")
    return data


@mcp.tool()
def ansible_list_job_templates(organization: str = "") -> str:
    """
    List AAP job templates available to the launcher token (name + id).
    Use to discover approved automation before launch.
    """
    org = organization.strip() or os.environ.get("AAP_ORG", "netobserv-demo")
    items = list_job_templates(org or None)
    if not items:
        return f"(no job templates for organization={org or 'any'})"
    lines = [f"organization={org or 'any'}", "templates:"]
    for jt in items:
        lines.append(f"  - {jt.get('name')}\tid={jt.get('id')}")
    return "\n".join(lines)


@mcp.tool()
def ansible_launch_job(
    job_template: str,
    confirmed: bool = False,
    extra_vars_json: str = '{"confirmed": true}',
    poll: bool = True,
) -> str:
    """
    Launch an AAP job template by name. Requires confirmed=true after explicit user approval.

    job_template: e.g. netobserv-heal-db-path, netobserv-restore-policy, or future templates.
    extra_vars_json: JSON object passed to the playbook (default includes confirmed=true).
    poll: wait for job completion (default true).
    """
    if not confirmed:
        return (
            "refused: set confirmed=true only after the user explicitly confirms this "
            "automation in the current message."
        )
    name = (job_template or "").strip()
    if not name:
        return "refused: job_template name required"
    try:
        extra = _parse_extra_vars(extra_vars_json)
    except (json.JSONDecodeError, ValueError) as e:
        return f"refused: invalid extra_vars_json: {e}"
    extra.setdefault("confirmed", True)
    try:
        result = launch_job_template(name, extra, poll=poll)
        return format_launch_result(result)
    except Exception as e:
        return f"AAP launch failed: {e}"


@mcp.tool()
def ansible_job_status(job_id: int, include_stdout: bool = True) -> str:
    """Poll an existing AAP job by id (status + optional stdout highlights)."""
    try:
        data = job_status(int(job_id), include_stdout=include_stdout)
        lines = [
            f"AAP_JOB_ID={data.get('job_id')}",
            f"AAP_JOB_STATUS={data.get('status')}",
            f"AAP_JOB_TEMPLATE={data.get('job_template')}",
            f"AAP_JOB_URL={data.get('job_url')}",
        ]
        for h in data.get("highlights") or []:
            lines.append(h)
        return "\n".join(lines)
    except Exception as e:
        return f"AAP job status failed: {e}"


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
