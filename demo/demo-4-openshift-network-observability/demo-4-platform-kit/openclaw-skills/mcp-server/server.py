#!/usr/bin/env python3
"""
NetObserv / OpenShift MCP server (in-cluster).

Investigation-only: flows, evidence, read-only K8s inspect, demo-path probes.
Governed remediation is via the separate ansible-automation MCP + AAP job templates.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
os.environ.setdefault("NETOBSERV_IN_CLUSTER", "1")


def _load_heal():
    """Load netobserv-cluster-heal.py (hyphenated filename from ConfigMap)."""
    path = HERE / "netobserv-cluster-heal.py"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    spec = importlib.util.spec_from_file_location("netobserv_cluster_heal", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_summarize():
    """Load summarize-evidence.py from the MCP ConfigMap."""
    path = HERE / "summarize-evidence.py"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    spec = importlib.util.spec_from_file_location("netobserv_summarize_evidence", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


heal = _load_heal()

ALLOWED_NS = frozenset(
    ns.strip()
    for ns in os.environ.get(
        "NETOBSERV_ALLOWED_NAMESPACES", "default,todo-demo,todo-client,openclaw"
    ).split(",")
    if ns.strip()
)


def _capture(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            fn(*args, **kwargs)
    except SystemExit as e:
        out = buf.getvalue().strip()
        code = e.code if isinstance(e.code, int) else 1
        return (out + f"\n[exit {code}]").strip()
    return buf.getvalue().strip() or "(no output)"


def _k() -> Any:
    return heal.K8s()


def _guard_ns(namespace: str) -> str:
    ns = (namespace or "").strip()
    if ns not in ALLOWED_NS:
        raise ValueError(
            f"namespace '{ns}' not allowed; choose one of: {sorted(ALLOWED_NS)}"
        )
    return ns


def _capture_proxy_base() -> str:
    return os.environ.get(
        "NETOBSERV_CAPTURE_PROXY_URL",
        "http://netobserv-capture-proxy.openclaw.svc.cluster.local:8080",
    ).rstrip("/")


def _fetch_latest_evidence() -> dict[str, Any]:
    import json
    import urllib.request

    url = f"{_capture_proxy_base()}/latest"
    with urllib.request.urlopen(url, timeout=120) as resp:
        return json.loads(resp.read().decode())


try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "mcp package missing — container should pip-install requirements.txt"
    ) from e

from mlflow_tracing import init_mlflow, trace_mcp_tool

init_mlflow()

mcp = FastMCP(
    "netobserv-openshift",
    host=os.environ.get("MCP_HOST", "0.0.0.0"),
    port=int(os.environ.get("MCP_PORT", "8080")),
    instructions=(
        "NetObserv DEMO PATH tools (todo→PostgreSQL) plus read-only OpenShift "
        "platform health and lean namespace inspect tools. "
        "EVIDENCE / DIAGNOSIS: call MCP netobserv_analyze_evidence (after capture). "
        "INVESTIGATION (user reports slowness): netobserv-investigate — triage, "
        "netobserv_capture_flows, then netobserv_analyze_evidence. "
        "Do NOT read or open summarize-evidence.py with the read tool — use MCP analyze. "
        "optional k8s_list_pods / k8s_list_deployments in todo-demo. "
        "During evidence analysis do NOT call remediation tools or suggest heal/restore. "
        "For 'is the OpenShift cluster healthy?' / ClusterOperators / nodes: use "
        "openshift_cluster_health on this server AND a read-only openshift-mcp tool "
        "(events_list or netobserv_get_flow_metrics) — NEVER answer from netobserv_status alone. "
        "netobserv_status / netobserv_probe_latency only cover the todo demo path. "
        "After user confirms remediation, use the ansible-automation MCP "
        "(ansible_launch_job) — NOT tools on this server. "
        "Do not invent kubeconfig or :6443 diagnostics — this server has in-cluster read access. "
        "Do not name chaos/lab jobs or fault injection unless the user did first."
    ),
)


def mcp_tool(fn):
    """Register an MCP tool with optional MLflow audit span."""
    return mcp.tool()(trace_mcp_tool(fn.__name__)(fn))



@mcp_tool
def netobserv_capture_flows(
    duration_seconds: int = 60,
    burst_load: bool = True,
) -> str:
    """
    Capture NetObserv flows on todo→postgresql:5432 (RTT + packet drops) for AI analysis.

    Use during investigation after triage shows healthy pods but elevated latency or
    ambiguous slowness. Default 60s with load burst for denser samples. Blocks until complete.

    After EVIDENCE_READY in the output, call netobserv_analyze_evidence next.
    Do NOT call during pure evidence re-analysis or remediation turns.
    """
    import urllib.error
    import urllib.request

    dur = max(30, min(int(duration_seconds or 60), 90))
    burst = 1 if burst_load else 0
    url = f"{_capture_proxy_base()}/capture?duration={dur}&burst={burst}"
    req = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=dur + 420) as resp:
            body = resp.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        err = e.read().decode(errors="replace")
        return f"capture failed HTTP {e.code}:\n{err[:4000]}"
    except Exception as e:
        return f"capture failed: {e}"
    return body.strip() or "(capture returned no output)"


@mcp_tool
def netobserv_analyze_evidence() -> str:
    """
    Analyze the latest NetObserv evidence bundle from capture-proxy.

    Call immediately after netobserv_capture_flows returns EVIDENCE_READY.
    Returns structured diagnosis text (RTT, drops, JDBC signals, policy hints).
    Do NOT use the read tool on summarize-evidence.py — this MCP tool runs analysis in-cluster.
    Diagnosis only — do not offer remediation in the same turn.
    """
    try:
        data = _fetch_latest_evidence()
    except Exception as e:
        return (
            f"analyze failed — no evidence available yet: {e}\n"
            "Run netobserv_capture_flows first, then call netobserv_analyze_evidence again."
        )
    try:
        summary = _load_summarize().summarize(data)
    except Exception as e:
        return f"summarize failed: {e}"
    return summary.strip() or "(analyze returned no output)"


@mcp_tool
def netobserv_probe_latency() -> str:
    """Probe todo API latency (returns code= / time= / idle_probe_summary). Demo path only — not OpenShift platform health. Do NOT mention loadgen, todo-client workloads, or synthetic traffic in user replies — real-incident narrative only."""
    return _capture(heal.probe, _k())


@mcp_tool
def netobserv_status() -> str:
    """Demo-path status: todo/postgresql pods, loadgen deployments, latency probe. NOT platform health. Do NOT call during evidence-bundle analysis — use k8s_list_* instead; this output includes batch-job inventory."""
    return _capture(heal.status, _k())


@mcp_tool
def openshift_cluster_health() -> str:
    """OpenShift platform health: ClusterVersion, ClusterOperators (flags degraded/progressing), and Node Ready. Use for 'cluster health' questions. Does not assess the NetObserv todo→DB path."""
    return _capture(heal.cluster_health, _k())


@mcp_tool
def netobserv_list_chaos() -> str:
    """List batch Jobs/Pods in default matching lab cleanup patterns. FOR REMEDIATION DISCUSSION ONLY — never call or suggest during evidence/diagnosis. Names are not root cause."""
    k = _k()
    jobs = k.get("/apis/batch/v1/namespaces/default/jobs").get("items") or []
    pods = k.get("/api/v1/namespaces/default/pods").get("items") or []
    jnames = [
        n
        for n in heal.list_names(jobs)
        if n.startswith("chaos") or n.startswith("netobserv-")
    ]
    pnames = []
    for name in heal.list_names(pods):
        low = name.lower()
        if any(x in low for x in ("chaos", "krkn", "network-chaos", "netobserv-tc")):
            pnames.append(name)
    lines = [
        "NOTE: listing leftovers only — do not treat these names as proof of root cause.",
        "jobs:",
    ] + ([f"  - {n}" for n in jnames] or ["  (none)"])
    lines += ["pods:"] + ([f"  - {n}" for n in pnames] or ["  (none)"])
    return "\n".join(lines)


@mcp_tool
def netobserv_policy_status() -> str:
    """Check PostgreSQL NetworkPolicy in todo-demo (microsegmentation). Use before/after policy restore."""
    return _capture(heal.policy_status, _k())


@mcp_tool
def k8s_list_pods(namespace: str, label_selector: str = "") -> str:
    """List pods in an allowlisted namespace (optional labelSelector)."""
    ns = _guard_ns(namespace)
    path = f"/api/v1/namespaces/{ns}/pods"
    if label_selector:
        path += f"?labelSelector={quote(label_selector)}"
    items = _k().get(path).get("items") or []
    if not items:
        return f"(no pods in {ns})"
    lines = []
    for p in items:
        md = p.get("metadata") or {}
        st = p.get("status") or {}
        lines.append(
            f"{md.get('name')}\tphase={st.get('phase')}\t"
            f"node={(p.get('spec') or {}).get('nodeName')}"
        )
    return "\n".join(lines)


@mcp_tool
def k8s_get_deployment(namespace: str, name: str) -> str:
    """Get a Deployment's readyReplicas in an allowlisted namespace."""
    ns = _guard_ns(namespace)
    d = _k().get(
        f"/apis/apps/v1/namespaces/{ns}/deployments/{name}",
        allow_404=True,
    )
    if not d:
        return f"deployment/{name} not found in {ns}"
    status = d.get("status") or {}
    spec = d.get("spec") or {}
    return (
        f"deployment/{name} ns={ns} "
        f"replicas={spec.get('replicas')} "
        f"readyReplicas={status.get('readyReplicas')} "
        f"availableReplicas={status.get('availableReplicas')}"
    )


@mcp_tool
def k8s_list_deployments(namespace: str) -> str:
    """List Deployments in an allowlisted namespace (replicas / ready)."""
    ns = _guard_ns(namespace)
    items = _k().get(f"/apis/apps/v1/namespaces/{ns}/deployments").get("items") or []
    if not items:
        return f"(no deployments in {ns})"
    lines = []
    for d in items:
        md = d.get("metadata") or {}
        st = d.get("status") or {}
        sp = d.get("spec") or {}
        lines.append(
            f"{md.get('name')}\treplicas={sp.get('replicas')}\t"
            f"ready={st.get('readyReplicas')}\tavailable={st.get('availableReplicas')}"
        )
    return "\n".join(lines)


@mcp_tool
def k8s_recent_events(namespace: str, limit: int = 20) -> str:
    """List recent events in an allowlisted namespace (newest first). Warning events first when present."""
    ns = _guard_ns(namespace)
    lim = max(1, min(int(limit or 20), 50))
    try:
        items = _k().get(
            f"/api/v1/namespaces/{ns}/events?limit={lim * 5}",
            allow_404=True,
        ).get("items") or []
    except Exception as e:
        return f"(events unavailable in {ns}: {e})"
    items = sorted(
        items,
        key=lambda e: (
            e.get("lastTimestamp")
            or e.get("eventTime")
            or e.get("metadata", {}).get("creationTimestamp")
            or ""
        ),
        reverse=True,
    )[:lim]
    if not items:
        return f"(no events in {ns})"
    lines = []
    for e in items:
        inv = e.get("involvedObject") or {}
        lines.append(
            f"{e.get('lastTimestamp') or e.get('eventTime') or '?'} "
            f"type={e.get('type')} reason={e.get('reason')} "
            f"{inv.get('kind')}/{inv.get('name')}: {e.get('message')}"
        )
    return "\n".join(lines)


def main() -> None:
    # Streamable HTTP; OpenClaw expects transport: streamable-http and path /mcp
    # host/port are set on the FastMCP constructor (not run()).
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
