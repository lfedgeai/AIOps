#!/usr/bin/env python3
"""Summarize a NetObserv evidence.json bundle for agent diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _num(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _policy_drop_signal(causes: list[Any]) -> bool:
    markers = (
        "network_policy",
        "netpol",
        "networkpolicy",
        "policy",
        "netfilter",
        "ovn",
        "drop_ct",
        "rp_filter",
    )
    for item in causes:
        if not isinstance(item, dict):
            continue
        cause = str(item.get("cause", "")).lower()
        if any(m in cause for m in markers):
            return True
    return False


def _ingress_allows_todo(policy_ev: dict[str, Any], todo_label: str = "todo") -> bool:
    peers = policy_ev.get("ingress_from_pod_labels") or []
    if not peers:
        return False
    for labels in peers:
        if isinstance(labels, dict) and labels.get("app") == todo_label:
            return True
    return False


def _policy_connectivity_incident(
    policy_ev: dict[str, Any],
    causes: list[Any],
    drops: int,
    elevated_rtt: bool,
    jdbc_issues: bool,
) -> bool:
    if policy_ev:
        todo_labels = policy_ev.get("todo_workload_labels") or {}
        todo_app = todo_labels.get("app", "todo")
        if policy_ev.get("db_policy_name") and not _ingress_allows_todo(policy_ev, todo_app):
            return True
    if drops <= 0:
        return False
    if _policy_drop_signal(causes):
        return True
    if jdbc_issues and not elevated_rtt and drops > 0:
        return True
    return False


def summarize(data: dict[str, Any]) -> str:
    net = data.get("network_evidence") or {}
    app = data.get("application_evidence") or {}
    policy_ev = data.get("policy_evidence") or {}
    rtt = net.get("rtt_ms") or {}
    avg = rtt.get("avg")
    p95 = rtt.get("p95")
    drops = int(net.get("dropped_packets") or 0) + int(net.get("dropped_flows") or 0)
    causes = net.get("drop_causes") or []
    errors = app.get("error_log_excerpts") or []
    err_blob = "\n".join(str(e) for e in errors).lower()
    jdbc_issues = any(
        k in err_blob
        for k in ("acquisition timeout", "agroal", "jdbc", "unable to acquire")
    )
    high_rtt = isinstance(avg, (int, float)) and avg >= 200
    high_p95 = isinstance(p95, (int, float)) and p95 >= 200
    elevated_rtt = bool(high_rtt or high_p95)
    policy_incident = _policy_connectivity_incident(
        policy_ev, causes, drops, elevated_rtt, jdbc_issues
    )

    lines: list[str] = []
    lines.append("# NetObserv evidence summary")
    lines.append("")
    lines.append("## Executive summary (use these headings in your reply)")
    lines.append("")
    lines.append("### Impact")
    if policy_incident:
        lines.append(
            "- User-visible: database connectivity failure on the todo → PostgreSQL path "
            "(API errors, JDBC pool timeouts) — enforcement/denial pattern, not elevated path latency."
        )
    elif jdbc_issues:
        lines.append(
            "- User-visible: application struggles to get DB connections "
            "(JDBC/Agroal acquisition timeouts under load)."
        )
    elif elevated_rtt:
        lines.append(
            "- User-visible: elevated latency on the todo → PostgreSQL path; "
            "API calls to DB-backed endpoints may be slow or time out."
        )
    else:
        lines.append(
            "- User-visible: check application errors and path latency; "
            "network metrics in this bundle are not strongly elevated."
        )
    lines.append("")
    lines.append("### Network (NetObserv)")
    lines.append(f"- Path: todo → postgresql:{data.get('db_port', '5432')} (namespace {data.get('app_namespace', 'todo-demo')})")
    lines.append(
        f"- Flow RTT (ms): min={_num(rtt.get('min'))} avg={_num(avg)} "
        f"p95={_num(p95)} max={_num(rtt.get('max'))}"
    )
    lines.append(f"- Dropped flows/packets: flows={net.get('dropped_flows', 0)} packets={net.get('dropped_packets', 0)}")
    lines.append(f"- Elevated RTT flag: {elevated_rtt}")
    if policy_ev.get("db_policy_name"):
        peers = policy_ev.get("ingress_from_pod_labels") or []
        todo_labels = policy_ev.get("todo_workload_labels") or {}
        lines.append(
            f"- DB NetworkPolicy `{policy_ev.get('db_policy_name')}` ingress peers: "
            f"{peers!s}; todo workload labels: {todo_labels!s}"
        )
    lines.append("")
    lines.append("### Application")
    lines.append(f"- Todo pod restarts (window): {app.get('todo_restart_count', 0)}")
    lines.append(f"- Pool/JDBC errors in logs: {jdbc_issues}")
    ds = (app.get("datasource_config") or "").strip()
    if ds and "postgresql" in ds.lower():
        lines.append("- Datasource: still targets PostgreSQL directly (no proxy hostname in excerpt).")
    elif ds:
        lines.append("- Datasource excerpt present (see detail below).")
    else:
        lines.append("- Datasource excerpt: missing from bundle.")
    lines.append("")
    lines.append("### Likely cause class")
    if policy_incident:
        lines.append(
            "- **Connectivity denied** by microsegmentation (NetworkPolicy on PostgreSQL) — "
            "ingress peer labels do not admit the todo workload, or ingress is empty. "
            "This is not a latency/degradation incident on an open path."
        )
    elif elevated_rtt and drops > 0:
        lines.append(
            "- Network degradation on the existing app→DB path (high RTT plus drops); "
            "not evidence of a new in-path proxy or topology change."
        )
    elif elevated_rtt:
        lines.append(
            "- Network latency on the existing app→DB path; topology unchanged in datasource config."
        )
    elif jdbc_issues:
        lines.append(
            "- Application-layer pool exhaustion/timeouts; correlate with network RTT above."
        )
    else:
        lines.append("- Insufficient strong signals in this bundle; gather live probes or a longer capture.")
    lines.append("")
    lines.append("### Suggested follow-up (diagnostic only)")
    if policy_incident:
        lines.append(
            "1. Review NetworkPolicy ingress on PostgreSQL pods in the application namespace."
        )
        lines.append(
            "2. Verify allowed ingress peer labels match the todo workload (expected app=todo)."
        )
        lines.append(
            "3. Correlate denied flows and drop causes in Observe → Network Traffic (Console)."
        )
        lines.append(
            "4. Escalate to platform/security operations with policy diff and this evidence bundle."
        )
        lines.append(
            "5. Do **not** treat this as a network-latency or tc/heal remediation case — fix policy/labels."
        )
    else:
        lines.append(
            "1. Confirm todo and PostgreSQL workloads are Ready in the application namespace."
        )
        lines.append(
            "2. Re-measure API latency to a database-backed endpoint under comparable load."
        )
        lines.append(
            "3. If degradation persists, escalate to platform/network operations with this bundle."
        )
        lines.append(
            "4. Compare against a baseline capture from a known-healthy window when available."
        )
    lines.append("")
    lines.append("### Confidence")
    measured = []
    if rtt:
        measured.append("Flow RTT stats")
    if drops:
        measured.append("drop counts")
    if errors:
        measured.append("todo error log excerpts")
    lines.append(f"- Measured: {', '.join(measured) or 'limited fields in bundle'}")
    inferred_bits = []
    if policy_incident:
        inferred_bits.append(
            "NetworkPolicy/microsegmentation blocking todo→PostgreSQL (connectivity, not latency)"
        )
    elif elevated_rtt or drops:
        inferred_bits.append("network path degradation on the existing DB route")
    elif jdbc_issues:
        inferred_bits.append("application/pool stress (correlate with network signals above)")
    else:
        inferred_bits.append("cause class uncertain from this bundle alone")
    lines.append(f"- Inferred: {inferred_bits[0]}")
    lines.append("")
    lines.append("## Agent notes (do not paste into user reply)")
    lines.append(
        "- Diagnosis only: no remediation tools, batch-job names, or heal offers in the executive summary."
    )
    lines.append(
        "- Do not name fault-injection tooling (Kraken, chaos jobs, tc shaping) unless the user named them first."
    )
    if policy_incident:
        lines.append(
            "- Policy/connectivity incident: recommend NetworkPolicy/label correction — do NOT suggest ansible automation or tc cleanup."
        )
    lines.append("")
    lines.append("## Detail")
    lines.append(f"- app_namespace: {data.get('app_namespace', 'n/a')}")
    lines.append(f"- db_port: {data.get('db_port', 'n/a')}")
    if policy_ev:
        lines.append("")
        lines.append("## Policy evidence (snapshot)")
        lines.append(f"- db_policy_name: {policy_ev.get('db_policy_name', 'n/a')}")
        lines.append(f"- postgres_pod_selector: {policy_ev.get('postgres_pod_selector', {})}")
        lines.append(f"- ingress_from_pod_labels: {policy_ev.get('ingress_from_pod_labels', [])}")
        lines.append(f"- todo_workload_labels: {policy_ev.get('todo_workload_labels', {})}")
    lines.append("")
    lines.append("## Network evidence")
    lines.append(f"- total_db_flows: {net.get('total_db_flows', 0)}")
    lines.append(
        "- rtt_ms: "
        f"min={_num(rtt.get('min'))} avg={_num(rtt.get('avg'))} "
        f"p95={_num(rtt.get('p95'))} max={_num(rtt.get('max'))}"
    )
    lines.append(f"- dropped_flows: {net.get('dropped_flows', 0)}")
    lines.append(f"- dropped_packets: {net.get('dropped_packets', 0)}")
    causes = net.get("drop_causes") or []
    if causes:
        lines.append("- drop_causes:")
        for c in causes[:8]:
            if isinstance(c, dict):
                lines.append(
                    f"  - {c.get('cause', '?')}: "
                    f"flows={c.get('flows', 0)} packets={c.get('packets', 0)}"
                )
            else:
                lines.append(f"  - {c}")
    else:
        lines.append("- drop_causes: (none)")
    talkers = net.get("top_talkers_to_db") or []
    if talkers:
        lines.append("- top_talkers_to_db:")
        for t in talkers[:5]:
            if not isinstance(t, dict):
                continue
            lines.append(
                f"  - {t.get('src')} -> {t.get('dst')}:{t.get('dst_port')} "
                f"flows={t.get('flows', 0)} avg_rtt_ms={_num(t.get('avg_rtt_ms'))}"
            )
    lines.append("")
    lines.append("## Application evidence")
    lines.append(f"- todo_restart_count: {app.get('todo_restart_count', 0)}")
    ds = (app.get("datasource_config") or "").strip()
    if ds:
        # Prefer jdbc URL lines if present; else first non-empty lines.
        jdbc_lines = [ln for ln in ds.splitlines() if "jdbc" in ln.lower() or "datasource" in ln.lower()]
        show = jdbc_lines[:6] if jdbc_lines else ds.splitlines()[:8]
        lines.append("- datasource_config (excerpt):")
        for ln in show:
            lines.append(f"  {ln}")
    else:
        lines.append("- datasource_config: (missing)")
    errors = app.get("error_log_excerpts") or []
    lines.append(f"- error_log_excerpts: {len(errors)} line(s)")
    for ln in errors[:12]:
        lines.append(f"  | {ln[:200]}")
    lines.append("")
    lines.append("## Heuristic flags (observations only)")
    avg = rtt.get("avg")
    p95 = rtt.get("p95")
    high_rtt = isinstance(avg, (int, float)) and avg >= 200
    high_p95 = isinstance(p95, (int, float)) and p95 >= 200
    drops = int(net.get("dropped_packets") or 0) + int(net.get("dropped_flows") or 0)
    err_blob = "\n".join(str(e) for e in errors).lower()
    jdbc_issues = any(
        k in err_blob
        for k in ("acquisition timeout", "agroal", "jdbc", "unable to acquire")
    )
    lines.append(f"- elevated_flow_rtt: {bool(high_rtt or high_p95)}")
    lines.append(f"- packet_or_flow_drops_present: {drops > 0}")
    lines.append(f"- datasource_pool_errors_present: {jdbc_issues}")
    lines.append(f"- policy_connectivity_denial_suspected: {policy_incident}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "evidence",
        nargs="?",
        default="evidence/latest.json",
        help="Path to evidence.json (default: evidence/latest.json)",
    )
    ap.add_argument("--json", action="store_true", help="Emit compact JSON instead of Markdown")
    args = ap.parse_args()
    path = Path(args.evidence)
    if not path.is_file():
        print(f"error: evidence file not found: {path}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text())
    if args.json:
        net = data.get("network_evidence") or {}
        app = data.get("application_evidence") or {}
        rtt = net.get("rtt_ms") or {}
        out = {
            "app_namespace": data.get("app_namespace"),
            "db_port": data.get("db_port"),
            "rtt_ms": rtt,
            "dropped_flows": net.get("dropped_flows"),
            "dropped_packets": net.get("dropped_packets"),
            "drop_causes": net.get("drop_causes"),
            "top_talkers_to_db": net.get("top_talkers_to_db"),
            "todo_restart_count": app.get("todo_restart_count"),
            "error_log_excerpts": (app.get("error_log_excerpts") or [])[:12],
            "datasource_config": app.get("datasource_config"),
        }
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(summarize(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
