#!/usr/bin/env python3
"""In-cluster NetObserv flow capture + evidence bundle (stdlib only)."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Reuse K8s client from heal module (same directory / ConfigMap mount).
HERE = Path(__file__).resolve().parent


def _load_heal():
    import importlib.util

    path = HERE / "netobserv-cluster-heal.py"
    spec = importlib.util.spec_from_file_location("netobserv_cluster_heal", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


heal = _load_heal()

APP_NS = os.environ.get("APP_NS", "todo-demo")
CLIENT_NS = os.environ.get("CLIENT_NS", "todo-client")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
POLICY_NAME = os.environ.get("DB_POLICY_NAME", "allow-db-from-todo-only")
DEFAULT_DURATION = int(os.environ.get("CAPTURE_DURATION", "60"))
LOAD_BURST_REPLICAS = int(os.environ.get("LOAD_BURST_REPLICAS", "3"))
NETOBSERV_BIN = os.environ.get("NETOBSERV_BIN", "oc-netobserv")
KUBECONFIG = os.environ.get("KUBECONFIG", "/opt/kube/config")


def step(msg: str) -> None:
    print(f"==> {msg}")


def ok(msg: str) -> None:
    print(f"[ ok ] {msg}")


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def die(msg: str) -> None:
    print(f"[fail] {msg}", file=sys.stderr)
    raise SystemExit(1)


def netobserv_cmd(*args: str, cwd: Path | None = None, timeout: int = 300) -> None:
    env = os.environ.copy()
    if Path(KUBECONFIG).is_file():
        env["KUBECONFIG"] = KUBECONFIG
    bin_dir = str(Path(NETOBSERV_BIN).parent)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    proc = subprocess.run(
        [NETOBSERV_BIN, *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if out.strip():
        print(out.strip())
    if proc.returncode != 0:
        die(f"netobserv {' '.join(args)} failed (exit {proc.returncode})")


def find_netobserv_bin() -> str:
    for candidate in (
        NETOBSERV_BIN,
        "/usr/local/bin/oc-netobserv",
        "/oc-netobserv",
        "oc-netobserv",
    ):
        if shutil.which(candidate):
            return candidate
    die(
        f"{NETOBSERV_BIN} not found — capture proxy needs the NetObserv CLI image "
        "or NETOBSERV_BIN set"
    )
    return NETOBSERV_BIN


def scale_load_burst(k: heal.K8s, enable: bool) -> int | None:
    """Scale loadgen-heavy for denser flow samples during short captures."""
    path = f"/apis/apps/v1/namespaces/{CLIENT_NS}/deployments/loadgen-heavy"
    dep = k.get(path, allow_404=True)
    if not dep:
        warn("loadgen-heavy not found — skip load burst")
        return None
    prev = (dep.get("spec") or {}).get("replicas")
    if not enable:
        return prev
    target = max(LOAD_BURST_REPLICAS, int(prev or 1))
    if target == prev:
        return prev
    step(f"Burst load: scale loadgen-heavy replicas {prev} → {target}")
    _patch_merge(k, path, {"spec": {"replicas": target}})
    time.sleep(8)
    return prev


def _patch_merge(k: heal.K8s, path: str, body: dict[str, Any]) -> None:
    url = f"{k.server}{path}"
    data = json.dumps(body).encode()
    headers = {
        "Authorization": f"Bearer {k.token}",
        "Accept": "application/json",
        "Content-Type": "application/strategic-merge-patch+json",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="PATCH")
    with urllib.request.urlopen(req, context=k.ssl, timeout=60) as resp:
        resp.read()


def restore_load(k: heal.K8s, prev: int | None) -> None:
    if prev is None:
        return
    path = f"/apis/apps/v1/namespaces/{CLIENT_NS}/deployments/loadgen-heavy"
    dep = k.get(path, allow_404=True)
    if not dep:
        return
    cur = (dep.get("spec") or {}).get("replicas")
    if cur == prev:
        return
    step(f"Restore loadgen-heavy replicas → {prev}")
    _patch_merge(k, path, {"spec": {"replicas": prev}})


def has_col(cols: set[str], name: str) -> bool:
    return name in cols


def sql_one(conn: sqlite3.Connection, query: str) -> Any:
    try:
        row = conn.execute(query).fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None


def extract_evidence(db_path: Path, k: heal.K8s) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(flow);").fetchall()
    }
    dport = "DstPort" if has_col(cols, "DstPort") else ""
    sport = "SrcPort" if has_col(cols, "SrcPort") else ""
    if dport and sport:
        portf = f"(DstPort={DB_PORT} OR SrcPort={DB_PORT})"
    elif dport:
        portf = f"DstPort={DB_PORT}"
    else:
        portf = "1=1"

    total = int(sql_one(conn, f"SELECT COUNT(*) FROM flow WHERE {portf};") or 0)

    rtt_col = ""
    for c in ("TimeFlowRTTNs", "TimeFlowRttNs", "TimeFlowRTT", "FlowRtt"):
        if has_col(cols, c):
            rtt_col = c
            break

    def rtt_ms(agg: str) -> float | None:
        if not rtt_col:
            return None
        v = sql_one(
            conn,
            f"SELECT ROUND({agg}({rtt_col})/1000000.0,2) FROM flow "
            f"WHERE {portf} AND {rtt_col}>0;",
        )
        return float(v) if v is not None else None

    rtt_min = rtt_ms("MIN")
    rtt_avg = rtt_ms("AVG")
    rtt_max = rtt_ms("MAX")
    rtt_p95 = None
    if rtt_col:
        n = int(sql_one(conn, f"SELECT COUNT(*) FROM flow WHERE {portf} AND {rtt_col}>0;") or 0)
        if n > 0:
            off = min(n * 95 // 100, n - 1)
            v = sql_one(
                conn,
                f"SELECT ROUND({rtt_col}/1000000.0,2) FROM flow WHERE {portf} "
                f"AND {rtt_col}>0 ORDER BY {rtt_col} LIMIT 1 OFFSET {off};",
            )
            rtt_p95 = float(v) if v is not None else None

    drop_flows = 0
    drop_packets = 0
    if has_col(cols, "PktDropPackets"):
        drop_flows = int(
            sql_one(
                conn,
                f"SELECT COUNT(*) FROM flow WHERE {portf} AND PktDropPackets IS NOT NULL "
                f"AND PktDropPackets<>0;",
            )
            or 0
        )
        drop_packets = int(
            sql_one(
                conn,
                f"SELECT COALESCE(SUM(PktDropPackets),0) FROM flow WHERE {portf} "
                f"AND PktDropPackets IS NOT NULL AND PktDropPackets<>0;",
            )
            or 0
        )
    elif has_col(cols, "PktDropLatestDropCause"):
        drop_flows = int(
            sql_one(
                conn,
                f"SELECT COUNT(*) FROM flow WHERE {portf} AND PktDropLatestDropCause "
                f"IS NOT NULL AND PktDropLatestDropCause<>'';",
            )
            or 0
        )

    drop_causes: list[dict[str, Any]] = []
    if has_col(cols, "PktDropLatestDropCause"):
        try:
            for row in conn.execute(
                f"SELECT PktDropLatestDropCause, COUNT(*), COALESCE(SUM(PktDropPackets),0) "
                f"FROM flow WHERE {portf} AND PktDropLatestDropCause IS NOT NULL "
                f"AND PktDropLatestDropCause<>'' "
                f"GROUP BY PktDropLatestDropCause ORDER BY 2 DESC LIMIT 10;"
            ):
                drop_causes.append(
                    {"cause": row[0], "flows": row[1], "packets": row[2]}
                )
        except sqlite3.Error:
            pass

    conn.close()

    app_errors = ""
    try:
        logs = k.request(
            "GET",
            f"/api/v1/namespaces/{APP_NS}/pods?labelSelector=app=todo",
            allow_404=True,
        )
        items = (logs or {}).get("items") or []
        if items:
            pod = items[0]["metadata"]["name"]
            raw = k.request(
                "GET",
                f"/api/v1/namespaces/{APP_NS}/pods/{pod}/log?tailLines=800",
                raw=True,
                allow_404=True,
            )
            text = raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)
            lines = [
                ln
                for ln in text.splitlines()
                if any(
                    x in ln.lower()
                    for x in (
                        "acquisition",
                        "timeout",
                        "agroal",
                        "pool",
                        "sqlstate",
                        "error",
                        "warn",
                        "500",
                    )
                )
            ]
            app_errors = "\n".join(lines[-40:])
    except (Exception, SystemExit) as e:
        warn(f"todo logs: {e}")

    ds_config = ""
    try:
        cm = k.get(f"/api/v1/namespaces/{APP_NS}/configmaps/todo-config", allow_404=True)
        if cm:
            ds_config = (cm.get("data") or {}).get("application.properties", "")
    except (Exception, SystemExit):
        pass

    todo_restarts = 0
    try:
        pods = k.get(f"/api/v1/namespaces/{APP_NS}/pods?labelSelector=app=todo").get(
            "items"
        ) or []
        if pods:
            cs = (pods[0].get("status") or {}).get("containerStatuses") or []
            if cs:
                todo_restarts = int(cs[0].get("restartCount") or 0)
    except Exception:
        pass

    policy_evidence = _policy_snapshot(k)

    return {
        "app_namespace": APP_NS,
        "db_port": DB_PORT,
        "network_evidence": {
            "total_db_flows": total,
            "rtt_ms": {
                "min": rtt_min,
                "avg": rtt_avg,
                "p95": rtt_p95,
                "max": rtt_max,
            },
            "dropped_flows": drop_flows,
            "dropped_packets": drop_packets,
            "drop_causes": drop_causes,
            "top_talkers_to_db": [],
        },
        "application_evidence": {
            "todo_restart_count": todo_restarts,
            "error_log_excerpts": [ln for ln in app_errors.splitlines() if ln],
            "datasource_config": ds_config,
        },
        "policy_evidence": policy_evidence,
    }


def _policy_snapshot(k: heal.K8s) -> dict[str, Any]:
    pol = k.get(
        f"/apis/networking.k8s.io/v1/namespaces/{APP_NS}/networkpolicies/{POLICY_NAME}",
        allow_404=True,
    )
    if not pol:
        return {
            "db_policy_name": POLICY_NAME,
            "postgres_pod_selector": {},
            "ingress_from_pod_labels": [],
            "ingress_port_count": 0,
            "todo_workload_labels": {},
        }
    spec = pol.get("spec") or {}
    peers = []
    for rule in spec.get("ingress") or []:
        for fr in rule.get("from") or []:
            labels = (fr.get("podSelector") or {}).get("matchLabels") or {}
            if labels:
                peers.append(dict(labels))
    todo = k.get(f"/apis/apps/v1/namespaces/{APP_NS}/deployments/todo", allow_404=True)
    todo_labels = (
        ((todo or {}).get("spec") or {}).get("selector") or {}
    ).get("matchLabels") or {}
    ports = sum(len(r.get("ports") or []) for r in spec.get("ingress") or [])
    return {
        "db_policy_name": POLICY_NAME,
        "postgres_pod_selector": (spec.get("podSelector") or {}).get("matchLabels")
        or {},
        "ingress_from_pod_labels": peers,
        "ingress_port_count": ports,
        "todo_workload_labels": dict(todo_labels),
    }


def admin_k8s() -> heal.K8s:
    """Use mounted cluster-admin kubeconfig (not the limited in-cluster SA)."""
    saved_ic = os.environ.get("NETOBSERV_IN_CLUSTER")
    saved_kc = os.environ.get("KUBECONFIG")
    os.environ["NETOBSERV_IN_CLUSTER"] = "0"
    os.environ["KUBECONFIG"] = KUBECONFIG
    try:
        return heal.K8s()
    finally:
        if saved_ic is None:
            os.environ.pop("NETOBSERV_IN_CLUSTER", None)
        else:
            os.environ["NETOBSERV_IN_CLUSTER"] = saved_ic
        if saved_kc is None:
            os.environ.pop("KUBECONFIG", None)
        else:
            os.environ["KUBECONFIG"] = saved_kc


def capture_flows(
    duration: int = DEFAULT_DURATION,
    burst_load: bool = True,
    workdir: Path | None = None,
) -> dict[str, Any]:
    global NETOBSERV_BIN
    NETOBSERV_BIN = find_netobserv_bin()
    duration = max(30, min(int(duration), 120))
    k = heal.K8s()
    prev_replicas: int | None = None
    td = workdir or Path(tempfile.mkdtemp(prefix="netobserv-capture-"))
    td.mkdir(parents=True, exist_ok=True)
    step(f"Flow capture TCP:{DB_PORT} for {duration}s (RTT + packet drops)")
    try:
        if burst_load:
            prev_replicas = scale_load_burst(k, True)
        step("Preparing NetObserv CLI (cleanup stale capture namespace if present)")
        try:
            netobserv_cmd("cleanup", timeout=120)
        except SystemExit:
            warn("cleanup returned non-zero — continuing")
        netobserv_cmd(
            "flows",
            "--background",
            "--enable_rtt",
            "--enable_pkt_drop",
            "--action=Accept",
            "--cidr=0.0.0.0/0",
            f"--protocol=TCP",
            f"--port={DB_PORT}",
            cwd=td,
            timeout=600,
        )
        ok("Background capture started")
        step(f"Collecting flows for {duration}s")
        time.sleep(duration)
        netobserv_cmd("stop", cwd=td)
        netobserv_cmd("copy", cwd=td)
        netobserv_cmd("cleanup", cwd=td)
        db_files = list((td / "output").rglob("*.db")) if (td / "output").is_dir() else []
        if not db_files:
            die("No flow *.db under capture output/")
        db_path = db_files[0]
        ok(f"Flow database: {db_path}")
        step("Extracting evidence bundle")
        evidence = extract_evidence(db_path, k)
        evidence["capture_meta"] = {
            "duration_seconds": duration,
            "burst_load": burst_load,
            "workdir": str(td),
        }
        net = evidence.get("network_evidence") or {}
        total = net.get("total_db_flows", 0)
        rtt = (net.get("rtt_ms") or {}).get("avg")
        print(
            f"CAPTURE_SUMMARY: flows={total} avg_rtt_ms={rtt} "
            f"duration={duration}s burst={burst_load}"
        )
        return evidence
    finally:
        if burst_load:
            restore_load(k, prev_replicas)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] != "capture":
        print(
            "Usage: netobserv-cluster-capture.py capture [duration_seconds]",
            file=sys.stderr,
        )
        return 1
    dur = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DURATION
    burst = os.environ.get("CAPTURE_BURST", "1") not in ("0", "false", "no")
    evidence = capture_flows(duration=dur, burst_load=burst)
    out = Path(os.environ.get("CAPTURE_OUTPUT", "/data/latest-evidence.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2))
    ok(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
