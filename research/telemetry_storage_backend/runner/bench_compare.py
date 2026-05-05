#!/usr/bin/env python3
"""
Compare Doris vs ClickHouse vs Druid: run same queries on all backends, produce combined report.

Usage: python3 runner/bench_compare.py --all --data-dir telemetry_data --out out
"""
from __future__ import annotations
import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from datetime import datetime
import requests

ROOT = Path(__file__).resolve().parents[1]
DORIS_SCHEMA = ROOT / "schemas" / "doris.sql"
CH_SCHEMA = ROOT / "schemas" / "clickhouse.sql"
DORIS_QDIR = ROOT / "queries" / "doris"
CH_QDIR = ROOT / "queries" / "clickhouse"
DRUID_QDIR = ROOT / "queries" / "druid"
OB_QDIR = ROOT / "queries" / "oceanbase"
VM_QDIR = ROOT / "queries" / "victoriametrics"
OB_SCHEMA = ROOT / "schemas" / "oceanbase.sql"

DORIS_FE_HTTP = os.getenv("DORIS_FE_HTTP", "http://localhost:8030")
DORIS_MYSQL_PORT = int(os.getenv("DORIS_MYSQL_PORT", "9030"))
DORIS_PASS = os.getenv("DORIS_PASS", "")
CH_HTTP = os.getenv("CLICKHOUSE_HTTP", "http://localhost:8123")
CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
DRUID_HTTP = os.getenv("DRUID_HTTP", "http://localhost:8888")
OB_HOST = os.getenv("OCEANBASE_HOST", "127.0.0.1")
OB_PORT = int(os.getenv("OCEANBASE_PORT", "2881"))
OB_CONTAINER = os.getenv("OCEANBASE_CONTAINER", "tsb-oceanbase")
LOKI_HTTP = os.getenv("LOKI_HTTP", "http://localhost:3100")
VM_HTTP = os.getenv("VM_HTTP", "http://localhost:8428")
VL_HTTP = os.getenv("VL_HTTP", "http://localhost:9428")
VT_HTTP = os.getenv("VT_HTTP", "http://localhost:10428")
DB = "telemetry"

def _port_of(url: str) -> int:
    from urllib.parse import urlparse
    p = urlparse(url)
    return p.port or (443 if p.scheme == "https" else 80)

def _ch_params(extra: dict | None = None) -> dict:
    params = dict(extra) if extra else {}
    if CH_PASSWORD:
        params["user"] = "default"
        params["password"] = CH_PASSWORD
    return params

def wait_port(host: str, port: int, timeout_s: int = 120) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except Exception:
            time.sleep(1)
    return False

def wait_doris_be_ready(timeout_s: int = 300) -> bool:
    """Wait until Doris has at least one online backend."""
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8030/api/health", timeout=5) as r:
                data = json.loads(r.read().decode())
                online = data.get("data", {}).get("online_backend_num", 0)
                if online >= 1:
                    print(f"[wait] Doris BE ready (online_backend_num={online})")
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False

def _run_doris_sql(sql: str, db: str = "information_schema") -> None:
    pass_arg = f"-p{DORIS_PASS}" if DORIS_PASS else ""
    cmd = [
        "docker", "run", "--rm", "--network", "tsb-net",
        "mysql:8", "sh", "-lc",
        f"echo '{sql}' | mysql -h tsb-doris -P 9030 -uroot {pass_arg} -D {db}"
    ]
    subprocess.run(cmd, check=True)

def _run_oceanbase_sql(sql: str, db: str | None = DB) -> None:
    cmd = ["docker", "run", "--rm", "-i", "--network", "tsb-net",
           "mysql:8", "mysql", "-h", OB_CONTAINER, "-P", "2881", "-uroot"]
    if db:
        cmd.extend(["-D", db])
    subprocess.run(cmd, input=sql.encode(), check=True)

def apply_doris_schema() -> None:
    sql = DORIS_SCHEMA.read_text()
    _run_doris_sql(sql)
    print("[schema] Doris applied")

def truncate_doris_tables() -> None:
    for t in ("logs", "spans", "metrics"):
        _run_doris_sql(f"TRUNCATE TABLE {DB}.{t};", db=DB)
    print("[truncate] Doris tables cleared")

def apply_clickhouse_schema() -> None:
    sql = CH_SCHEMA.read_text()
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if not stmt or stmt.startswith("--"):
            continue
        for attempt in range(5):
            try:
                r = requests.post(CH_HTTP, params=_ch_params(), data=stmt + ";", timeout=30)
                if r.status_code != 200:
                    print(f"[schema] ClickHouse: {r.status_code} {r.text[:200]}")
                break
            except requests.exceptions.ConnectionError as e:
                if attempt < 4:
                    print(f"[schema] ClickHouse connection error, retry {attempt + 1}/5 in 5s...")
                    time.sleep(5)
                else:
                    raise
    print("[schema] ClickHouse applied")

def truncate_clickhouse_tables() -> None:
    for t in ("logs", "spans", "metrics"):
        r = requests.post(CH_HTTP, params=_ch_params(), data=f"TRUNCATE TABLE {DB}.{t};", timeout=30)
        if r.status_code != 200:
            print(f"[truncate] ClickHouse {t}: {r.status_code} {r.text[:200]}")
    print("[truncate] ClickHouse tables cleared")

def wait_oceanbase_ready(timeout_s: int = 300) -> bool:
    """Wait until OceanBase accepts queries (bootstrap can take 3-5 min)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            cmd = ["docker", "run", "--rm", "-i", "--network", "tsb-net",
                   "mysql:8", "mysql", "-h", OB_CONTAINER, "-P", "2881", "-uroot", "-e", "SELECT 1"]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if out.returncode == 0:
                print("[wait] OceanBase ready")
                return True
        except Exception:
            pass
        time.sleep(10)
    return False

def apply_oceanbase_schema() -> None:
    if not wait_oceanbase_ready(300):
        raise RuntimeError("OceanBase not ready (bootstrap ~3-5 min)")
    sql = OB_SCHEMA.read_text()
    for attempt in range(6):
        try:
            cmd = ["docker", "run", "--rm", "-i", "--network", "tsb-net",
                   "mysql:8", "mysql", "-h", OB_CONTAINER, "-P", "2881", "-uroot"]
            out = subprocess.run(cmd, input=sql.encode(), capture_output=True, timeout=60)
            if out.returncode == 0:
                print("[schema] OceanBase applied")
                return
            err = (out.stderr or out.stdout or b"").decode(errors="replace")
            if "log stream is not leader" in err and attempt < 5:
                time.sleep(20)
                continue
            raise RuntimeError(f"OceanBase schema failed: {err[:500]}")
        except subprocess.TimeoutExpired:
            if attempt < 5:
                time.sleep(10)
                continue
            raise
    raise RuntimeError("OceanBase schema failed after retries")

def truncate_oceanbase_tables() -> None:
    for t in ("logs", "spans", "metrics"):
        try:
            _run_oceanbase_sql(f"TRUNCATE TABLE {DB}.{t};")
        except Exception as e:
            print(f"[truncate] OceanBase {t}: {e}")
    print("[truncate] OceanBase tables cleared")

def reset_vm_storage() -> None:
    """Stop VM containers, remove volumes, restart with clean storage."""
    compose = ["docker", "compose", "-f", str(ROOT / "docker-compose.yml"),
               "-f", str(ROOT / "docker-compose.victoriametrics.yml")]
    subprocess.run(compose + ["stop", "victoriametrics", "victorialogs", "victoriatraces"],
                   capture_output=True, check=False)
    subprocess.run(compose + ["rm", "-f", "victoriametrics", "victorialogs", "victoriatraces"],
                   capture_output=True, check=False)
    for vol_suffix in ["vmdata", "vldata", "vtdata"]:
        out = subprocess.run(["docker", "volume", "ls", "-q", "--filter", f"name={vol_suffix}"],
                             capture_output=True, text=True)
        for vol in out.stdout.strip().splitlines():
            subprocess.run(["docker", "volume", "rm", "-f", vol], capture_output=True, check=False)
    subprocess.run(compose + ["up", "-d", "victoriametrics", "victorialogs", "victoriatraces"], check=True)
    print("[truncate] VM storage reset, containers restarted")

def wait_vm_healthy(timeout_s: int = 120) -> bool:
    """Wait until all three VM services are healthy after restart."""
    endpoints = [
        (VM_HTTP, "/-/healthy", "VictoriaMetrics"),
        (VL_HTTP, "/health", "VictoriaLogs"),
        (VT_HTTP, "/health", "VictoriaTraces"),
    ]
    for base, path, name in endpoints:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                r = requests.get(f"{base}{path}", timeout=5)
                if r.status_code == 200:
                    print(f"[wait] {name} healthy")
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            print(f"[wait] {name} not healthy after {timeout_s}s")
            return False
    return True

def run_doris_query(sql: str) -> dict:
    pass_arg = f"-p{DORIS_PASS}" if DORIS_PASS else ""
    sh_script = (
        "cat > /tmp/bench.sql <<'EOSQL'\n" + sql + "\nEOSQL\n"
        f"mysql -h tsb-doris -P 9030 -uroot {pass_arg} -D {DB} -N -B < /tmp/bench.sql\n"
    )
    cmd = ["docker", "run", "--rm", "--network", "tsb-net", "mysql:8", "sh", "-lc", sh_script]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if out.returncode != 0:
        return {"error": (out.stderr or out.stdout or "").strip()[:500]}
    rows = [ln for ln in (out.stdout or "").splitlines() if ln.strip()]
    return {"latency_s": dt, "rows": len(rows)}

def run_clickhouse_query(sql: str) -> dict:
    t0 = time.time()
    r = requests.post(CH_HTTP, params=_ch_params({"query": sql}), timeout=120)
    dt = time.time() - t0
    if r.status_code != 200:
        return {"error": r.text[:500]}
    rows = [ln for ln in r.text.strip().splitlines() if ln.strip()]
    return {"latency_s": dt, "rows": len(rows)}

def run_druid_query(sql: str) -> dict:
    t0 = time.time()
    r = requests.post(
        f"{DRUID_HTTP}/druid/v2/sql",
        json={"query": sql, "resultFormat": "array"},
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    dt = time.time() - t0
    if r.status_code != 200:
        return {"error": r.text[:500]}
    try:
        arr = r.json()
        rows = len(arr) if isinstance(arr, list) else 0
        return {"latency_s": dt, "rows": rows}
    except Exception:
        return {"latency_s": dt, "rows": 0, "error": r.text[:200]}

def run_oceanbase_query(sql: str) -> dict:
    cmd = ["docker", "run", "--rm", "-i", "--network", "tsb-net",
           "mysql:8", "mysql", "-h", OB_CONTAINER, "-P", "2881", "-uroot", "-N", "-B", "-D", DB]
    t0 = time.time()
    out = subprocess.run(cmd, input=sql, capture_output=True, text=True, timeout=120)
    dt = time.time() - t0
    if out.returncode != 0:
        return {"error": (out.stderr or out.stdout or "").strip()[:500]}
    rows = [ln for ln in (out.stdout or "").splitlines() if ln.strip()]
    return {"latency_s": dt, "rows": len(rows)}


# Loki LogQL queries (logs only) - filter by run_id for isolation
LOKI_LOG_QUERIES = {
    "logs_errors_by_service": (
        'sum by (service) (count_over_time({{job="telemetry", run_id="{run_id}"}} |~ "(?i)error" [24h]))'
    ),
    "logs_recent": (
        '{{job="telemetry", run_id="{run_id}"}}'
    ),
    "logs_search_error": (
        '{{job="telemetry", run_id="{run_id}"}} |= "error"'
    ),
}


def run_loki_log_query(name: str, logql: str, run_id: str, expect_count: bool = False) -> dict:
    """Run a LogQL query against Loki. For range queries returns row count from values."""
    url = f"{LOKI_HTTP}/loki/api/v1/query_range"
    query = logql.format(run_id=run_id)
    now_ns = int(time.time() * 1_000_000_000)
    start_ns = now_ns - (30 * 24 * 3600 * 1_000_000_000)  # 30 days ago
    # Limit 5 to stay under 4MB gRPC default (log lines can be ~200KB each)
    params = {"query": query, "start": str(start_ns), "end": str(now_ns), "limit": 5}
    t0 = time.time()
    try:
        r = requests.get(url, params=params, timeout=120)
        dt = time.time() - t0
        if r.status_code != 200:
            return {"latency_s": dt, "rows": 0, "error": r.text[:300]}
        data = r.json()
        results = data.get("data", {}).get("result", [])
        if expect_count and "sum by" in query:
            # Instant/metrics-style: each stream has one value [ts, count]
            rows = sum(1 for s in results for _ in s.get("values", []))
        else:
            rows = sum(len(s.get("values", [])) for s in results)
        return {"latency_s": dt, "rows": rows}
    except Exception as e:
        return {"latency_s": 0, "rows": 0, "error": str(e)[:200]}


def bench_loki_logs(run_id: str) -> dict:
    """Run the 3 log queries against Loki. Returns partial results (logs only)."""
    res = {}
    res["logs_errors_by_service"] = run_loki_log_query(
        "logs_errors_by_service", LOKI_LOG_QUERIES["logs_errors_by_service"], run_id, expect_count=True
    )
    res["logs_recent"] = run_loki_log_query(
        "logs_recent", LOKI_LOG_QUERIES["logs_recent"], run_id
    )
    res["logs_search_error"] = run_loki_log_query(
        "logs_search_error", LOKI_LOG_QUERIES["logs_search_error"], run_id
    )
    return res


def run_vm_logql(query: str, run_id: str = "") -> dict:
    """Run a LogsQL query against VictoriaLogs native API."""
    url = f"{VL_HTTP}/select/logsql/query"
    if run_id:
        if " | " in query:
            filt, pipes = query.split(" | ", 1)
            query = f"run_id:{run_id} {filt} | {pipes}"
        else:
            query = f"run_id:{run_id} {query}"
    params = {"query": query, "limit": 1000}
    t0 = time.time()
    try:
        r = requests.get(url, params=params, timeout=120)
        dt = time.time() - t0
        if r.status_code != 200:
            return {"latency_s": dt, "rows": 0, "error": r.text[:300]}
        rows = len([ln for ln in r.text.strip().splitlines() if ln.strip()])
        return {"latency_s": dt, "rows": rows}
    except Exception as e:
        return {"latency_s": 0, "rows": 0, "error": str(e)[:200]}


def run_vm_metricsql(query: str) -> dict:
    """Run a MetricsQL query against VictoriaMetrics."""
    url = f"{VM_HTTP}/api/v1/query"
    params = {"query": query, "step": "24h"}
    t0 = time.time()
    try:
        r = requests.get(url, params=params, timeout=120)
        dt = time.time() - t0
        if r.status_code != 200:
            return {"latency_s": dt, "rows": 0, "error": r.text[:300]}
        data = r.json()
        results = data.get("data", {}).get("result", [])
        return {"latency_s": dt, "rows": len(results)}
    except Exception as e:
        return {"latency_s": 0, "rows": 0, "error": str(e)[:200]}


def run_vm_traceql(query: str) -> dict:
    """Run a LogsQL query against VictoriaTraces (uses same query language as VictoriaLogs)."""
    url = f"{VT_HTTP}/select/logsql/query"
    params = {"query": query, "limit": 1000}
    t0 = time.time()
    try:
        r = requests.get(url, params=params, timeout=120)
        dt = time.time() - t0
        if r.status_code != 200:
            return {"latency_s": dt, "rows": 0, "error": r.text[:300]}
        rows = len([ln for ln in r.text.strip().splitlines() if ln.strip()])
        return {"latency_s": dt, "rows": rows}
    except Exception as e:
        return {"latency_s": 0, "rows": 0, "error": str(e)[:200]}


def bench_vm(run_id: str = "") -> tuple[dict, dict, dict]:
    """Run all VM queries. Returns (victorialogs_results, victoriametrics_results, victoriatraces_results)."""
    vl, vm, vt = {}, {}, {}
    for f in sorted(VM_QDIR.glob("*")):
        if f.suffix not in (".logql", ".metricsql", ".traceql"):
            continue
        name = f.stem
        query = f.read_text().strip()
        try:
            if f.suffix == ".logql":
                vl[name] = run_vm_logql(query, run_id)
            elif f.suffix == ".metricsql":
                vm[name] = run_vm_metricsql(query)
            elif f.suffix == ".traceql":
                vt[name] = run_vm_traceql(query)
        except Exception as e:
            target = vl if f.suffix == ".logql" else vm if f.suffix == ".metricsql" else vt
            target[name] = {"error": str(e)[:200]}
    return vl, vm, vt


def bench_backend(qdir: Path, run_fn) -> dict:
    results = {}
    for f in sorted(qdir.glob("*.sql")):
        name = f.stem
        sql = f.read_text()
        try:
            m = run_fn(sql)
        except Exception as e:
            m = {"error": str(e)}
        results[name] = m
    return results


def get_data_volume(use_doris: bool, use_oceanbase: bool = True, use_vm: bool = False, use_sql: bool = True) -> tuple:
    """Run full-scan COUNT on each backend; return (doris_vol, ch_vol, druid_vol, ob_vol, vl_vol, vm_vol, vt_vol)."""
    doris_vol = {}
    ch_vol = {}
    druid_vol = {}
    ob_vol = {}
    vl_vol = {}
    vm_vol = {}
    vt_vol = {}
    def _parse_tsv(lines: list[str]) -> dict:
        counts = {}
        for ln in lines:
            parts = ln.strip().split("\t")
            if len(parts) >= 2:
                tbl, cnt = parts[0].strip().lower(), parts[1].strip()
                counts[tbl] = int(cnt) if cnt.isdigit() else 0
        counts["total"] = counts.get("logs", 0) + counts.get("spans", 0) + counts.get("metrics", 0)
        return counts

    if use_doris:
        try:
            sql_doris = (DORIS_QDIR / "data_volume.sql").read_text()
            pass_arg = f"-p{DORIS_PASS}" if DORIS_PASS else ""
            sh_script = (
                "cat > /tmp/dv.sql <<'EOSQL'\n" + sql_doris + "\nEOSQL\n"
                f"mysql -h tsb-doris -P 9030 -uroot {pass_arg} -D {DB} -N -B < /tmp/dv.sql\n"
            )
            t0 = time.time()
            out = subprocess.run(
                ["docker", "run", "--rm", "--network", "tsb-net", "mysql:8", "sh", "-lc", sh_script],
                capture_output=True, text=True, timeout=120,
            )
            dt = time.time() - t0
            if out.returncode == 0:
                lines = [ln for ln in (out.stdout or "").splitlines() if ln.strip()]
                doris_vol = _parse_tsv(lines)
                doris_vol["latency_s"] = dt
        except Exception as e:
            doris_vol = {"error": str(e)[:200]}

    if use_sql:
        sql_ch = (CH_QDIR / "data_volume.sql").read_text()
        sql_druid = (DRUID_QDIR / "data_volume.sql").read_text()
        try:
            t0 = time.time()
            r = requests.post(CH_HTTP, params=_ch_params({"query": sql_ch}), timeout=120)
            dt = time.time() - t0
            if r.status_code == 200:
                lines = [ln for ln in r.text.strip().splitlines() if ln.strip()]
                ch_vol = _parse_tsv(lines)
                ch_vol["latency_s"] = dt
            else:
                ch_vol = {"error": r.text[:200]}
        except Exception as e:
            ch_vol = {"error": str(e)[:200]}

        try:
            t0 = time.time()
            r = requests.post(
                f"{DRUID_HTTP}/druid/v2/sql",
                json={"query": sql_druid, "resultFormat": "array"},
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            dt = time.time() - t0
            if r.status_code == 200:
                arr = r.json()
                counts = {}
                for row in (arr or []):
                    if len(row) >= 2:
                        tbl = str(row[0]).lower()
                        cnt = int(row[1]) if isinstance(row[1], (int, float)) else 0
                        counts[tbl] = cnt
                counts["total"] = counts.get("logs", 0) + counts.get("spans", 0) + counts.get("metrics", 0)
                counts["latency_s"] = dt
                druid_vol = counts
            else:
                druid_vol = {"error": r.text[:200]}
        except Exception as e:
            druid_vol = {"error": str(e)[:200]}

    if use_oceanbase:
        sql_ob = (OB_QDIR / "data_volume.sql").read_text()
        try:
            t0 = time.time()
            out = subprocess.run(
                ["docker", "run", "--rm", "-i", "--network", "tsb-net",
                 "mysql:8", "mysql", "-h", OB_CONTAINER, "-P", "2881", "-uroot", "-N", "-B", "-D", DB],
                input=sql_ob, capture_output=True, text=True, timeout=120,
            )
            dt = time.time() - t0
            if out.returncode == 0:
                lines = [ln for ln in (out.stdout or "").splitlines() if ln.strip()]
                ob_vol = _parse_tsv(lines)
                ob_vol["latency_s"] = dt
            else:
                ob_vol = {"error": (out.stderr or out.stdout or "")[:200]}
        except Exception as e:
            ob_vol = {"error": str(e)[:200]}

    if use_vm:
        # VictoriaLogs: count log entries via stats
        try:
            t0 = time.time()
            r = requests.get(f"{VL_HTTP}/select/logsql/query",
                             params={"query": "* | stats count() as total", "limit": 1},
                             timeout=30)
            dt = time.time() - t0
            if r.status_code == 200:
                lines = [ln for ln in r.text.strip().splitlines() if ln.strip()]
                if lines:
                    row = json.loads(lines[0])
                    vl_vol = {"rows": int(row.get("total", 0)), "latency_s": dt}
                else:
                    vl_vol = {"rows": 0, "latency_s": dt}
        except Exception as e:
            vl_vol = {"error": str(e)[:200]}
        # VictoriaMetrics: count total inserted rows via /metrics internal counter
        try:
            t0 = time.time()
            r = requests.get(f"{VM_HTTP}/metrics", timeout=30)
            dt = time.time() - t0
            if r.status_code == 200:
                total_rows = 0
                for line in r.text.splitlines():
                    if line.startswith("vm_rows_inserted_total{"):
                        parts = line.rsplit(" ", 1)
                        if len(parts) == 2:
                            total_rows += int(float(parts[1]))
                vm_vol = {"rows": total_rows, "latency_s": dt}
        except Exception as e:
            vm_vol = {"error": str(e)[:200]}
        # VictoriaTraces: count spans via LogsQL stats
        try:
            t0 = time.time()
            r = requests.get(f"{VT_HTTP}/select/logsql/query",
                             params={"query": "* | stats count() as total", "limit": 1},
                             timeout=30)
            dt = time.time() - t0
            if r.status_code == 200:
                lines = [ln for ln in r.text.strip().splitlines() if ln.strip()]
                if lines:
                    row = json.loads(lines[0])
                    vt_vol = {"rows": int(row.get("total", 0)), "latency_s": dt}
                else:
                    vt_vol = {"rows": 0, "latency_s": dt}
        except Exception as e:
            vt_vol = {"error": str(e)[:200]}

    return doris_vol, ch_vol, druid_vol, ob_vol, vl_vol, vm_vol, vt_vol

def write_combined_report(out_dir: Path, doris_ingest: dict, ch_ingest: dict, druid_ingest: dict,
                         doris_qres: dict, ch_qres: dict, druid_qres: dict,
                         ob_ingest: dict | None = None, ob_qres: dict | None = None,
                         loki_ingest: dict | None = None, loki_qres: dict | None = None,
                         vl_ingest: dict | None = None, vl_qres: dict | None = None,
                         vm_ingest: dict | None = None, vm_qres: dict | None = None,
                         vt_ingest: dict | None = None, vt_qres: dict | None = None,
                         otlp_ingest: dict | None = None,
                         data_vol: tuple | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    backends = [("doris", doris_qres), ("clickhouse", ch_qres), ("druid", druid_qres)]
    if ob_qres:
        backends.append(("oceanbase", ob_qres))
    if loki_qres:
        backends.append(("loki", loki_qres))
    if vl_qres:
        backends.append(("victorialogs", vl_qres))
    if vm_qres:
        backends.append(("victoriametrics", vm_qres))
    if vt_qres:
        backends.append(("victoriatraces", vt_qres))
    all_queries = sorted(set().union(*(qr.keys() for _, qr in backends)))
    _display = {"doris": "Doris", "clickhouse": "ClickHouse", "druid": "Druid",
                 "oceanbase": "OceanBase", "loki": "Loki",
                 "victorialogs": "VictoriaLogs", "victoriametrics": "VictoriaMetrics",
                 "victoriatraces": "VictoriaTraces"}
    rows = []
    chart_data = {"queries": []}
    for bname, _ in backends:
        chart_data[bname] = []
    for q in all_queries:
        lats = []
        cells = ""
        for bname, bqres in backends:
            bq = bqres.get(q, {})
            lat = bq.get("latency_s", "")
            cells += f"<td>{lat}</td><td>{bq.get('rows', '')}</td><td>{bq.get('error', '')}</td>"
            lats.append((lat, _display.get(bname, bname)))
            chart_data[bname].append(round(lat, 4) if isinstance(lat, (int, float)) else None)
        valid = [(x, n) for x, n in lats if isinstance(x, (int, float))]
        winner, pct_diff = "", ""
        if len(valid) >= 2:
            fastest = min(valid, key=lambda t: t[0])
            slowest = max(valid, key=lambda t: t[0])
            winner = fastest[1]
            if winner in ("VictoriaLogs", "VictoriaTraces"):
                winner = "VictoriaMetrics"
            if slowest[0] >= 1e-9:
                pct = (slowest[0] - fastest[0]) / slowest[0] * 100
                pct_diff = f"{pct:.1f}%"
        rows.append(f"<tr><td>{q}</td>{cells}<td>{pct_diff}</td><td>{winner}</td></tr>")
        chart_data["queries"].append(q)
    all_ingests = [("Doris", doris_ingest), ("ClickHouse", ch_ingest), ("Druid", druid_ingest)]
    if ob_ingest and ob_ingest.get("status") == "ok":
        all_ingests.append(("OceanBase", ob_ingest))
    if loki_ingest and loki_ingest.get("status") == "ok":
        all_ingests.append(("Loki", loki_ingest))
    if vl_ingest and vl_ingest.get("status") == "ok":
        all_ingests.append(("VictoriaLogs", vl_ingest))
    if vm_ingest and vm_ingest.get("status") == "ok":
        all_ingests.append(("VictoriaMetrics", vm_ingest))
    if vt_ingest and vt_ingest.get("status") == "ok":
        all_ingests.append(("VictoriaTraces", vt_ingest))
    ingest_labels = [n for n, _ in all_ingests]
    ingest_dur = [i.get("duration_s") for _, i in all_ingests]
    ingest_rps = [i.get("rows_per_sec") for _, i in all_ingests]
    ingest_chart = {"labels": ingest_labels, "duration_s": ingest_dur, "rows_per_sec": ingest_rps}
    data_vol_row = ""
    if data_vol:
        d_vol, c_vol, dr_vol = data_vol[0], data_vol[1], data_vol[2]
        ob_vol_d = data_vol[3] if len(data_vol) > 3 else {}
        vl_vol_d = data_vol[4] if len(data_vol) > 4 else {}
        vm_vol_d = data_vol[5] if len(data_vol) > 5 else {}
        vt_vol_d = data_vol[6] if len(data_vol) > 6 else {}
        def _fmt(v: dict) -> str:
            if not v or "error" in v:
                return v.get("error", "-") if v else "-"
            if "total" in v:
                total = v["total"]
                return f"{total:,} (logs={v.get('logs',0):,}, spans={v.get('spans',0):,}, metrics={v.get('metrics',0):,})"
            return f"{v.get('rows', 0):,}"
        def _lat(v: dict) -> str:
            lat = v.get("latency_s") if v else None
            return f"{lat:.3f}s" if isinstance(lat, (int, float)) else "-"
        vol_rows = [
            ("Doris", d_vol), ("ClickHouse", c_vol), ("Druid", dr_vol),
        ]
        if ob_vol_d:
            vol_rows.append(("OceanBase", ob_vol_d))
        if vl_vol_d:
            vol_rows.append(("VictoriaLogs", vl_vol_d))
        if vm_vol_d:
            vol_rows.append(("VictoriaMetrics", vm_vol_d))
        if vt_vol_d:
            vol_rows.append(("VictoriaTraces", vt_vol_d))
        vol_html = "\n".join(f"    <tr><td>{name}</td><td>{_fmt(v)}</td><td>{_lat(v)}</td></tr>" for name, v in vol_rows)
        data_vol_row = f"""
  <h3>Data volume (full scan)</h3>
  <p>Total rows in telemetry tables at query time. Latency = full COUNT(*) time.</p>
  <table>
    <tr><th>Backend</th><th>Total rows</th><th>Full-scan latency</th></tr>
{vol_html}
  </table>
"""

    chart_json = json.dumps(chart_data)
    ingest_json = json.dumps(ingest_chart)
    report_title = " vs ".join(_display.get(n, n) for n, _ in backends) + " benchmark"
    ingest_rows_html = "\n".join(
        f'    <tr><td>{name}</td><td>{ing.get("mechanism", "-")}</td><td>{ing.get("duration_s", "-")}</td>'
        f'<td>{ing.get("rows", "-")}</td><td>{ing.get("rows_per_sec", "-")}</td></tr>'
        for name, ing in all_ingests)
    if otlp_ingest:
        mech = otlp_ingest.get("mechanism", "OTLP")
        dur = otlp_ingest.get("duration_s", "-")
        otlp_count = otlp_ingest.get("count", 1000)
        vm_mech = {
            "victorialogs": f"OTLP ({otlp_count} logs)",
            "victoriametrics": f"OTLP ({otlp_count} metrics)",
            "victoriatraces": f"OTLP ({otlp_count} spans)",
        }
        for backend, key in [("Doris", "doris"), ("ClickHouse", "clickhouse"),
                               ("VictoriaLogs", "victorialogs"), ("VictoriaMetrics", "victoriametrics"),
                               ("VictoriaTraces", "victoriatraces")]:
            d = otlp_ingest.get(key, {})
            if d and d.get("rows", 0) > 0:
                row_mech = vm_mech.get(key, mech)
                ingest_rows_html += f'\n    <tr><td>{backend}</td><td>{row_mech}</td><td>{dur}</td><td>{d.get("rows", "-")}</td><td>{d.get("rows_per_sec", "-")}</td></tr>'
    query_th = "".join(f'<th>{_display.get(n,n)} lat</th><th>{_display.get(n,n)} rows</th><th>{_display.get(n,n)} err</th>' for n, _ in backends)
    colors = ['rgba(54,162,235,0.7)', 'rgba(75,192,192,0.7)', 'rgba(255,159,64,0.7)',
              'rgba(153,102,255,0.7)', 'rgba(255,99,132,0.7)',
              'rgba(0,200,83,0.7)', 'rgba(255,206,86,0.7)', 'rgba(231,76,60,0.7)']
    chart_datasets = ", ".join(
        f'{{ label: "{_display.get(n,n)}", data: data["{n}"], backgroundColor: "{colors[i % len(colors)]}" }}'
        for i, (n, _) in enumerate(backends))
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{report_title}</title>
<style>table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 10px}}#chartWrap{{max-width:900px;height:400px;margin:1em 0}}</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head><body>
  <h1>{report_title}</h1>
  <div>Generated at {datetime.now().isoformat()}</div>
  <h2>Ingestion comparison</h2>
  <div id="ingestChartWrap" style="max-width:600px;height:250px;margin:1em 0"><canvas id="ingestChart"></canvas></div>
  <table>
    <tr><th>Backend</th><th>Mechanism</th><th>Duration (s)</th><th>Rows</th><th>Rows/sec</th></tr>
{ingest_rows_html}
  </table>
  {data_vol_row}
  <h2>Query latency (seconds)</h2>
  <div id="chartWrap"><canvas id="latencyChart"></canvas></div>
  <h2>Query comparison</h2>
  <table>
    <tr><th>query</th>{query_th}<th>% diff</th><th>faster</th></tr>
    {chr(10).join(rows)}
  </table>
  <script>
    const data = {chart_json};
    const ingestData = {ingest_json};
    if (ingestData.duration_s.some(x => x != null)) {{
      const ctx = document.getElementById('ingestChart').getContext('2d');
      new Chart(ctx, {{
        type: 'bar',
        data: {{
          labels: ingestData.labels,
          datasets: [{{ label: 'Ingestion time (s)', data: ingestData.duration_s }}]
        }},
        options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true }} }} }}
      }});
    }}
    const ctx = document.getElementById('latencyChart').getContext('2d');
    const datasets = [{chart_datasets}];
    new Chart(ctx, {{
      type: 'bar',
      data: {{ labels: data.queries, datasets: datasets }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ position: 'top' }} }},
        scales: {{
          x: {{ ticks: {{ maxRotation: 45, minRotation: 45 }} }},
          y: {{ beginAtZero: true, title: {{ display: true, text: 'Latency (s)' }} }}
        }}
      }}
    }});
  </script>
</body></html>"""
    (out_dir / "compare.html").write_text(html)
    (out_dir / "doris_queries.json").write_text(json.dumps(doris_qres, indent=2))
    (out_dir / "clickhouse_queries.json").write_text(json.dumps(ch_qres, indent=2))
    (out_dir / "druid_queries.json").write_text(json.dumps(druid_qres, indent=2))
    if ob_qres:
        (out_dir / "oceanbase_queries.json").write_text(json.dumps(ob_qres, indent=2))
    if loki_qres:
        (out_dir / "loki_queries.json").write_text(json.dumps(loki_qres, indent=2))
    if vl_qres:
        (out_dir / "victorialogs_queries.json").write_text(json.dumps(vl_qres, indent=2))
    if vm_qres:
        (out_dir / "victoriametrics_queries.json").write_text(json.dumps(vm_qres, indent=2))
    if vt_qres:
        (out_dir / "victoriatraces_queries.json").write_text(json.dumps(vt_qres, indent=2))
    print(f"[bench] wrote {out_dir}/compare.html")

def _detect_backends(run_dir: Path) -> str:
    """Detect which backends participated in a run from *_queries.json files."""
    name_map = {"doris": "Doris", "clickhouse": "ClickHouse", "druid": "Druid",
                "oceanbase": "OceanBase", "loki": "Loki",
                "victorialogs": "VictoriaLogs", "victoriametrics": "VictoriaMetrics",
                "victoriatraces": "VictoriaTraces"}
    found = []
    for key, label in name_map.items():
        if (run_dir / f"{key}_queries.json").exists():
            found.append(label)
    return " vs ".join(found) if found else "unknown"

def write_rolling_report(out_base: Path) -> None:
    """Write rolling_index.html listing all benchmark runs."""
    import re
    entries = []
    for d in out_base.glob("storage_bench_doris_*"):
        if (d / "summary.html").exists():
            ts = d.name.replace("storage_bench_doris_", "").replace("_", " ", 1)
            entries.append((d.name, "summary.html", "Doris", ts))
    for d in out_base.glob("storage_bench_compare_*"):
        if (d / "compare.html").exists():
            ts = d.name.replace("storage_bench_compare_", "").replace("_", " ", 1)
            run_type = _detect_backends(d)
            entries.append((d.name, "compare.html", run_type, ts))
    def sort_key(e):
        m = re.search(r"storage_bench_(?:doris|compare)_(\d{8})_?(\d{6})?", e[0])
        return (m.group(1), m.group(2) or "") if m else ("", "")
    entries.sort(key=sort_key, reverse=True)
    rows = [f'    <tr><td><a href="{n}/{f}">{n}</a></td><td>{t}</td><td>{ts}</td></tr>' for n, f, t, ts in entries]
    out_base.mkdir(parents=True, exist_ok=True)
    index_html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Telemetry storage benchmark - runs</title>
<style>table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:8px}}a{{text-decoration:none}}</style>
</head><body>
  <h1>Telemetry storage benchmark</h1>
  <p>Latest runs (newest first):</p>
  <table>
    <tr><th>run</th><th>type</th><th>date_time</th></tr>
{chr(10).join(rows) if rows else '    <tr><td colspan="3">No runs yet</td></tr>'}
  </table>
</body></html>"""
    (out_base / "rolling_index.html").write_text(index_html)
    print(f"[bench] wrote {out_base}/rolling_index.html")

def wait_druid_ready(timeout_s: int = 300) -> bool:
    """Wait until Druid is ready."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{DRUID_HTTP}/status/health", timeout=5)
            if r.status_code == 200:
                print("[wait] Druid ready")
                return True
        except Exception:
            pass
        time.sleep(5)
    return False

def main() -> int:
    ap = argparse.ArgumentParser(description="Compare Doris vs ClickHouse vs Druid")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--clickhouse-only", action="store_true", help="Skip Doris (e.g. when BE not ready)")
    ap.add_argument("--vm-only", action="store_true", help="Run only VictoriaMetrics/Logs/Traces (skip SQL backends)")
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--batch", type=int, default=5000, help="Rows per load batch")
    ap.add_argument("--scale-to", type=int, default=None, help="Target row count per type (50k = 50000)")
    ap.add_argument("--streaming-batch", type=int, default=None,
                    help="Smaller batch size to simulate real-time ingestion (e.g. 500 = 100 batches of 500 rows)")
    ap.add_argument("--otlp", action="store_true", help="Also run OTLP ingestion via telemetrygen")
    ap.add_argument("--otlp-count", type=int, default=1000, help="Spans, logs, metrics each for OTLP (default 1000)")
    args = ap.parse_args()

    vm_only = getattr(args, "vm_only", False)
    use_doris = not args.clickhouse_only and not vm_only
    use_sql_backends = not vm_only
    if use_sql_backends:
        if use_doris:
            assert wait_port("127.0.0.1", _port_of(DORIS_FE_HTTP), 180), f"Doris FE {_port_of(DORIS_FE_HTTP)} not ready"
            assert wait_port("127.0.0.1", DORIS_MYSQL_PORT, 180), f"Doris MySQL {DORIS_MYSQL_PORT} not ready"
            assert wait_doris_be_ready(300), "Doris BE not ready (no online backends). Use --clickhouse-only to run without Doris."
        assert wait_port("127.0.0.1", _port_of(CH_HTTP), 180), f"ClickHouse {_port_of(CH_HTTP)} not ready"
        assert wait_port("127.0.0.1", _port_of(DRUID_HTTP), 300), f"Druid {_port_of(DRUID_HTTP)} not ready"
        assert wait_druid_ready(300), "Druid not ready"
        assert wait_port("127.0.0.1", OB_PORT, 360), f"OceanBase {OB_PORT} not ready (bootstrap ~3-5 min)"
    loki_available = not vm_only and wait_port("127.0.0.1", _port_of(LOKI_HTTP), 60)
    if not loki_available and not vm_only:
        print(f"[bench] Loki not available (port {_port_of(LOKI_HTTP)}), skipping logs-only backend")
    vm_available = (wait_port("127.0.0.1", _port_of(VM_HTTP), 60)
                    and wait_port("127.0.0.1", _port_of(VL_HTTP), 60)
                    and wait_port("127.0.0.1", _port_of(VT_HTTP), 60))
    if not vm_available:
        print("[bench] VictoriaMetrics stack not fully available (8428/9428/10428), skipping VM backend")
    if vm_only:
        assert vm_available, "VM services not available. Run: make up-vm"

    tsdir = args.out / f"storage_bench_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_id = tsdir.name
    tsdir.mkdir(parents=True, exist_ok=True)
    doris_ingest = {"status": "skipped"}
    ch_ingest = {"status": "skipped"}
    druid_ingest = {"status": "skipped"}
    ob_ingest = {"status": "skipped"}
    doris_qres = {}
    ch_qres = {}
    druid_qres = {}
    ob_qres = {}
    loki_ingest = {"status": "skipped"}
    vl_ingest = {"status": "skipped"}
    vm_ingest = {"status": "skipped"}
    vt_ingest = {"status": "skipped"}

    if args.init or args.all:
        if use_sql_backends:
            if use_doris:
                apply_doris_schema()
            apply_clickhouse_schema()
            apply_oceanbase_schema()

    if args.all:
        if use_sql_backends:
            if use_doris:
                truncate_doris_tables()
            truncate_clickhouse_tables()
            truncate_oceanbase_tables()
        if vm_available:
            reset_vm_storage()
            assert wait_vm_healthy(120), "VM services not healthy after storage reset"
        stats_dir = tsdir / "ingest_stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        batch_arg = args.streaming_batch if getattr(args, "streaming_batch", None) else args.batch
        if use_doris:
            cmd = ["python3", str(ROOT / "loaders" / "replay_doris.py"), "--data-dir", str(args.data_dir), "--batch", str(batch_arg),
                   "--stats", str(stats_dir / "doris.json")]
            if getattr(args, "scale_to", None):
                cmd.extend(["--scale-to", str(args.scale_to)])
            t0 = time.time()
            subprocess.run(cmd, check=True, env={**os.environ, "DORIS_USER": "root", "DORIS_PASS": DORIS_PASS})
            doris_ingest["status"] = "ok"
            doris_ingest["duration_s"] = round(time.time() - t0, 2)
            doris_ingest["mechanism"] = f"Batch file load ({batch_arg} rows)"
            if (stats_dir / "doris.json").exists():
                s = json.loads((stats_dir / "doris.json").read_text())
                doris_ingest["rows"] = s.get("logs", 0) + s.get("spans", 0) + s.get("metrics", 0)
                doris_ingest["rows_per_sec"] = round(doris_ingest["rows"] / doris_ingest["duration_s"], 0) if doris_ingest["duration_s"] > 0 else 0
        if use_sql_backends:
            ch_cmd = ["python3", str(ROOT / "loaders" / "replay_clickhouse.py"), "--data-dir", str(args.data_dir), "--batch", str(batch_arg),
                      "--stats", str(stats_dir / "clickhouse.json")]
            if getattr(args, "scale_to", None):
                ch_cmd.extend(["--scale-to", str(args.scale_to)])
            t0 = time.time()
            subprocess.run(ch_cmd, check=True, env={**os.environ, "CLICKHOUSE_PASSWORD": CH_PASSWORD})
            ch_ingest["status"] = "ok"
            ch_ingest["duration_s"] = round(time.time() - t0, 2)
            ch_ingest["mechanism"] = f"Batch file load ({batch_arg} rows)"
            if (stats_dir / "clickhouse.json").exists():
                s = json.loads((stats_dir / "clickhouse.json").read_text())
                ch_ingest["rows"] = s.get("logs", 0) + s.get("spans", 0) + s.get("metrics", 0)
                ch_ingest["rows_per_sec"] = round(ch_ingest["rows"] / ch_ingest["duration_s"], 0) if ch_ingest["duration_s"] > 0 else 0
            druid_cmd = ["python3", str(ROOT / "loaders" / "replay_druid.py"), "--data-dir", str(args.data_dir), "--batch", str(batch_arg),
                         "--stats", str(stats_dir / "druid.json")]
            if getattr(args, "scale_to", None):
                druid_cmd.extend(["--scale-to", str(args.scale_to)])
            t0 = time.time()
            subprocess.run(druid_cmd, check=True, env={**os.environ})
            druid_ingest["status"] = "ok"
            druid_ingest["duration_s"] = round(time.time() - t0, 2)
            druid_ingest["mechanism"] = f"Batch file load ({batch_arg} rows)"
            if (stats_dir / "druid.json").exists():
                s = json.loads((stats_dir / "druid.json").read_text())
                druid_ingest["rows"] = s.get("logs", 0) + s.get("spans", 0) + s.get("metrics", 0)
                druid_ingest["rows_per_sec"] = round(druid_ingest["rows"] / druid_ingest["duration_s"], 0) if druid_ingest["duration_s"] > 0 else 0
            ob_cmd = ["python3", str(ROOT / "loaders" / "replay_oceanbase.py"), "--data-dir", str(args.data_dir), "--batch", str(batch_arg),
                      "--stats", str(stats_dir / "oceanbase.json")]
            if getattr(args, "scale_to", None):
                ob_cmd.extend(["--scale-to", str(args.scale_to)])
            t0 = time.time()
            try:
                subprocess.run(ob_cmd, check=True, env={**os.environ, "OCEANBASE_HOST": "127.0.0.1", "OCEANBASE_PORT": "2881"})
                ob_ingest["status"] = "ok"
                ob_ingest["duration_s"] = round(time.time() - t0, 2)
                ob_ingest["mechanism"] = f"Batch file load ({batch_arg} rows)"
                if (stats_dir / "oceanbase.json").exists():
                    s = json.loads((stats_dir / "oceanbase.json").read_text())
                    ob_ingest["rows"] = s.get("logs", 0) + s.get("spans", 0) + s.get("metrics", 0)
                    ob_ingest["rows_per_sec"] = round(ob_ingest["rows"] / ob_ingest["duration_s"], 0) if ob_ingest["duration_s"] > 0 else 0
            except Exception as e:
                ob_ingest["status"] = "error"
                ob_ingest["error"] = str(e)[:200]
        if loki_available:
            # Use smaller batch for Loki to avoid ingestion rate limit (default 4MB/s)
            loki_batch = min(25, batch_arg)  # Keep small to avoid Loki 4MB/s rate limit with large log lines
            loki_cmd = ["python3", str(ROOT / "loaders" / "replay_loki.py"), "--data-dir", str(args.data_dir),
                        "--batch", str(loki_batch), "--run-id", run_id, "--stats", str(stats_dir / "loki.json")]
            if getattr(args, "scale_to", None):
                loki_cmd.extend(["--scale-to", str(args.scale_to)])
            t0 = time.time()
            try:
                subprocess.run(loki_cmd, check=True, env=os.environ.copy())
                loki_ingest["status"] = "ok"
                loki_ingest["duration_s"] = round(time.time() - t0, 2)
                loki_ingest["mechanism"] = "Loki push (logs only)"
                if (stats_dir / "loki.json").exists():
                    s = json.loads((stats_dir / "loki.json").read_text())
                    loki_ingest["rows"] = s.get("logs", 0)
                    loki_ingest["rows_per_sec"] = round(loki_ingest["rows"] / loki_ingest["duration_s"], 0) if loki_ingest["duration_s"] > 0 else 0
            except Exception as e:
                loki_ingest["status"] = "error"
                loki_ingest["error"] = str(e)[:200]
        if vm_available:
            vm_cmd = ["python3", str(ROOT / "loaders" / "replay_victoriametrics.py"), "--data-dir", str(args.data_dir),
                      "--batch", str(batch_arg), "--run-id", run_id, "--stats", str(stats_dir / "vm.json")]
            if getattr(args, "scale_to", None):
                vm_cmd.extend(["--scale-to", str(args.scale_to)])
            t0 = time.time()
            try:
                subprocess.run(vm_cmd, check=True, env={**os.environ,
                               "VM_HTTP": VM_HTTP, "VL_HTTP": VL_HTTP, "VT_HTTP": VT_HTTP})
                total_dur = round(time.time() - t0, 2)
                if (stats_dir / "vm.json").exists():
                    s = json.loads((stats_dir / "vm.json").read_text())
                else:
                    s = {}
                for mech, key, dur_key, target in [
                    ("VictoriaLogs push (logs only)", "logs", "logs_duration_s", vl_ingest),
                    ("VictoriaMetrics push (metrics only)", "metrics", "metrics_duration_s", vm_ingest),
                    ("VictoriaTraces push (traces only)", "spans", "spans_duration_s", vt_ingest),
                ]:
                    rows = s.get(key, 0)
                    dur = s.get(dur_key, total_dur)
                    target["status"] = "ok"
                    target["duration_s"] = dur
                    target["mechanism"] = mech
                    target["rows"] = rows
                    target["rows_per_sec"] = round(rows / dur, 0) if dur > 0 else 0
            except Exception as e:
                for target in (vl_ingest, vm_ingest, vt_ingest):
                    target["status"] = "error"
                    target["error"] = str(e)[:200]
        if use_sql_backends:
            # Wait for Druid segments to be available for querying
            print("[wait] Druid segments loading...")
            for _ in range(24):
                try:
                    r = requests.get(f"{DRUID_HTTP}/proxy/coordinator/druid/coordinator/v1/metadata/datasources", timeout=5)
                    if r.status_code == 200:
                        ds = r.json()
                        if "telemetry_logs" in ds and "telemetry_spans" in ds and "telemetry_metrics" in ds:
                            print("[wait] Druid datasources ready")
                            break
                except Exception:
                    pass
                time.sleep(5)
            time.sleep(15)

    otlp_ingest = None
    if getattr(args, "otlp", False) and use_sql_backends:
        assert wait_port("127.0.0.1", 4317, 60), "OTLP collector 4317 not ready. Run: make up-otel"
        stats_path = tsdir / "ingest_stats" / "otlp.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "python3", str(ROOT / "runner" / "run_otlp_ingest.py"),
            "--stats", str(stats_path),
            "--count", str(getattr(args, "otlp_count", 1000)),
        ], check=True, env={**os.environ, "CLICKHOUSE_PASSWORD": CH_PASSWORD,
                            "CLICKHOUSE_HTTP": CH_HTTP,
                            "VM_HTTP": VM_HTTP, "VL_HTTP": VL_HTTP, "VT_HTTP": VT_HTTP})
        # Map otel.* into telemetry.* so canonical queries run against batch + OTLP data
        subprocess.run([
            "python3", str(ROOT / "runner" / "map_otlp_to_telemetry.py"), "--both",
        ], check=True, env={**os.environ, "CLICKHOUSE_PASSWORD": CH_PASSWORD, "DORIS_PASS": DORIS_PASS})
        if stats_path.exists():
            otlp_ingest = json.loads(stats_path.read_text())

    if use_sql_backends:
        if use_doris:
            doris_qres = bench_backend(DORIS_QDIR, run_doris_query)
        ch_qres = bench_backend(CH_QDIR, run_clickhouse_query)
        druid_qres = bench_backend(DRUID_QDIR, run_druid_query)
        ob_qres = bench_backend(OB_QDIR, run_oceanbase_query)
    loki_qres = bench_loki_logs(run_id) if loki_ingest.get("status") == "ok" else {}
    vm_any_ok = any(d.get("status") == "ok" for d in (vl_ingest, vm_ingest, vt_ingest))
    if vm_any_ok:
        print("[wait] waiting for VictoriaMetrics to flush ingested data...")
        for i in range(30):
            try:
                r = requests.get(f"{VM_HTTP}/api/v1/query",
                                 params={"query": 'count({__name__!=""})', "step": "24h"}, timeout=5)
                if r.status_code == 200 and r.json().get("data", {}).get("result"):
                    print(f"[wait] VictoriaMetrics data ready after {i * 2}s")
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            print("[wait] VictoriaMetrics data not ready after 60s, proceeding anyway")
    vl_qres, vm_qres, vt_qres = bench_vm(run_id) if vm_any_ok else ({}, {}, {})

    data_vol = get_data_volume(use_doris and use_sql_backends, use_oceanbase=use_sql_backends, use_vm=vm_any_ok, use_sql=use_sql_backends)

    write_combined_report(tsdir, doris_ingest, ch_ingest, druid_ingest, doris_qres, ch_qres, druid_qres,
                          ob_ingest=ob_ingest, ob_qres=ob_qres, loki_ingest=loki_ingest, loki_qres=loki_qres,
                          vl_ingest=vl_ingest, vl_qres=vl_qres,
                          vm_ingest=vm_ingest, vm_qres=vm_qres,
                          vt_ingest=vt_ingest, vt_qres=vt_qres,
                          otlp_ingest=otlp_ingest, data_vol=data_vol)
    write_rolling_report(args.out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
