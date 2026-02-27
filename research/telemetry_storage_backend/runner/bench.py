#!/usr/bin/env python3
"""
Doris storage benchmark runner.

This script orchestrates:
- readiness checks for the Doris FE HTTP and MySQL ports
- schema application using a transient mysql client container
- ingestion via the Stream Load-based loader (loaders/replay.py)
- execution of canonical benchmark queries
- writing a timestamped HTML summary and JSON artifacts

Environment:
- DORIS_FE_HTTP (default: http://localhost:8030)
- DORIS_MYSQL_HOST (default: 127.0.0.1)
- DORIS_MYSQL_PORT (default: 9030)
- DORIS_DB (default: telemetry)
- DORIS_USER (default: root)
- DORIS_PASS (default: empty)
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

FE_HTTP = os.getenv("DORIS_FE_HTTP", "http://localhost:8030")
MYSQL_HOST = os.getenv("DORIS_MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("DORIS_MYSQL_PORT", "9030"))
DB = os.getenv("DORIS_DB", "telemetry")
USER = os.getenv("DORIS_USER", "root")
PASS = os.getenv("DORIS_PASS", "")

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SQL = ROOT / "schemas" / "doris.sql"
QDIR = ROOT / "queries" / "doris"

def wait_port(host: str, port: int, timeout_s: int = 120) -> bool:
    """
    Wait until a TCP port on a host is reachable or timeout elapses.

    Args:
      host: target hostname or IP
      port: TCP port
      timeout_s: max seconds to wait

    Returns:
      True if the port became reachable; False otherwise.
    """
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except Exception:
            time.sleep(1)
    return False

def apply_schema() -> None:
    """
    Apply Doris schema by piping local SQL into a mysql client container.

    Uses Docker's 'tsb-net' network to reach the 'tsb-doris' service,
    avoiding local mysql client dependencies.
    """
    sql = SCHEMA_SQL.read_text()
    # Use mysql client in a transient container to execute SQL (avoids Python deps)
    pass_arg = f"-p{PASS}" if PASS else ""
    cmd = [
        "docker", "run", "--rm", "--network", "tsb-net",
        "mysql:8",
        "sh", "-lc",
        f"echo '{sql}' | mysql -h tsb-doris -P 9030 -u{USER} {pass_arg} -D information_schema"
    ]
    subprocess.run(cmd, check=True)
    print("[schema] applied")

def run_query(sql: str) -> dict:
    """
    Execute a SQL statement against Doris via a mysql client container.

    Args:
      sql: SQL text to execute

    Returns:
      Dict with 'latency_s' and 'rows' (row count in stdout); on failure,
      callers typically catch exceptions and record error info.
    """
    # Use mysql client container for simplicity
    pass_arg = f"-p{PASS}" if PASS else ""
    # Avoid shell quoting pitfalls by writing SQL to a temp file and piping to mysql
    sh_script = (
        "cat > /tmp/bench.sql <<'EOSQL'\n"
        f"{sql}\n"
        "EOSQL\n"
        f"mysql -h tsb-doris -P 9030 -u{USER} {pass_arg} -D {DB} -N -B < /tmp/bench.sql\n"
    )
    cmd = [
        "docker", "run", "--rm", "--network", "tsb-net",
        "mysql:8", "sh", "-lc", sh_script
    ]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if out.returncode != 0:
        return {"error": (out.stderr or out.stdout or "").strip()}
    rows = [ln for ln in (out.stdout or "").splitlines() if ln.strip()]
    return {"latency_s": dt, "rows": len(rows)}

def bench_queries() -> dict:
    """
    Run all .sql files in queries/doris/ and collect timings.

    Returns:
      Mapping of query name to {'latency_s', 'rows'} or {'error'}.
    """
    results = {}
    for f in sorted(QDIR.glob("*.sql")):
        name = f.stem
        sql = f.read_text()
        try:
            m = run_query(sql)
            m["sql"] = sql.strip()
        except Exception as e:
            m = {"error": str(e), "sql": sql.strip()}
        results[name] = m
    return results

def write_summary(out_dir: Path, ingest: dict, qres: dict) -> None:
    """
    Persist benchmark artifacts and render a minimal HTML summary.

    Artifacts:
      - ingest.json: status/notes from ingestion step
      - queries.json: per-query timings and row counts
      - summary.html: human-readable report
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ingest.json").write_text(json.dumps(ingest, indent=2))
    (out_dir / "queries.json").write_text(json.dumps(qres, indent=2))
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Doris storage benchmark</title>
<style>table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 10px;text-align:center}}</style>
</head><body>
  <h1>Doris storage benchmark</h1>
  <div>Generated at {datetime.now().isoformat()}</div>
  <h2>Ingest</h2>
  <pre>{json.dumps(ingest, indent=2)}</pre>
  <h2>Queries</h2>
  <table>
    <tr><th>query</th><th>latency (s)</th><th>rows</th><th>error</th></tr>
    {''.join(f"<tr><td>{k}</td><td>{v.get('latency_s','')}</td><td>{v.get('rows','')}</td><td>{v.get('error','')}</td></tr>" for k,v in qres.items())}
  </table>
</body></html>"""
    (out_dir / "summary.html").write_text(html)
    print(f"[bench] wrote {out_dir}/summary.html")


def write_rolling_report(out_base: Path) -> None:
    """
    Create/update rolling_index.html in out/ listing all runs (Doris + compare) with latest on top.
    """
    import re
    entries = []
    for d in out_base.glob("storage_bench_doris_*"):
        if (d / "summary.html").exists():
            ts = d.name.replace("storage_bench_doris_", "").replace("_", " ", 1)
            entries.append((d.name, "summary.html", "Doris", ts))
    for d in out_base.glob("storage_bench_compare_*"):
        if (d / "compare.html").exists():
            ts = d.name.replace("storage_bench_compare_", "").replace("_", " ", 1)
            entries.append((d.name, "compare.html", "Doris vs ClickHouse", ts))
    # Sort by timestamp (YYYYMMDD_HHMMSS) so newest is first
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
    print(f"[bench] wrote {out_base}/rolling_index.html (rolling report)")

def main() -> int:
    """
    CLI entrypoint for the Doris benchmark harness.

    Flags:
      --data-dir: path to replay source data (otel_ground_truth_data)
      --init: apply schema only
      --queries: run benchmark queries only
      --all: apply schema, replay data, run queries, write report
      --out: base output directory (a timestamped subdir is created)
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--queries", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    args = ap.parse_args()

    assert wait_port("127.0.0.1", 8030, 180), "FE HTTP 8030 not ready"
    assert wait_port("127.0.0.1", 9030, 180), "FE MySQL 9030 not ready"

    tsdir = args.out / f"storage_bench_doris_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ingest = {"notes": "Stream Load via loaders/replay.py", "status": "skipped"}
    qres = {}
    if args.init or args.all:
        apply_schema()
    if args.all:
        # Run loader as a subprocess to keep separation of concerns
        subprocess.run([
            "python3", str(ROOT / "loaders" / "replay.py"),
            "--data-dir", str(args.data_dir),
            "--table-prefix", "otel",
            "--batch", "5000",
        ], check=True)
        ingest["status"] = "ok"
    if args.queries or args.all:
        qres = bench_queries()
    write_summary(tsdir, ingest, qres)
    write_rolling_report(args.out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

