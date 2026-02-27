#!/usr/bin/env python3
"""
Compare Doris vs ClickHouse: run same queries on both backends, produce combined report.

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

DORIS_FE_HTTP = os.getenv("DORIS_FE_HTTP", "http://localhost:8030")
DORIS_PASS = os.getenv("DORIS_PASS", "")
CH_HTTP = os.getenv("CLICKHOUSE_HTTP", "http://localhost:8123")
CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
DB = "telemetry"

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
        r = requests.post(CH_HTTP, params=_ch_params(), data=stmt + ";", timeout=30)
        if r.status_code != 200:
            print(f"[schema] ClickHouse: {r.status_code} {r.text[:200]}")
    print("[schema] ClickHouse applied")

def truncate_clickhouse_tables() -> None:
    for t in ("logs", "spans", "metrics"):
        r = requests.post(CH_HTTP, params=_ch_params(), data=f"TRUNCATE TABLE {DB}.{t};", timeout=30)
        if r.status_code != 200:
            print(f"[truncate] ClickHouse {t}: {r.status_code} {r.text[:200]}")
    print("[truncate] ClickHouse tables cleared")

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

def write_combined_report(out_dir: Path, doris_ingest: dict, ch_ingest: dict,
                         doris_qres: dict, ch_qres: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_queries = sorted(set(doris_qres.keys()) | set(ch_qres.keys()))
    rows = []
    for q in all_queries:
        d = doris_qres.get(q, {})
        c = ch_qres.get(q, {})
        d_lat = d.get("latency_s", "")
        d_rows = d.get("rows", "")
        d_err = d.get("error", "")
        c_lat = c.get("latency_s", "")
        c_rows = c.get("rows", "")
        c_err = c.get("error", "")
        winner = ""
        if isinstance(d_lat, (int, float)) and isinstance(c_lat, (int, float)):
            winner = "Doris" if d_lat < c_lat else "ClickHouse"
        rows.append(f"<tr><td>{q}</td><td>{d_lat}</td><td>{d_rows}</td><td>{d_err}</td>"
                   f"<td>{c_lat}</td><td>{c_rows}</td><td>{c_err}</td><td>{winner}</td></tr>")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Doris vs ClickHouse benchmark</title>
<style>table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 10px}}</style>
</head><body>
  <h1>Doris vs ClickHouse benchmark</h1>
  <div>Generated at {datetime.now().isoformat()}</div>
  <h2>Ingest</h2>
  <p><b>Doris:</b> {json.dumps(doris_ingest)}</p>
  <p><b>ClickHouse:</b> {json.dumps(ch_ingest)}</p>
  <h2>Query comparison</h2>
  <table>
    <tr><th>query</th><th>Doris latency (s)</th><th>Doris rows</th><th>Doris error</th>
        <th>ClickHouse latency (s)</th><th>ClickHouse rows</th><th>ClickHouse error</th>
        <th>faster</th></tr>
    {chr(10).join(rows)}
  </table>
</body></html>"""
    (out_dir / "compare.html").write_text(html)
    (out_dir / "doris_queries.json").write_text(json.dumps(doris_qres, indent=2))
    (out_dir / "clickhouse_queries.json").write_text(json.dumps(ch_qres, indent=2))
    print(f"[bench] wrote {out_dir}/compare.html")

def write_rolling_report(out_base: Path) -> None:
    """Write rolling_index.html with both Doris and Doris vs ClickHouse runs."""
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

def main() -> int:
    ap = argparse.ArgumentParser(description="Compare Doris vs ClickHouse")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    args = ap.parse_args()

    assert wait_port("127.0.0.1", 8030, 180), "Doris FE 8030 not ready"
    assert wait_port("127.0.0.1", 9030, 180), "Doris MySQL 9030 not ready"
    assert wait_doris_be_ready(300), "Doris BE not ready (no online backends)"
    assert wait_port("127.0.0.1", 8123, 180), "ClickHouse 8123 not ready"

    tsdir = args.out / f"storage_bench_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    doris_ingest = {"status": "skipped"}
    ch_ingest = {"status": "skipped"}
    doris_qres = {}
    ch_qres = {}

    if args.init or args.all:
        apply_doris_schema()
        apply_clickhouse_schema()

    if args.all:
        truncate_doris_tables()
        truncate_clickhouse_tables()
        subprocess.run([
            "python3", str(ROOT / "loaders" / "replay.py"),
            "--data-dir", str(args.data_dir), "--batch", "5000",
        ], check=True, env={**os.environ, "DORIS_USER": "root", "DORIS_PASS": DORIS_PASS})
        doris_ingest["status"] = "ok"
        subprocess.run([
            "python3", str(ROOT / "loaders" / "replay_clickhouse.py"),
            "--data-dir", str(args.data_dir), "--batch", "5000",
        ], check=True, env={**os.environ, "CLICKHOUSE_PASSWORD": CH_PASSWORD})
        ch_ingest["status"] = "ok"

    doris_qres = bench_backend(DORIS_QDIR, run_doris_query)
    ch_qres = bench_backend(CH_QDIR, run_clickhouse_query)

    write_combined_report(tsdir, doris_ingest, ch_ingest, doris_qres, ch_qres)
    write_rolling_report(args.out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
