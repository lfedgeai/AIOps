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

DORIS_FE_HTTP = os.getenv("DORIS_FE_HTTP", "http://localhost:8030")
DORIS_PASS = os.getenv("DORIS_PASS", "")
CH_HTTP = os.getenv("CLICKHOUSE_HTTP", "http://localhost:8123")
CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
DRUID_HTTP = os.getenv("DRUID_HTTP", "http://localhost:8888")
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


def get_data_volume(use_doris: bool) -> tuple[dict, dict, dict]:
    """Run full-scan COUNT on each backend; return (doris_vol, ch_vol, druid_vol)."""
    doris_vol = {}
    ch_vol = {}
    druid_vol = {}
    sql_doris = (DORIS_QDIR / "data_volume.sql").read_text()
    sql_ch = (CH_QDIR / "data_volume.sql").read_text()
    sql_druid = (DRUID_QDIR / "data_volume.sql").read_text()

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

    return doris_vol, ch_vol, druid_vol

def write_combined_report(out_dir: Path, doris_ingest: dict, ch_ingest: dict, druid_ingest: dict,
                         doris_qres: dict, ch_qres: dict, druid_qres: dict,
                         otlp_ingest: dict | None = None,
                         data_vol: tuple[dict, dict, dict] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_queries = sorted(set(doris_qres.keys()) | set(ch_qres.keys()) | set(druid_qres.keys()))
    rows = []
    chart_data = {"queries": [], "doris": [], "clickhouse": [], "druid": []}
    for q in all_queries:
        d = doris_qres.get(q, {})
        c = ch_qres.get(q, {})
        dr = druid_qres.get(q, {})
        d_lat = d.get("latency_s", "")
        d_rows = d.get("rows", "")
        d_err = d.get("error", "")
        c_lat = c.get("latency_s", "")
        c_rows = c.get("rows", "")
        c_err = c.get("error", "")
        dr_lat = dr.get("latency_s", "")
        dr_rows = dr.get("rows", "")
        dr_err = dr.get("error", "")
        winner = ""
        pct_diff = ""
        lats = [(d_lat, "Doris"), (c_lat, "ClickHouse"), (dr_lat, "Druid")]
        valid = [(x, n) for x, n in lats if isinstance(x, (int, float))]
        if len(valid) >= 2:
            fastest = min(valid, key=lambda t: t[0])
            slowest = max(valid, key=lambda t: t[0])
            winner = fastest[1]
            if slowest[0] >= 1e-9:
                pct = (slowest[0] - fastest[0]) / slowest[0] * 100
                pct_diff = f"{pct:.1f}%"
        rows.append(f"<tr><td>{q}</td><td>{d_lat}</td><td>{d_rows}</td><td>{d_err}</td>"
                   f"<td>{c_lat}</td><td>{c_rows}</td><td>{c_err}</td>"
                   f"<td>{dr_lat}</td><td>{dr_rows}</td><td>{dr_err}</td>"
                   f"<td>{pct_diff}</td><td>{winner}</td></tr>")
        chart_data["queries"].append(q)
        chart_data["doris"].append(round(d_lat, 4) if isinstance(d_lat, (int, float)) else None)
        chart_data["clickhouse"].append(round(c_lat, 4) if isinstance(c_lat, (int, float)) else None)
        chart_data["druid"].append(round(dr_lat, 4) if isinstance(dr_lat, (int, float)) else None)
    ingest_chart = {
        "labels": ["Doris", "ClickHouse", "Druid"],
        "duration_s": [
            doris_ingest.get("duration_s"),
            ch_ingest.get("duration_s"),
            druid_ingest.get("duration_s"),
        ],
        "rows_per_sec": [
            doris_ingest.get("rows_per_sec"),
            ch_ingest.get("rows_per_sec"),
            druid_ingest.get("rows_per_sec"),
        ],
    }
    otlp_rows = ""
    if otlp_ingest:
        mech = otlp_ingest.get("mechanism", "OTLP")
        dur = otlp_ingest.get("duration_s", "-")
        for backend, key in [("Doris", "doris"), ("ClickHouse", "clickhouse")]:
            d = otlp_ingest.get(key, {})
            r = d.get("rows", "-")
            rps = d.get("rows_per_sec", "-")
            otlp_rows += f'    <tr><td>{backend}</td><td>{mech}</td><td>{dur}</td><td>{r}</td><td>{rps}</td></tr>\n'

    data_vol_row = ""
    if data_vol:
        d_vol, c_vol, dr_vol = data_vol
        def _fmt(v: dict) -> str:
            if not v or "error" in v:
                return v.get("error", "-") if v else "-"
            total = v.get("total", 0)
            return f"{total:,} (logs={v.get('logs',0):,}, spans={v.get('spans',0):,}, metrics={v.get('metrics',0):,})"
        def _lat(v: dict) -> str:
            lat = v.get("latency_s") if v else None
            return f"{lat:.3f}s" if isinstance(lat, (int, float)) else "-"
        data_vol_row = f"""
  <h3>Data volume (full scan)</h3>
  <p>Total rows in telemetry tables at query time. Latency = full COUNT(*) time.</p>
  <table>
    <tr><th>Backend</th><th>Total rows (logs, spans, metrics)</th><th>Full-scan latency</th></tr>
    <tr><td>Doris</td><td>{_fmt(d_vol)}</td><td>{_lat(d_vol)}</td></tr>
    <tr><td>ClickHouse</td><td>{_fmt(c_vol)}</td><td>{_lat(c_vol)}</td></tr>
    <tr><td>Druid</td><td>{_fmt(dr_vol)}</td><td>{_lat(dr_vol)}</td></tr>
  </table>
"""

    chart_json = json.dumps(chart_data)
    ingest_json = json.dumps(ingest_chart)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Doris vs ClickHouse vs Druid benchmark</title>
<style>table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 10px}}#chartWrap{{max-width:900px;height:400px;margin:1em 0}}</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head><body>
  <h1>Doris vs ClickHouse vs Druid benchmark</h1>
  <div>Generated at {datetime.now().isoformat()}</div>
  <h2>Ingestion comparison</h2>
  <div id="ingestChartWrap" style="max-width:600px;height:250px;margin:1em 0"><canvas id="ingestChart"></canvas></div>
  <table>
    <tr><th>Backend</th><th>Mechanism</th><th>Duration (s)</th><th>Rows</th><th>Rows/sec</th></tr>
    <tr><td>Doris</td><td>{doris_ingest.get('mechanism', '-')}</td><td>{doris_ingest.get('duration_s', '-')}</td><td>{doris_ingest.get('rows', '-')}</td><td>{doris_ingest.get('rows_per_sec', '-')}</td></tr>
    <tr><td>ClickHouse</td><td>{ch_ingest.get('mechanism', '-')}</td><td>{ch_ingest.get('duration_s', '-')}</td><td>{ch_ingest.get('rows', '-')}</td><td>{ch_ingest.get('rows_per_sec', '-')}</td></tr>
    <tr><td>Druid</td><td>{druid_ingest.get('mechanism', '-')}</td><td>{druid_ingest.get('duration_s', '-')}</td><td>{druid_ingest.get('rows', '-')}</td><td>{druid_ingest.get('rows_per_sec', '-')}</td></tr>
    {otlp_rows}
  </table>
  <p><small>Raw ingest: Doris {json.dumps(doris_ingest)} | CH {json.dumps(ch_ingest)} | Druid {json.dumps(druid_ingest)}</small></p>
  {data_vol_row}
  <h2>Query latency (seconds)</h2>
  <div id="chartWrap"><canvas id="latencyChart"></canvas></div>
  <h2>Query comparison</h2>
  <table>
    <tr><th>query</th><th>Doris lat</th><th>Doris rows</th><th>Doris err</th>
        <th>CH lat</th><th>CH rows</th><th>CH err</th>
        <th>Druid lat</th><th>Druid rows</th><th>Druid err</th>
        <th>% diff</th><th>faster</th></tr>
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
          datasets: [{{ label: 'Ingestion time (s)', data: ingestData.duration_s, backgroundColor: ['rgba(54,162,235,0.7)','rgba(75,192,192,0.7)','rgba(255,159,64,0.7)'] }}]
        }},
        options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true }} }} }}
      }});
    }}
    const ctx = document.getElementById('latencyChart').getContext('2d');
    new Chart(ctx, {{
      type: 'bar',
      data: {{
        labels: data.queries,
        datasets: [
          {{ label: 'Doris', data: data.doris, backgroundColor: 'rgba(54,162,235,0.7)' }},
          {{ label: 'ClickHouse', data: data.clickhouse, backgroundColor: 'rgba(75,192,192,0.7)' }},
          {{ label: 'Druid', data: data.druid, backgroundColor: 'rgba(255,159,64,0.7)' }}
        ]
      }},
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
            entries.append((d.name, "compare.html", "Doris vs ClickHouse vs Druid", ts))
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
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--batch", type=int, default=5000, help="Rows per load batch")
    ap.add_argument("--scale-to", type=int, default=None, help="Target row count per type (50k = 50000)")
    ap.add_argument("--streaming-batch", type=int, default=None,
                    help="Smaller batch size to simulate real-time ingestion (e.g. 500 = 100 batches of 500 rows)")
    ap.add_argument("--otlp", action="store_true", help="Also run OTLP ingestion via telemetrygen")
    ap.add_argument("--otlp-count", type=int, default=1000, help="Spans, logs, metrics each for OTLP (default 1000)")
    args = ap.parse_args()

    use_doris = not args.clickhouse_only
    if use_doris:
        assert wait_port("127.0.0.1", 8030, 180), "Doris FE 8030 not ready"
        assert wait_port("127.0.0.1", 9030, 180), "Doris MySQL 9030 not ready"
        assert wait_doris_be_ready(300), "Doris BE not ready (no online backends). Use --clickhouse-only to run without Doris."
    assert wait_port("127.0.0.1", 8123, 180), "ClickHouse 8123 not ready"
    assert wait_port("127.0.0.1", 8888, 300), "Druid 8888 not ready"
    assert wait_druid_ready(300), "Druid not ready"

    tsdir = args.out / f"storage_bench_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tsdir.mkdir(parents=True, exist_ok=True)
    doris_ingest = {"status": "skipped"}
    ch_ingest = {"status": "skipped"}
    druid_ingest = {"status": "skipped"}
    doris_qres = {}
    ch_qres = {}
    druid_qres = {}

    if args.init or args.all:
        if use_doris:
            apply_doris_schema()
        apply_clickhouse_schema()

    if args.all:
        if use_doris:
            truncate_doris_tables()
        truncate_clickhouse_tables()
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
    if getattr(args, "otlp", False):
        assert wait_port("127.0.0.1", 4317, 60), "OTLP collector 4317 not ready. Run: make up-otel"
        stats_path = tsdir / "ingest_stats" / "otlp.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "python3", str(ROOT / "runner" / "run_otlp_ingest.py"),
            "--stats", str(stats_path),
            "--count", str(getattr(args, "otlp_count", 1000)),
        ], check=True, env={**os.environ, "CLICKHOUSE_PASSWORD": CH_PASSWORD})
        # Map otel.* into telemetry.* so canonical queries run against batch + OTLP data
        subprocess.run([
            "python3", str(ROOT / "runner" / "map_otlp_to_telemetry.py"), "--both",
        ], check=True, env={**os.environ, "CLICKHOUSE_PASSWORD": CH_PASSWORD, "DORIS_PASS": DORIS_PASS})
        if stats_path.exists():
            otlp_ingest = json.loads(stats_path.read_text())

    if use_doris:
        doris_qres = bench_backend(DORIS_QDIR, run_doris_query)
    ch_qres = bench_backend(CH_QDIR, run_clickhouse_query)
    druid_qres = bench_backend(DRUID_QDIR, run_druid_query)

    data_vol = get_data_volume(use_doris)

    write_combined_report(tsdir, doris_ingest, ch_ingest, druid_ingest, doris_qres, ch_qres, druid_qres, otlp_ingest, data_vol)
    write_rolling_report(args.out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
