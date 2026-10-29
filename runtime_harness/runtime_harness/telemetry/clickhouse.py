"""ClickHouse OTEL queries — logs / metrics / traces + service adoption.

Purpose
-------
Talk to the OTEL→ClickHouse path used in the Oct-12 research harness. Telemetry
is how applications are **self-adopted** (``ServiceName`` inventory), not via a
manual app registry.

How it connects
---------------
- Used by ``slots/telemetry.py`` during each cycle.
- Exposed as MCP tools in ``mcp/server.py`` (``list_services``, ``search_*``).
- Downstream detect/RCA/remediate consume ``ctx["signals"]`` built from these queries.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import httpx

DEFAULT_CLICKHOUSE_HTTP = os.environ.get(
    "CLICKHOUSE_RT_HTTP",
    os.environ.get("CLICKHOUSE_HTTP", "http://127.0.0.1:38124"),
)
DEFAULT_DATABASE = os.environ.get("CLICKHOUSE_DATABASE", "otel")


def _since_for_clickhouse(since_ts: str) -> str:
    """Convert ISO8601 to ClickHouse-friendly format (YYYY-MM-DD HH:MM:SS)."""
    try:
        dt = datetime.fromisoformat(since_ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return since_ts


def _default_since(window_minutes: int = 30) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).isoformat()


def _esc(value: str) -> str:
    return value.replace("'", "''")


class ClickHouseClient:
    """Thin HTTP client for the OTEL ClickHouse exporter tables."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        database: str = DEFAULT_DATABASE,
        timeout_s: float = 30.0,
    ):
        self.base_url = (base_url or DEFAULT_CLICKHOUSE_HTTP).rstrip("/")
        self.database = database
        self.timeout_s = timeout_s

    @property
    def reachable_hint(self) -> bool:
        """True when URL does not look like the local stub default."""
        return "127.0.0.1" not in self.base_url and bool(self.base_url)

    def query(self, sql: str) -> str:
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(self.base_url, content=sql.encode("utf-8"))
                resp.raise_for_status()
                return (resp.text or "").strip()
        except Exception as exc:  # noqa: BLE001 — surface as tool error string
            return f"Error: {exc}"

    def ping(self) -> bool:
        out = self.query("SELECT 1 FORMAT TabSeparated")
        return out.strip() == "1"


def search_logs(
    query: str = "",
    since_ts: Optional[str] = None,
    service_name: Optional[str] = None,
    limit: int = 20,
    *,
    client: Optional[ClickHouseClient] = None,
) -> str:
    """Search OTEL logs by Body text / ServiceName (research search_logs pattern)."""
    ch = client or ClickHouseClient()
    since_sql = _since_for_clickhouse(since_ts or _default_since())
    conditions = [f"Timestamp >= '{since_sql}'"]
    if query:
        conditions.append(f"positionCaseInsensitive(Body, '{_esc(query)}') > 0")
    if service_name:
        conditions.append(f"ServiceName = '{_esc(service_name)}'")
    where = " AND ".join(conditions)
    sql = f"""
    SELECT formatDateTime(Timestamp, '%Y-%m-%d %H:%M:%S') as ts, ServiceName, Body
    FROM {ch.database}.otel_logs
    WHERE {where}
    ORDER BY Timestamp DESC
    LIMIT {int(limit)}
    FORMAT TabSeparated
    """
    out = ch.query(sql)
    if out.startswith("Error:"):
        return _stub_logs(query, service_name, since_sql, error=out)
    lines = [ln for ln in out.split("\n") if ln]
    if not lines:
        return "No matching logs found."
    results = []
    for ln in lines:
        parts = ln.split("\t", 2)
        if len(parts) >= 3:
            results.append(
                {"ts": parts[0], "service": parts[1], "message": parts[2][:500]}
            )
    return json.dumps(results, indent=2)


def search_traces(
    service_name: Optional[str] = None,
    since_ts: Optional[str] = None,
    min_duration_ms: Optional[float] = None,
    limit: int = 20,
    *,
    client: Optional[ClickHouseClient] = None,
) -> str:
    """Search OTEL traces — filter by service, time, min duration."""
    ch = client or ClickHouseClient()
    conditions = ["1=1"]
    since_sql = _since_for_clickhouse(since_ts or _default_since())
    conditions.append(f"Timestamp >= '{since_sql}'")
    if service_name:
        conditions.append(f"ServiceName = '{_esc(service_name)}'")
    if min_duration_ms is not None:
        ns = int(float(min_duration_ms) * 1_000_000)
        conditions.append(f"Duration > {ns}")
    where = " AND ".join(conditions)
    sql = f"""
    SELECT formatDateTime(Timestamp, '%Y-%m-%d %H:%M:%S') as ts,
           ServiceName,
           Duration/1000000 as duration_ms,
           StatusCode,
           SpanName
    FROM {ch.database}.otel_traces
    WHERE {where}
    ORDER BY Timestamp DESC
    LIMIT {int(limit)}
    FORMAT TabSeparated
    """
    out = ch.query(sql)
    if out.startswith("Error:"):
        return _stub_traces(service_name, since_sql, error=out)
    lines = [ln for ln in out.split("\n") if ln]
    if not lines:
        return "No matching traces found."
    results = []
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) >= 3:
            try:
                row: dict[str, Any] = {
                    "ts": parts[0],
                    "service": parts[1],
                    "duration_ms": float(parts[2]),
                }
                if len(parts) >= 4:
                    row["status"] = parts[3]
                if len(parts) >= 5:
                    row["span"] = parts[4]
                results.append(row)
            except ValueError:
                pass
    return json.dumps(results, indent=2)


def search_metrics(
    metric_name: Optional[str] = None,
    since_ts: Optional[str] = None,
    service_name: Optional[str] = None,
    limit: int = 50,
    *,
    client: Optional[ClickHouseClient] = None,
) -> str:
    """Search OTEL metrics (gauge/sum tables — research search_metrics_data pattern)."""
    ch = client or ClickHouseClient()
    since_sql = _since_for_clickhouse(since_ts or _default_since())
    tables_and_time_col = [
        (f"{ch.database}.otel_metrics_gauge", "TimeUnix"),
        (f"{ch.database}.otel_metrics_sum", "TimeUnix"),
        (f"{ch.database}.otel_metrics", "Timestamp"),
    ]
    errors: list[str] = []
    for table, time_col in tables_and_time_col:
        conditions = [f"{time_col} >= toDateTime64('{since_sql}', 3)"]
        if metric_name:
            conditions.append(f"MetricName = '{_esc(metric_name)}'")
        if service_name:
            # ServiceName column when present; ResourceAttributes fallback omitted for stub simplicity
            conditions.append(f"ServiceName = '{_esc(service_name)}'")
        where = " AND ".join(conditions)
        sql = f"""
        SELECT formatDateTime({time_col}, '%Y-%m-%d %H:%M:%S') as ts,
               MetricName,
               ServiceName,
               Value
        FROM {table}
        WHERE {where}
        ORDER BY {time_col} DESC
        LIMIT {int(limit)}
        FORMAT TabSeparated
        """
        out = ch.query(sql)
        if out.startswith("Error:"):
            errors.append(f"{table}: {out}")
            continue
        lines = [ln for ln in out.split("\n") if ln]
        if not lines:
            continue
        results = []
        for ln in lines:
            parts = ln.split("\t")
            if len(parts) >= 3:
                results.append(
                    {
                        "ts": parts[0],
                        "metric": parts[1],
                        "service": parts[2] if len(parts) > 2 else "",
                        "value": parts[3] if len(parts) > 3 else None,
                    }
                )
        return json.dumps(results, indent=2)
    if errors:
        return _stub_metrics(metric_name, service_name, since_sql, error="; ".join(errors[:2]))
    return "No matching metrics found."


def list_services(
    since_ts: Optional[str] = None,
    limit: int = 50,
    *,
    client: Optional[ClickHouseClient] = None,
) -> str:
    """Discover adopted applications from telemetry ServiceName (self-adoption)."""
    ch = client or ClickHouseClient()
    since_sql = _since_for_clickhouse(since_ts or _default_since(window_minutes=120))
    sql = f"""
    SELECT ServiceName, count() AS c
    FROM (
      SELECT ServiceName FROM {ch.database}.otel_logs
      WHERE Timestamp >= '{since_sql}' AND ServiceName != ''
      UNION ALL
      SELECT ServiceName FROM {ch.database}.otel_traces
      WHERE Timestamp >= '{since_sql}' AND ServiceName != ''
    )
    GROUP BY ServiceName
    ORDER BY c DESC
    LIMIT {int(limit)}
    FORMAT TabSeparated
    """
    out = ch.query(sql)
    if out.startswith("Error:"):
        return json.dumps(
            {
                "source": "stub",
                "error": out,
                "services": [
                    {"service": "cart", "signal_count": 12},
                    {"service": "checkout", "signal_count": 8},
                    {"service": "payment", "signal_count": 5},
                ],
                "note": (
                    "ClickHouse unreachable — stub inventory. "
                    "Wire CLICKHOUSE_HTTP to adopt real services from OTEL."
                ),
            },
            indent=2,
        )
    lines = [ln for ln in out.split("\n") if ln]
    services = []
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) >= 2:
            try:
                services.append({"service": parts[0], "signal_count": int(parts[1])})
            except ValueError:
                services.append({"service": parts[0], "signal_count": 0})
    return json.dumps(
        {"source": "clickhouse", "services": services, "since": since_sql},
        indent=2,
    )


def _stub_logs(
    query: str, service_name: Optional[str], since_sql: str, *, error: str
) -> str:
    svc = service_name or "cart"
    return json.dumps(
        {
            "source": "stub",
            "error": error,
            "since": since_sql,
            "results": [
                {
                    "ts": since_sql,
                    "service": svc,
                    "message": f"[stub] no ready pods / connection refused ({query or 'error'})",
                }
            ],
        },
        indent=2,
    )


def _stub_traces(
    service_name: Optional[str], since_sql: Optional[str], *, error: str
) -> str:
    svc = service_name or "cart"
    return json.dumps(
        {
            "source": "stub",
            "error": error,
            "results": [
                {
                    "ts": since_sql or "",
                    "service": svc,
                    "duration_ms": 2500.0,
                    "status": "Error",
                    "span": "GET /",
                }
            ],
        },
        indent=2,
    )


def _stub_metrics(
    metric_name: Optional[str],
    service_name: Optional[str],
    since_sql: str,
    *,
    error: str,
) -> str:
    return json.dumps(
        {
            "source": "stub",
            "error": error,
            "results": [
                {
                    "ts": since_sql,
                    "metric": metric_name or "k8s.deployment.available",
                    "service": service_name or "cart",
                    "value": "0",
                }
            ],
        },
        indent=2,
    )


def clickhouse_http_url_from_config(config: dict[str, Any] | None) -> str:
    """Resolve ClickHouse HTTP URL from assembly telemetry section / env."""
    cfg = config or {}
    url = str(cfg.get("clickhouse_url") or DEFAULT_CLICKHOUSE_HTTP)
    # Normalize: assembly may use path-style; agent tools use bare HTTP endpoint.
    if "?" in url:
        url = url.split("?", 1)[0]
    return url.rstrip("/")


# ---------------------------------------------------------------------------
# Telemetry status — feeds the UI panel (GET /telemetry-status)
# ---------------------------------------------------------------------------

_FRESHNESS_TABLES = [
    ("logs", "otel_logs", "Timestamp"),
    ("traces", "otel_traces", "Timestamp"),
    ("metrics_gauge", "otel_metrics_gauge", "TimeUnix"),
    ("metrics_sum", "otel_metrics_sum", "TimeUnix"),
]


def _age_s(ts_str: str) -> float | None:
    try:
        dt = datetime.fromisoformat(ts_str.replace(" ", "T", 1).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except ValueError:
        return None


def telemetry_status(client: ClickHouseClient | None = None) -> dict[str, Any]:
    """Health snapshot of the runtime's own ClickHouse ingestion."""
    ch = client or ClickHouseClient()
    status: dict[str, Any] = {"url": ch.base_url, "database": ch.database}

    if not ch.ping():
        status["reachable"] = False
        return status
    status["reachable"] = True

    # Per-table freshness + 1h volume
    tables = []
    for label, table, time_col in _FRESHNESS_TABLES:
        try:
            out = ch.query(
                f"SELECT formatDateTime(max({time_col}), '%Y-%m-%dT%H:%i:%S'), "
                f"countIf({time_col} >= now() - INTERVAL 1 HOUR) "
                f"FROM {ch.database}.{table} FORMAT TabSeparated"
            )
            if out.startswith("Error:") or not out.strip():
                continue
            parts = out.strip().split("\t")
            ts = parts[0]
            tables.append(
                {
                    "table": label,
                    "last_event": ts,
                    "age_s": _age_s(ts),
                    "rows_1h": int(parts[1]) if len(parts) > 1 else 0,
                }
            )
        except Exception:  # noqa: BLE001 — status must never crash the UI
            continue
    status["tables"] = tables

    # Ingestion rate over 5 minutes (logs)
    out = ch.query(
        f"SELECT count() FROM {ch.database}.otel_logs "
        f"WHERE Timestamp >= now() - INTERVAL 5 MINUTE FORMAT TabSeparated"
    )
    if not out.startswith("Error:") and out.strip().isdigit():
        status["logs_last_5m"] = int(out.strip())

    # Adopted services (2h window) with counts
    out = list_services(limit=12, client=ch)
    try:
        import json as _json

        inv = _json.loads(out)
        status["services"] = inv.get("services") or []
        status["source"] = inv.get("source", "clickhouse")
    except Exception:  # noqa: BLE001
        status["services"] = []
    return status
