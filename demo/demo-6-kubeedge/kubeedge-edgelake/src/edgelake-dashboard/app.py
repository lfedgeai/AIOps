import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import requests

app = FastAPI(title="EdgeLake Observability Dashboard")

EDGELAKE_URL = os.getenv(
    "EDGELAKE_URL",
    "http://127.0.0.1:32159",
)


def edgelake_query(sql):
    # EdgeLake receives SQL through the HTTP command header.
    # HTTP header values cannot contain newlines, so normalize
    # multiline SQL into a single line.
    sql = " ".join(sql.split())
    command = f"sql monitoring {sql}"

    response = requests.get(
        EDGELAKE_URL,
        headers={
            "command": command,
            "User-Agent": "curl/8.0.0",
        },
        timeout=5,
    )

    response.raise_for_status()

    payload = response.json()
    return payload.get("Query", [])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "edgelake-dashboard",
    }


@app.get("/api/latest")
def latest():
    rows = edgelake_query(
        """
        select
          row_id,
          timestamp,
          edge_node,
          cpu_percent,
          memory_used_bytes,
          memory_free_bytes,
          memory_total_bytes,
          memory_percent,
          disk_used_bytes,
          disk_free_bytes,
          disk_total_bytes,
          disk_percent,
          network_recv_bytes,
          network_sent_bytes,
          load_1,
          load_5,
          load_15
        from otel_edge_metrics
        order by timestamp desc
        limit 1
        """
    )

    return rows[0] if rows else {}


@app.get("/api/history")
def history(limit: int = 60):
    limit = max(2, min(limit, 360))

    rows = edgelake_query(
        f"""
        select
          row_id,
          timestamp,
          edge_node,
          cpu_percent,
          memory_percent,
          disk_percent,
          network_recv_bytes,
          network_sent_bytes,
          load_1,
          load_5,
          load_15
        from otel_edge_metrics
        order by timestamp desc
        limit {limit}
        """
    )

    rows.reverse()

    previous = None

    for row in rows:
        row["rx_bytes_sec"] = 0
        row["tx_bytes_sec"] = 0

        if previous:
            try:
                from datetime import datetime

                current_time = datetime.fromisoformat(
                    row["timestamp"]
                )
                previous_time = datetime.fromisoformat(
                    previous["timestamp"]
                )

                seconds = (
                    current_time - previous_time
                ).total_seconds()

                if seconds > 0:
                    rx_delta = (
                        int(row["network_recv_bytes"])
                        - int(previous["network_recv_bytes"])
                    )

                    tx_delta = (
                        int(row["network_sent_bytes"])
                        - int(previous["network_sent_bytes"])
                    )

                    if rx_delta >= 0:
                        row["rx_bytes_sec"] = (
                            rx_delta / seconds
                        )

                    if tx_delta >= 0:
                        row["tx_bytes_sec"] = (
                            tx_delta / seconds
                        )

            except Exception:
                pass

        previous = row

    return rows

# ---------------------------------------------------------
# Web Dashboard
# Keep this AFTER all /api routes.
# ---------------------------------------------------------
app.mount(
    "/",
    StaticFiles(directory="static", html=True),
    name="static",
)
