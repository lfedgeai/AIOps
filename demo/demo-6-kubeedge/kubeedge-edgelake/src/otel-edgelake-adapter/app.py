import gzip
import os
import threading
from datetime import datetime

import requests
from fastapi import FastAPI, Request, Response
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
    ExportMetricsServiceResponse,
)

app = FastAPI(title="OTel EdgeLake Adapter")

EDGELAKE_URL = os.getenv(
    "EDGELAKE_URL",
    "http://127.0.0.1:32159",
)
EDGELAKE_DB = os.getenv("EDGELAKE_DB", "monitoring")
EDGELAKE_TABLE = os.getenv(
    "EDGELAKE_TABLE",
    "otel_edge_metrics",
)

# Previous cumulative CPU counters by node.
previous_cpu = {}
cpu_lock = threading.Lock()


def attrs_to_dict(attributes):
    result = {}

    for attr in attributes:
        value = attr.value

        if value.HasField("string_value"):
            result[attr.key] = value.string_value
        elif value.HasField("int_value"):
            result[attr.key] = value.int_value
        elif value.HasField("double_value"):
            result[attr.key] = value.double_value
        elif value.HasField("bool_value"):
            result[attr.key] = value.bool_value

    return result


def point_value(point):
    if point.HasField("as_double"):
        return float(point.as_double)

    if point.HasField("as_int"):
        return float(point.as_int)

    return None


def metric_points(metric):
    if metric.HasField("gauge"):
        return metric.gauge.data_points

    if metric.HasField("sum"):
        return metric.sum.data_points

    return []


def sql_string(value):
    if value is None:
        return ""

    return str(value).replace("'", "''")


def calculate_cpu(node, current):
    """
    current:
      {
        ("cpu0", "idle"): 1234.5,
        ("cpu0", "user"): 20.1,
        ...
      }

    CPU utilization is calculated from delta(non-idle) / delta(all states).
    """

    with cpu_lock:
        previous = previous_cpu.get(node)
        previous_cpu[node] = current.copy()

    if previous is None:
        return None

    total_delta = 0.0
    idle_delta = 0.0

    for key, current_value in current.items():
        if key not in previous:
            continue

        delta = current_value - previous[key]

        # Counter reset/restart.
        if delta < 0:
            continue

        total_delta += delta

        _, state = key

        # Linux CPU idle calculation generally treats idle + wait
        # as non-busy time for host utilization.
        if state in ("idle", "wait"):
            idle_delta += delta

    if total_delta <= 0:
        return None

    busy = total_delta - idle_delta

    cpu_percent = (busy / total_delta) * 100.0

    return max(0.0, min(100.0, cpu_percent))


def write_edgelake(snapshot):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    command = f"""
sql {EDGELAKE_DB} insert into {EDGELAKE_TABLE}
(timestamp,
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
 load_15)
values
('{timestamp}',
 '{sql_string(snapshot["edge_node"])}',
 {snapshot["cpu_percent"]:.4f},
 {int(snapshot["memory_used_bytes"])},
 {int(snapshot["memory_free_bytes"])},
 {int(snapshot["memory_total_bytes"])},
 {snapshot["memory_percent"]:.4f},
 {int(snapshot["disk_used_bytes"])},
 {int(snapshot["disk_free_bytes"])},
 {int(snapshot["disk_total_bytes"])},
 {snapshot["disk_percent"]:.4f},
 {int(snapshot["network_recv_bytes"])},
 {int(snapshot["network_sent_bytes"])},
 {snapshot["load_1"]:.4f},
 {snapshot["load_5"]:.4f},
 {snapshot["load_15"]:.4f})
""".replace("\n", " ").strip()

    response = requests.post(
        EDGELAKE_URL,
        headers={
            "command": command,
            "User-Agent": "curl/8.0.0",
        },
        timeout=5,
    )

    response.raise_for_status()

    print(
        f"EdgeLake INSERT: "
        f"node={snapshot['edge_node']} "
        f"cpu={snapshot['cpu_percent']:.2f}% "
        f"memory={snapshot['memory_percent']:.2f}% "
        f"disk={snapshot['disk_percent']:.2f}% "
        f"rx={snapshot['network_recv_bytes']:.0f} "
        f"tx={snapshot['network_sent_bytes']:.0f}",
        flush=True,
    )


def process_metrics(msg):
    nodes = {}

    for resource_metric in msg.resource_metrics:

        resource_attrs = attrs_to_dict(
            resource_metric.resource.attributes
        )

        resource_node = resource_attrs.get(
            "edge_node",
            "unknown",
        )

        if resource_node not in nodes:
            nodes[resource_node] = {
                "cpu": {},
                "memory": {},
                "filesystem": {},
                "network": {
                    "receive": 0.0,
                    "transmit": 0.0,
                },
                "load_1": None,
                "load_5": None,
                "load_15": None,
            }

        node_data = nodes[resource_node]

        for scope_metric in resource_metric.scope_metrics:

            for metric in scope_metric.metrics:

                name = metric.name

                for point in metric_points(metric):

                    value = point_value(point)

                    if value is None:
                        continue

                    attrs = attrs_to_dict(point.attributes)

                    # CPU cumulative time.
                    if name == "system.cpu.time":
                        cpu = attrs.get("cpu")
                        state = attrs.get("state")

                        if cpu and state:
                            node_data["cpu"][(cpu, state)] = value

                    # Memory.
                    elif name == "system.memory.usage":
                        state = attrs.get("state")

                        if state:
                            node_data["memory"][state] = (
                                node_data["memory"].get(state, 0.0)
                                + value
                            )

                    # Root filesystem only.
                    elif name == "system.filesystem.usage":

                        if attrs.get("mountpoint") != "/":
                            continue

                        state = attrs.get("state")

                        if state:
                            node_data["filesystem"][state] = value

                    # Network totals, excluding loopback.
                    elif name == "system.network.io":

                        device = attrs.get("device")
                        direction = attrs.get("direction")

                        if device == "lo":
                            continue

                        if direction in ("receive", "transmit"):
                            node_data["network"][direction] += value

                    elif name == "system.cpu.load_average.1m":
                        node_data["load_1"] = value

                    elif name == "system.cpu.load_average.5m":
                        node_data["load_5"] = value

                    elif name == "system.cpu.load_average.15m":
                        node_data["load_15"] = value

    snapshots = []

    for node, data in nodes.items():

        if node == "unknown":
            continue

        cpu_percent = calculate_cpu(
            node,
            data["cpu"],
        )

        # First batch establishes the CPU baseline.
        if cpu_percent is None:
            print(
                f"CPU baseline established for {node}; "
                f"waiting for next sample",
                flush=True,
            )
            continue

        memory_used = data["memory"].get("used", 0.0)
        memory_free = data["memory"].get("free", 0.0)

        # Include all memory states in total so total represents
        # the complete host memory reported by OTel.
        memory_total = sum(data["memory"].values())

        memory_percent = (
            (memory_used / memory_total) * 100.0
            if memory_total > 0
            else 0.0
        )

        disk_used = data["filesystem"].get("used", 0.0)
        disk_free = data["filesystem"].get("free", 0.0)
        disk_reserved = data["filesystem"].get("reserved", 0.0)

        disk_total = (
            disk_used
            + disk_free
            + disk_reserved
        )

        disk_percent = (
            (disk_used / disk_total) * 100.0
            if disk_total > 0
            else 0.0
        )

        snapshot = {
            "edge_node": node,
            "cpu_percent": cpu_percent,

            "memory_used_bytes": memory_used,
            "memory_free_bytes": memory_free,
            "memory_total_bytes": memory_total,
            "memory_percent": memory_percent,

            "disk_used_bytes": disk_used,
            "disk_free_bytes": disk_free,
            "disk_total_bytes": disk_total,
            "disk_percent": disk_percent,

            "network_recv_bytes":
                data["network"]["receive"],

            "network_sent_bytes":
                data["network"]["transmit"],

            "load_1": data["load_1"] or 0.0,
            "load_5": data["load_5"] or 0.0,
            "load_15": data["load_15"] or 0.0,
        }

        snapshots.append(snapshot)

    return snapshots


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "otel-edgelake-adapter",
        "edgelake": EDGELAKE_URL,
    }


@app.post("/v1/metrics")
async def receive_metrics(request: Request):
    body = await request.body()

    content_encoding = request.headers.get(
        "content-encoding",
        "",
    )

    if content_encoding.lower() == "gzip":
        try:
            body = gzip.decompress(body)
        except Exception as exc:
            print(
                f"gzip decompression failed: {exc}",
                flush=True,
            )

            return Response(
                content="invalid gzip payload",
                status_code=400,
                media_type="text/plain",
            )

    msg = ExportMetricsServiceRequest()

    try:
        msg.ParseFromString(body)
    except DecodeError as exc:
        print(
            f"OTLP decode failed: {exc}",
            flush=True,
        )

        return Response(
            content="invalid OTLP protobuf payload",
            status_code=400,
            media_type="text/plain",
        )

    try:
        snapshots = process_metrics(msg)

        for snapshot in snapshots:
            write_edgelake(snapshot)

    except Exception as exc:
        print(
            f"Metric processing/EdgeLake error: {exc}",
            flush=True,
        )

        # Return failure so OTel retries rather than silently
        # dropping the telemetry.
        return Response(
            content=str(exc),
            status_code=503,
            media_type="text/plain",
        )

    response = ExportMetricsServiceResponse()

    return Response(
        content=response.SerializeToString(),
        status_code=200,
        media_type="application/x-protobuf",
    )
