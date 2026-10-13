# otel_exporter_service.py
import os
import grpc
from concurrent import futures
import pandas as pd
import pyarrow as pa
from fastapi import FastAPI, Request, Response
import uvicorn
import threading
import time
import logging

from google.protobuf.json_format import MessageToDict

# OTLP Protos
from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2_grpc
from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2

# Iceberg
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import StringType, DoubleType, LongType, NestedField
from pyiceberg.exceptions import NamespaceAlreadyExistsError

import redis
import json

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "LLM_INFERENCE_QUEUE")

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
LOG = logging.getLogger("edge_exporter")
LOG.setLevel(logging.INFO)
LOG.propagate = True

# -------------------------------------------------------------------
# Iceberg Catalog Setup
# -------------------------------------------------------------------
catalog = load_catalog(
    "edge",
    **{
        "type": "rest",
        "warehouse": os.getenv("ICEBERG_WAREHOUSE", "s3://warehouse"),
        "uri": os.getenv("ICEBERG_REST_URI", "http://iceberg-catalog:8181"),
        "s3.endpoint": os.getenv("S3_ENDPOINT", "http://minio:9000"),
        "s3.access-key-id": os.getenv("S3_ACCESS_KEY", "minioAdmin"),
        "s3.secret-access-key": os.getenv("S3_SECRET_KEY", "minio1234"),
        "s3.path-style-access": os.getenv("S3_PATH_STYLE", "true"),
        "s3.region": os.getenv("S3_REGION", "us-east-1"),
    }
)

namespace = "edge"
try:
    catalog.create_namespace(namespace)
    LOG.info("Created namespace '%s'", namespace)
except NamespaceAlreadyExistsError:
    LOG.debug("Namespace '%s' already exists", namespace)

table_identifier = f"{namespace}.metrics"
if not catalog.table_exists(table_identifier):
    schema = Schema(
        NestedField(id=1, name="host", field_type=StringType()),
        NestedField(id=2, name="metric_name", field_type=StringType()),
        NestedField(id=3, name="value", field_type=DoubleType()),
        NestedField(id=4, name="timestamp", field_type=LongType())
    )
    catalog.create_table(table_identifier, schema)
    LOG.info("Created table '%s'", table_identifier)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def extract_value_from_datapoint(dp):
    """Extract numeric value robustly from OTLP data point."""
    try:
        d = MessageToDict(dp, preserving_proto_field_name=True)
    except Exception:
        for attr in ("as_double", "value", "double_value", "int_value", "as_int"):
            if hasattr(dp, attr):
                try:
                    return float(getattr(dp, attr))
                except Exception:
                    return None
        return None

    for key in ("as_double", "value", "double_value", "int_value", "as_int"):
        if key in d:
            try:
                return float(d[key])
            except Exception:
                continue
    if "sum" in d:
        try:
            return float(d["sum"])
        except Exception:
            return None
    return None

def extract_attr(resource_attrs, key):
    """Safely extract string attribute from resource attributes."""
    if key not in resource_attrs:
        return None
    v = resource_attrs[key]
    if isinstance(v, str):
        return v
    try:
        if hasattr(v, "string_value") and v.string_value:
            return v.string_value
        if hasattr(v, "int_value"):
            return str(v.int_value)
        if hasattr(v, "double_value"):
            return str(v.double_value)
    except Exception:
        pass
    return None

def process_metrics_request(request):
    """Convert OTLP request into DataFrame and write to Iceberg."""
    rows = []

    for resource_metric in request.resource_metrics:
        try:
            resource_attrs = {attr.key: attr.value.string_value for attr in resource_metric.resource.attributes}
        except Exception:
            resource_attrs = {}

        host = (
            extract_attr(resource_attrs, "host")
            or extract_attr(resource_attrs, "service.name")
            or extract_attr(resource_attrs, "k8s.pod.name")
            or extract_attr(resource_attrs, "service.instance.id")
            or "unknown"
        )
        LOG.info(f"Host name: {host}")

        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                metric_name = metric.name

                # GAUGE
                if metric.HasField("gauge"):
                    for dp in metric.gauge.data_points:
                        val = extract_value_from_datapoint(dp)
                        if val is None or val == 0:
                            continue
                        rows.append({
                            "host": host,
                            "metric_name": metric_name,
                            "value": val,
                            "timestamp": getattr(dp, "time_unix_nano", int(time.time() * 1e9)),
                        })


                # SUM
                if metric.HasField("sum"):
                    for dp in metric.sum.data_points:
                        val = extract_value_from_datapoint(dp)
                        if val is None or val == 0:
                            continue
                        rows.append({
                            "host": host,
                            "metric_name": metric_name,
                            "value": val,
                            "timestamp": getattr(dp, "time_unix_nano", int(time.time() * 1e9)),
                        })

                # HISTOGRAM
                if metric.HasField("histogram"):
                    for dp in metric.histogram.data_points:
                        sum_val = getattr(dp, "sum", None)
                        if sum_val is None:
                            sum_val = extract_value_from_datapoint(dp)
                        count_val = getattr(dp, "count", None)
                        if sum_val is None or sum_val == 0:
                            continue

                        rows.append({
                            "host": host,
                            "metric_name": metric_name + ".histogram_sum",
                            "value": sum_val,
                            "timestamp": getattr(dp, "time_unix_nano", int(time.time() * 1e9)),
                        })
                        if count_val:
                            rows.append({
                                "host": host,
                                "metric_name": metric_name + ".histogram_count",
                                "value": count_val,
                                "timestamp": getattr(dp, "time_unix_nano", int(time.time() * 1e9)),
                            })
                        # Optional: add min/max if needed
                        # rows.append({... metric_name + ".histogram_min"})
                        # rows.append({... metric_name + ".histogram_max"})

    if not rows:
        LOG.debug("No rows extracted from request")
        return

    df = pd.DataFrame(rows)
    table = catalog.load_table(table_identifier)

    try:
        arrow_table = pa.Table.from_pandas(df)
        with table.transaction() as tx:
            tx.append(arrow_table)
        LOG.info("[OTLP] Appended %d rows to Iceberg table '%s'", len(df), table_identifier)
    except Exception as e:
        LOG.exception("Failed to append to Iceberg: %s", e)

# -------------------------------------------------------------------
# gRPC Metrics Receiver
# -------------------------------------------------------------------
class GrpcReceiver(metrics_service_pb2_grpc.MetricsServiceServicer):
    def Export(self, request, context):
        LOG.info("Received gRPC OTLP metrics request")
        try:
            process_metrics_request(request)
        except Exception:
            LOG.exception("Error processing gRPC request")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("processing error")
            return metrics_service_pb2.ExportMetricsServiceResponse()
        return metrics_service_pb2.ExportMetricsServiceResponse()

def serve_grpc():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    metrics_service_pb2_grpc.add_MetricsServiceServicer_to_server(GrpcReceiver(), server)
    port = int(os.getenv("GRPC_PORT", 50051))
    server.add_insecure_port(f"[::]:{port}")
    LOG.info("gRPC OTLP server listening on :%d", port)
    server.start()
    server.wait_for_termination()

def push_anomaly_to_redis(row):
    """Convert metric row into anomaly JSON and push into Redis queue."""

    anomaly = {
        "createdtime": (
            row.get("timestamp")  # already in unix ns
        ),
        "clusterName": row.get("clusterName", "unknown"),
        "podName": row.get("podName", row.get("host", "unknown")),
        "servicename": row.get("metric_name"),
        "memory_total_mb": row.get("memory_total_mb"),
        "cpu_cores": row.get("cpu_cores"),
        "value": row.get("value"),
    }

    try:
        redis_client.lpush(REDIS_QUEUE, json.dumps(anomaly))
        LOG.info(f"[REDIS] pushed anomaly → {REDIS_QUEUE}: {anomaly}")
    except Exception as e:
        LOG.error(f"Failed to push anomaly to Redis: {e}")

# -------------------------------------------------------------------
# FastAPI HTTP Metrics Receiver
# -------------------------------------------------------------------
app = FastAPI()

@app.post("/v1/metrics")
async def otlp_http_metrics(request: Request):
    body = await request.body()
    req = metrics_service_pb2.ExportMetricsServiceRequest()
    try:
        req.ParseFromString(body)
    except Exception as e:
        LOG.exception("Failed to parse ExportMetricsServiceRequest: %s", e)
        return Response(status_code=400, content="bad request")

    LOG.info("Received HTTP OTLP metrics request")
    try:
        process_metrics_request(req)
    except Exception:
        LOG.exception("Error processing HTTP OTLP request")
        return Response(status_code=500, content="processing error")

    return {"status": "ok"}

# Query API
@app.get("/metrics")
async def get_metrics(limit: int = 10):
    table = catalog.load_table(table_identifier)
    arrow_table = table.scan().to_arrow()
    df = arrow_table.to_pandas().head(limit)
    return df.to_dict(orient="records")

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    t = threading.Thread(target=serve_grpc, daemon=True)
    t.start()

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("HTTP_PORT", 8000)))
