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


def process_metrics_request(request):
    """Convert OTLP request into DataFrame and write to Iceberg."""
    rows = []

    for resource_metric in request.resource_metrics:
        try:
            resource_attrs = {attr.key: attr.value.string_value for attr in resource_metric.resource.attributes}
        except Exception:
            resource_attrs = {}
        # host = resource_attrs.get("host") or resource_attrs.get("k8s.pod.name") or "unknown"
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
                            continue  # skip zero values
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
                            continue  # skip zero values
                        rows.append({
                            "host": host,
                            "metric_name": metric_name,
                            "value": val,
                            "timestamp": getattr(dp, "time_unix_nano", int(time.time() * 1e9)),
                        })

                # HISTOGRAM
                if metric.HasField("histogram"):
                    for dp in metric.histogram.data_points:
                        # Prefer sum if available
                        val = getattr(dp, "sum", None)
                        if val is None:
                            val = extract_value_from_datapoint(dp)
                        if val is None or val == 0:
                            continue  # skip zero values
                        rows.append({
                            "host": host,
                            "metric_name": metric_name + ".histogram_sum",
                            "value": val,
                            "timestamp": getattr(dp, "time_unix_nano", int(time.time() * 1e9)),
                        })


    if not rows:
        LOG.debug("No rows extracted from request")
        return

    df = pd.DataFrame(rows)
    table = catalog.load_table(table_identifier)

    try:
        # table.append(df)  # PyIceberg Python append
        # Convert to Arrow
        arrow_table = pa.Table.from_pandas(df)

        # Transaction-safe append
        with table.transaction() as tx:
            tx.append(arrow_table)

        LOG.info("[OTLP] Appended %d rows to Iceberg table '%s'", len(df), table_identifier)

        # LOG.info("[OTLP] Wrote %d rows to Iceberg table %s", len(rows), table_identifier)
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

# -------------------------------------------------------------------
# FastAPI HTTP Metrics Receiver (OTLP/HTTP)
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

def extract_attr(resource_attrs, key):
    """
    Safely return attribute as string, whether stored as:
    - {"k8s.pod.name": "mypod"}  # plain string
    - {"k8s.pod.name": AnyValue(string_value="mypod")}
    """
    if key not in resource_attrs:
        return None

    v = resource_attrs[key]

    # Case 1: Already a string
    if isinstance(v, str):
        return v

    # Case 2: Protobuf AnyValue
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

# Query API
@app.get("/metrics")
async def get_metrics(limit: int = 10):
    table = catalog.load_table(table_identifier)
    # scanner = table.scan().limit(limit)
    # df = scanner.to_arrow().to_pandas()
    # return df.to_dict(orient="records")

    arrow_table = table.scan().to_arrow()

    df = arrow_table.to_pandas()

    # Apply limit manually
    df = df.head(limit)

    return df.to_dict(orient="records")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    t = threading.Thread(target=serve_grpc, daemon=True)
    t.start()

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("HTTP_PORT", 8000)))
