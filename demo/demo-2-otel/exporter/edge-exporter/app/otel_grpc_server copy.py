import os
import grpc
import logging
import threading
from concurrent import futures
from datetime import datetime

import pandas as pd
import pyarrow as pa
import uvicorn
from fastapi import FastAPI

from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2_grpc
from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2

from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import StringType, DoubleType, LongType, NestedField
from pyiceberg.exceptions import NamespaceAlreadyExistsError , NoSuchTableError

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("metrics-server")

# -----------------------------
# Iceberg Catalog Setup
# -----------------------------
catalog = load_catalog(
    "edge",
    **{
        "type": "rest",
        "warehouse": os.getenv("ICEBERG_WAREHOUSE", "s3://workspace"),
        "uri": os.getenv("ICEBERG_REST_URI", "http://iceberg-catalog:8181"),
        "s3.endpoint": os.getenv("S3_ENDPOINT", "http://minio:9000"),
        "s3.access-key-id": os.getenv("S3_ACCESS_KEY", "minioAdmin"),
        "s3.secret-access-key": os.getenv("S3_SECRET_KEY", "minio1234"),
        "s3.path-style-access": os.getenv("S3_PATH_STYLE", "true"),
        "s3.region": os.getenv("S3_REGION", "us-east-1"),
    }
)

# table_identifier = os.getenv("ICEBERG_TABLE_IDENTIFIER", "edge.metrics")

namespace = "edge"
try:
    catalog.create_namespace(namespace)
    logger.info(f"Namespace '{namespace}' created")
except NamespaceAlreadyExistsError:
    logger.info(f"Namespace '{namespace}' already exists")

table_identifier = f"{namespace}.metrics"


try:
    iceberg_table = catalog.load_table(table_identifier)
    logger.info(f"Loaded table {table_identifier}")
except NoSuchTableError:
    # Table doesn't exist → create
    schema = Schema(
        NestedField(id=1, name="host", field_type=StringType(), required=False),
        NestedField(id=2, name="metric_name", field_type=StringType(), required=False),
        NestedField(id=3, name="value", field_type=DoubleType(), required=False),
        NestedField(id=4, name="timestamp", field_type=LongType(), required=False),
    )
    iceberg_table = catalog.create_table(table_identifier, schema)
    logger.info(f"Created table {table_identifier}")
# try:
#     if not catalog.table_exists(table_identifier):
#         schema = Schema(
#             NestedField(id=1, name="host", field_type=StringType(), required=False),
#             NestedField(id=2, name="metric_name", field_type=StringType(), required=False),
#             NestedField(id=3, name="value", field_type=DoubleType(), required=False),
#             NestedField(id=4, name="timestamp", field_type=LongType(), required=False),
#             # Optional human-readable timestamp
#             # NestedField(id=5, name="ts_readable", field_type=StringType(), required=False),
#         )
#         catalog.create_table(table_identifier, schema)
#         logger.info(f"Table '{table_identifier}' created")
# except TableAlreadyExistsError:
#     logger.info(f"Table '{table_identifier}' already created")
#     iceberg_table = catalog.load_table(table_identifier)
# schema = Schema(
#     NestedField(id=1, name="host", field_type=StringType(), required=False),
#     NestedField(id=2, name="metric_name", field_type=StringType(), required=False),
#     NestedField(id=3, name="value", field_type=DoubleType(), required=False),
#     NestedField(id=4, name="timestamp", field_type=LongType(), required=False),
#     # Optional human-readable timestamp
#     # NestedField(id=5, name="ts_readable", field_type=StringType(), required=False),
# )

# global iceberg_table
# try:
#     iceberg_table = catalog.create_table(
#         identifier=table_identifier,
#         schema=schema,
#         properties={"format-version": "2"}
#     )
#     logger.info(f"Table '{table_identifier}' created")
# except TableAlreadyExistsError:
#     logger.info(f"Table '{table_identifier}' already created")
#     iceberg_table = catalog.load_table(table_identifier)


# Thread lock for safe Iceberg writes
table_lock = threading.Lock()

# -----------------------------
# gRPC Metrics Receiver
# -----------------------------
class MetricsReceiver(metrics_service_pb2_grpc.MetricsServiceServicer):
    def Export(self, request, context):
        logger.info("Received OTLP metrics request")
        rows = []

        for resource_metric in request.resource_metrics:
            resource_attrs = {attr.key: attr.value.string_value for attr in resource_metric.resource.attributes}
            host = resource_attrs.get("host", "unknown")

            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    data_points = []
                    if metric.HasField("sum"):
                        data_points = metric.sum.data_points
                    elif metric.HasField("gauge"):
                        data_points = metric.gauge.data_points

                    for dp in data_points:
                        rows.append({
                            "metric_name": metric.name,
                            "value": dp.as_double,
                            "timestamp": dp.time_unix_nano,
                            "host": host,
                            # Optional human-readable timestamp
                            # "ts_readable": pd.to_datetime(dp.time_unix_nano, unit="ns").isoformat(),
                        })

        if rows:
            df = pd.DataFrame(rows)
            logger.info(f"Writing {len(rows)} rows to Iceberg table {table_identifier}")
            table = catalog.load_table(table_identifier)
            arrow_table = pa.Table.from_pandas(df)

            with table_lock:
                with table.new_append() as append:
                    append.append(arrow_table)
                    append.commit()

        return metrics_service_pb2.ExportMetricsServiceResponse()

def serve_grpc():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    metrics_service_pb2_grpc.add_MetricsServiceServicer_to_server(MetricsReceiver(), server)
    server.add_insecure_port("[::]:50051")
    logger.info("gRPC OTLP server listening on port 50051...")
    server.start()
    server.wait_for_termination()

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Metrics API")

@app.get("/metrics")
async def get_metrics(limit: int = 10):
    """Return latest metrics from Iceberg table, sorted by timestamp descending."""
    table = catalog.load_table(table_identifier)
    arrow_table = table.scan().to_arrow()
    df = arrow_table.to_pandas().sort_values("timestamp", ascending=False).head(limit)
    return df.to_dict(orient="records")

# -----------------------------
# Main entry
# -----------------------------
if __name__ == "__main__":
    # Start gRPC server in background thread
    grpc_thread = threading.Thread(target=serve_grpc, daemon=True)
    grpc_thread.start()

    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
