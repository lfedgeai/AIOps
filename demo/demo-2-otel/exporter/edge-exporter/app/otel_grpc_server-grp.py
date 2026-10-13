import os
import grpc
from concurrent import futures
import pandas as pd
import pyarrow as pa
from datetime import datetime

from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2_grpc
from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2

from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import StringType, DoubleType, LongType, NestedField
from pyiceberg.exceptions import NamespaceAlreadyExistsError
import threading
import uvicorn
from fastapi import FastAPI


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
    print(f"Namespace '{namespace}' created")
except NamespaceAlreadyExistsError:
    pass

table_identifier = f"{namespace}.metrics"
if not catalog.table_exists(table_identifier):
    schema = Schema(
        
        NestedField(id=1, name="host", field_type=StringType(), required=False),
        NestedField(id=2, name="metric_name", field_type=StringType(), required=False),
        NestedField(id=3, name="value", field_type=DoubleType(), required=False),
        NestedField(id=4, name="timestamp", field_type=LongType(), required=False),
    )
    catalog.create_table(table_identifier, schema)
    print(f"Table '{table_identifier}' created")

class MetricsReceiver(metrics_service_pb2_grpc.MetricsServiceServicer):
    def Export(self, request, context):
        print("Received OTLP metrics request!")
        rows = []
        # print("Received OTLP metrics request!" ,request.resource_metrics )
        for resource_metric in request.resource_metrics:
            # Extract host/resource info
            resource_attrs = {attr.key: attr.value.string_value for attr in resource_metric.resource.attributes}
            host = resource_attrs.get("host", "unknown")
            # print("Received resource_attrs!",resource_metric )
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    # List of possible fields with data_points
                    metric_fields = ["sum", "gauge", "histogram"]

                    for field_name in metric_fields:
                        if metric.HasField(field_name):
                            data_points = getattr(metric, field_name).data_points
                            for dp in data_points:
                                if dp.as_double > 0:
                                    rows.append({
                                        "metric_name": metric.name,
                                        "value": dp.as_double,
                                        "timestamp": dp.time_unix_nano,
                                        # "timestamp": pd.to_datetime(dp.time_unix_nano, unit="ns"),
                                        "host": host,
                                    })

            
            # for scope_metric in resource_metric.scope_metrics:
            #     for metric in scope_metric.metrics:
            #         # Determine which data_points to use
            #         data_points = []
            #         if metric.HasField("sum"):
            #             data_points = metric.sum.data_points
            #         elif metric.HasField("gauge"):
            #             data_points = metric.gauge.data_points

            #         # Append only positive values
            #         for dp in data_points:
            #             if dp.as_double > 0:
            #                 rows.append({
            #                     "metric_name": metric.name,
            #                     "value": dp.as_double,
            #                     "timestamp": dp.time_unix_nano,
            #                     # "timestamp": pd.to_datetime(dp.time_unix_nano, unit="ns"),
            #                     "host": host,
            #                 })


            # for scope_metric in resource_metric.scope_metrics:
            #     for metric in scope_metric.metrics:
            #         if metric.HasField("sum"):
            #             for dp in metric.sum.data_points:
            #                 if dp.as_double > 0:
            #                     rows.append({
            #                         "metric_name": metric.name,
            #                         "value": dp.as_double,
            #                         "timestamp": dp.time_unix_nano,
            #                         # "timestamp": pd.to_datetime(dp.time_unix_nano, unit="ns"),
            #                         "host": host,
            #                     })
            #         if metric.HasField("gauge"):
            #             for dp in metric.gauge.data_points:
            #                 if dp.as_double > 0:
            #                     rows.append({
            #                         "metric_name": metric.name,
            #                         "value": dp.as_double,
            #                         "timestamp": dp.time_unix_nano,
            #                         # "timestamp": pd.to_datetime(dp.time_unix_nano, unit="ns"),
            #                         "host": host,
            #                     })
        if rows:
            df = pd.DataFrame(rows)
            print(df.head())
            
            # Load Iceberg table
            table = catalog.load_table(table_identifier)
            
            # Convert to PyArrow table (optional, DataFrame also works)
            arrow_table = pa.Table.from_pandas(df)
            
            # Append and commit
            table.append(arrow_table)
            print(f"Wrote {len(rows)} rows to Iceberg table {table_identifier}")
            # with table.new_append() as append:
            #     append.append(arrow_table)
            #     append.commit()  # <-- important, commits data to Iceberg

            # print(f"Wrote {len(rows)} rows to Iceberg table {table_identifier}")
            
        return metrics_service_pb2.ExportMetricsServiceResponse()

def serve_grpc():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    metrics_service_pb2_grpc.add_MetricsServiceServicer_to_server(MetricsReceiver(), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC OTLP server listening on port 50051...")
    server.start()
    server.wait_for_termination()
# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

@app.get("/metrics")
async def get_metrics(limit: int = 10):
    """Return latest metrics from Iceberg table."""
    table = catalog.load_table(table_identifier)
    arrow_table = table.scan().to_arrow()
    df = arrow_table.to_pandas().head(limit)
    return df.to_dict(orient="records")

if __name__ == "__main__":
    # serve()
    # Start gRPC in a background thread
    t = threading.Thread(target=serve_grpc, daemon=True)
    t.start()

    # Start FastAPI (HTTP) on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)