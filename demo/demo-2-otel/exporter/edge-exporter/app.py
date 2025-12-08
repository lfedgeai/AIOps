import os
from fastapi import FastAPI, Request
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import StringType, DoubleType, TimestampType
import pandas as pd
from datetime import datetime
import uvicorn

app = FastAPI()

# Setup Iceberg catalog
# catalog = load_catalog(
#     "default",
#     {
#         "uri": os.getenv("ICEBERG_CATALOG_URI", "http://minio:9000"),
#         "warehouse": "s3://metrics-warehouse",
#         "s3.endpoint": os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
#         "s3.access-key-id": os.getenv("MINIO_USER", "minio"),
#         "s3.secret-access-key": os.getenv("MINIO_PASS", "minio123"),
#     },
# )

# TABLE_NAME = "default.hostmetrics"

# Create the Iceberg table if not exists
# if not catalog.table_exists(TABLE_NAME):
#     schema = Schema(
#         ("host", StringType()),
#         ("metric_name", StringType()),
#         ("label", StringType()),
#         ("value", DoubleType()),
#         ("ts", TimestampType(with_zone=True)),
#     )
#     catalog.create_table(TABLE_NAME, schema)

@app.post("/ingest")
async def ingest(request: Request):
    body = await request.json()

    # Flatten OTLP ResourceMetrics JSON into rows
    rows = []
    now = datetime.utcnow()
    for resource_metric in body.get("resourceMetrics", []):
        host = ""
        if "resource" in resource_metric and "attributes" in resource_metric["resource"]:
            attrs = {a["key"]: a["value"].get("stringValue", "")
                     for a in resource_metric["resource"]["attributes"]}
            host = attrs.get("host.name", "")
        for scope_metric in resource_metric.get("scopeMetrics", []):
            for metric in scope_metric.get("metrics", []):
                metric_name = metric.get("name")
                if "sum" in metric:
                    for dp in metric["sum"]["dataPoints"]:
                        val = dp.get("asDouble") or dp.get("asInt", 0)
                        labels = dp.get("attributes", [])
                        label_str = ",".join(
                            [f'{a["key"]}={a["value"].get("stringValue","")}' for a in labels])
                        rows.append([host, metric_name, label_str, float(val), now])
                if "gauge" in metric:
                    for dp in metric["gauge"]["dataPoints"]:
                        val = dp.get("asDouble") or dp.get("asInt", 0)
                        labels = dp.get("attributes", [])
                        label_str = ",".join(
                            [f'{a["key"]}={a["value"].get("stringValue","")}' for a in labels])
                        rows.append([host, metric_name, label_str, float(val), now])

    if not rows:
        return {"status": "no metrics"}

    df = pd.DataFrame(rows, columns=["host", "metric_name", "label", "value", "ts"])
    
    print(df)

    # Append to Iceberg
    # table = catalog.load_table(TABLE_NAME)
    # table.append(df)

    return {"status": "ok", "rows_written": len(df)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=50051)
