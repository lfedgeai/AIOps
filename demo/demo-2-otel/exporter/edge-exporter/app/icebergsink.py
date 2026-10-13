import pandas as pd
from pyiceberg.catalog import load_catalog
# from minio import Minio
import os

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_USER = os.getenv("MINIO_USER", "minio")
MINIO_PASS = os.getenv("MINIO_PASS", "minio123")
ICEBERG_CATALOG_URI = os.getenv("ICEBERG_CATALOG_URI", MINIO_ENDPOINT)

# Example: Initialize Iceberg catalog
def get_catalog():
    return load_catalog("rest", catalog_uri=ICEBERG_CATALOG_URI)

def write_metrics(df: pd.DataFrame, table_name="otel_metrics"):
    catalog = get_catalog()
    try:
        table = catalog.load_table(table_name)
    except:
        table = catalog.create_table(table_name, schema=df.dtypes.to_dict())
    table.append(df)
