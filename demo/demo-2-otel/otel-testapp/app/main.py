# app/main.py
import asyncio
import math
import os
import time
from typing import Optional

import socket
import requests


from fastapi import FastAPI, Query
from pydantic import BaseModel
from opentelemetry import metrics, trace

# OTEL setup (tracing + metrics)
# from .otel_setup import configure_otel

# configure_otel(service_name=os.getenv("SERVICE_NAME", "otel-loadgen"))


# from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
# from opentelemetry.runtime.metrics import RuntimeMetrics
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor

# --------------------------
# Setup metrics
# --------------------------
exporter = OTLPMetricExporter(
    endpoint="http://otel-aggregator:4318/v1/metrics", 
    headers={}
)
# reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5000)
# provider = MeterProvider(metric_readers=[reader])
# metrics.set_meter_provider(provider)
meter = metrics.get_meter("otel-testapp-meter")
# RuntimeMetrics.init()
reader = PeriodicExportingMetricReader(exporter,export_interval_millis=5000)
provider = MeterProvider(metric_readers=[reader])

# Activate it
SystemMetricsInstrumentor().instrument(
    meter_provider=provider,
    track_cpu=True,
    track_memory=True
)

# Create manual test counter
manual_counter = meter.create_counter(
    "debug_manual_counter",
    description="Manual test counter"
)


app = FastAPI(title="OTEL Load Generator", version="1.0")




# Create Meter and Tracer manually (auto-instrumentation will still activate exporters)
meter = metrics.get_meter(__name__)
tracer = trace.get_tracer(__name__)

# A test metric
counter = meter.create_counter("debug_manual_counter")


def check_collector(endpoint: str):
    try:
        r = requests.get(endpoint, timeout=2)
        return {
            "reachable": True,
            "status_code": r.status_code,
            "response": r.text[:200],
        }
    except Exception as e:
        return {
            "reachable": False,
            "error": str(e),
        }


@app.get("/otel/test_export")
def test_export():
    manual_counter.add(1)
    return {"status": "metric emitted"}


@app.get("/")
def root():
    return {"message": "OTEL Debug Container Running"}


@app.get("/otel/status")
def otel_status():
    otel_env = {k: v for k, v in os.environ.items() if k.startswith("OTEL")}

    return {
        "hostname": socket.gethostname(),
        "otel_env": otel_env,
        "collector_check_http": check_collector(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "") + "/v1/metrics"),
        "collector_check_grpc": os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
        "note": "If HTTP check returns 405, it means collector is reachable (GOOD).",
        "manual_metric_status": "Ready",
    }


@app.get("/otel/test_export")
def otel_test_export():
    # Create a trace span
    with tracer.start_as_current_span("debug-test-span"):
        counter.add(1)

    return {
        "message": "Test metric + trace exported (check collector logs).",
        "timestamp": time.time(),
    }


# Simple in-memory heap for memory bursts
heap = []

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/cpu")
async def cpu(seconds: Optional[int] = Query(5, ge=1, le=600)):
    """
    Run CPU busy loop for `seconds` seconds.
    """
    end = time.time() + seconds

    def busy():
        s = 0.0
        # small chunks so we yield occasionally
        for i in range(100000):
            s += math.sqrt(i)
        return s

    # run asynchronously to avoid blocking the event loop fully
    loop = asyncio.get_running_loop()
    # run in background threads until time expires
    while time.time() < end:
        await loop.run_in_executor(None, busy)
    return {"cpu_seconds": seconds}

@app.get("/memory")
async def memory(mb: Optional[int] = Query(100, ge=1, le=2000), hold_seconds: Optional[int] = Query(10, ge=1, le=600)):
    """
    Allocate `mb` megabytes for `hold_seconds` seconds, then free.
    """
    # each int ~28 bytes on CPython, but lists of ints are heavy; use bytearray for predictable size
    bytes_to_alloc = mb * 1024 * 1024
    heap.append(bytearray(bytes_to_alloc))
    await asyncio.sleep(hold_seconds)
    # free the last allocation
    try:
        heap.pop()
    except Exception:
        pass
    return {"allocated_mb": mb, "holding_seconds": hold_seconds}

@app.get("/mix")
async def mix(seconds: Optional[int] = Query(10, ge=1, le=600), mb: Optional[int] = Query(50, ge=1, le=2000)):
    """
    Simultaneously run CPU busy loop and allocate memory for `seconds`.
    """
    # schedule memory allocation
    mem_task = asyncio.create_task(memory(mb=mb, hold_seconds=seconds))
    cpu_task = asyncio.create_task(cpu(seconds=seconds))
    res = await asyncio.gather(mem_task, cpu_task)
    return {"mix": res}

@app.get("/info")
def info():
    return {
        "pod_cpu_limit": os.getenv("CPU_LIMIT"),
        "pod_memory_limit": os.getenv("MEMORY_LIMIT"),
        "node_total_cpus": os.getenv("NODE_CPU_TOTAL"),
        "node_total_memory_bytes": os.getenv("NODE_MEM_TOTAL_BYTES")
    }
