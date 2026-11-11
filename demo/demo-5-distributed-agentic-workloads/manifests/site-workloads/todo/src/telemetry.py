"""OpenTelemetry setup for the demo-5 todo service (traces, metrics, logs)."""

from __future__ import annotations

import logging
import os
import time
from typing import Callable

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from starlette.requests import Request
from starlette.responses import Response

_log = logging.getLogger(__name__)


def _otlp_base() -> str:
    return os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://otel-collector.todo-demo.svc:4318",
    ).rstrip("/")


def _signal_endpoint(signal: str, specific_env: str) -> str:
    """Build OTLP/HTTP URL for traces|metrics|logs."""
    override = os.getenv(specific_env)
    if override:
        return override
    return f"{_otlp_base()}/v1/{signal}"


def _resource() -> Resource:
    service_name = os.getenv("OTEL_SERVICE_NAME", "todo")
    namespace = os.getenv("OTEL_SERVICE_NAMESPACE", "todo-demo")
    return Resource.create(
        {
            "service.name": service_name,
            "service.namespace": namespace,
            "k8s.namespace.name": namespace,
            "deployment.environment": os.getenv("DEPLOYMENT_ENVIRONMENT", "demo"),
            "site_id": os.getenv("SITE_ID", "site-local"),
            "app": "todo",
            "workload_class": "non-agentic",
            "component": "todo",
        }
    )


def setup_telemetry() -> None:
    resource = _resource()

    # --- traces ---
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=_signal_endpoint("traces", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))
        )
    )
    trace.set_tracer_provider(tracer_provider)

    # --- metrics ---
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(
            endpoint=_signal_endpoint("metrics", "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
        ),
        export_interval_millis=int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "15000")),
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # --- logs ---
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(endpoint=_signal_endpoint("logs", "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"))
        )
    )
    set_logger_provider(logger_provider)
    LoggingInstrumentor().instrument(set_logging_format=True)
    root = logging.getLogger()
    if not any(isinstance(h, LoggingHandler) for h in root.handlers):
        root.addHandler(LoggingHandler(level=logging.INFO, logger_provider=logger_provider))
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)

    _log.info(
        "otel configured signals=traces,metrics,logs service=%s site_id=%s endpoint=%s",
        os.getenv("OTEL_SERVICE_NAME", "todo"),
        os.getenv("SITE_ID", "site-local"),
        _otlp_base(),
    )


def instrument_app(app, engine) -> None:
    FastAPIInstrumentor.instrument_app(app)
    SQLAlchemyInstrumentor().instrument(engine=engine)
    Psycopg2Instrumentor().instrument()
    _install_http_metrics(app)


def _install_http_metrics(app) -> None:
    """RED-style HTTP metrics so Phase 2 verify has app metrics without relying on spanmetrics."""
    meter = metrics.get_meter("demo5.todo")
    req_counter = meter.create_counter(
        "http.server.request.count",
        description="HTTP requests",
        unit="{request}",
    )
    req_duration = meter.create_histogram(
        "http.server.request.duration",
        description="HTTP request duration",
        unit="s",
    )
    req_errors = meter.create_counter(
        "http.server.request.errors",
        description="HTTP requests with status >= 500",
        unit="{request}",
    )

    @app.middleware("http")
    async def otel_http_metrics(request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - start
            route = request.url.path
            attrs = {
                "http.method": request.method,
                "http.route": route,
                "http.status_code": status,
            }
            req_counter.add(1, attrs)
            req_duration.record(elapsed, attrs)
            if status >= 500:
                req_errors.add(1, attrs)
