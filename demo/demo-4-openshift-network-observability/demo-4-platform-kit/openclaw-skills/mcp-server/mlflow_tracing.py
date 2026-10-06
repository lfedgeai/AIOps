"""
MLflow Traces audit for NetObserv AIOps (Grafana bridge → LLM guard proxy → MCP).

Replaces Runs-only logging with MLflow Traces (start_span / distributed traceparent).
Gateway @mlflow/mlflow-openclaw remains optional; this path works without it.
"""

from __future__ import annotations

import functools
import json
import os
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Iterator, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_INITIALIZED = False
_ACTIVE_TRACE_LOCK = threading.Lock()
_ACTIVE_TRACE: dict[str, Any] = {}

_TRACE_CONTEXT_URL = os.environ.get(
    "MLFLOW_TRACE_CONTEXT_URL",
    "http://netobserv-grafana-bridge.openclaw.svc.cluster.local:8080/mlflow/trace-context",
)
_TRACE_CONTEXT_TTL_SEC = int(os.environ.get("MLFLOW_TRACE_CONTEXT_TTL_SEC", "3600"))

try:
    from mlflow.tracing import (
        get_tracing_context_headers_for_http_request,
        set_tracing_context_from_http_request_headers,
    )
except ImportError:
    get_tracing_context_headers_for_http_request = None  # type: ignore[misc, assignment]
    set_tracing_context_from_http_request_headers = None  # type: ignore[misc, assignment]


def _apply_kubernetes_auth() -> None:
    """Configure bearer token + workspace for RHOAI MLflow HTTPS."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    auth = os.environ.get("MLFLOW_TRACKING_AUTH", "").strip().lower()
    token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    ns_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    if not os.environ.get("MLFLOW_TRACKING_TOKEN") and os.path.isfile(token_path):
        with open(token_path, encoding="utf-8") as f:
            os.environ["MLFLOW_TRACKING_TOKEN"] = f.read().strip()
    if not os.environ.get("MLFLOW_WORKSPACE") and os.path.isfile(ns_path):
        with open(ns_path, encoding="utf-8") as f:
            os.environ["MLFLOW_WORKSPACE"] = f.read().strip()
    if auth in ("kubernetes", "kubernetes-namespaced") and not os.environ.get("MLFLOW_WORKSPACE"):
        os.environ.setdefault("MLFLOW_WORKSPACE", "openclaw")
    # Bridge/guard-proxy UBI images install mlflow but not mlflow[kubernetes].
    # Projected SA token + workspace is enough for RHOAI HTTPS tracking.
    if os.environ.get("MLFLOW_TRACKING_TOKEN") and auth in ("kubernetes", "kubernetes-namespaced"):
        os.environ.pop("MLFLOW_TRACKING_AUTH", None)
    if uri.startswith("https://") or os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        os.environ.setdefault("MLFLOW_TRACKING_INSECURE_TLS", "true")


def init_mlflow() -> bool:
    global _INITIALIZED
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        return False
    try:
        import mlflow
    except ImportError:
        return False
    if not _INITIALIZED:
        try:
            _apply_kubernetes_auth()
            mlflow.set_tracking_uri(uri)
            workspace = os.environ.get("MLFLOW_WORKSPACE", "").strip()
            if workspace and hasattr(mlflow, "set_workspace"):
                mlflow.set_workspace(workspace)
            exp = os.environ.get("MLFLOW_EXPERIMENT_NAME", "openclaw-netobserv")
            mlflow.set_experiment(exp)
            _INITIALIZED = True
        except Exception:
            return False
    return True


def set_active_incident_trace(
    headers: dict[str, str],
    *,
    idempotency_key: str = "",
    ttl_sec: int | None = None,
) -> None:
    """Publish W3C trace context for downstream LLM/MCP spans (in-process + HTTP fetch)."""
    ttl = ttl_sec if ttl_sec is not None else _TRACE_CONTEXT_TTL_SEC
    with _ACTIVE_TRACE_LOCK:
        _ACTIVE_TRACE.clear()
        _ACTIVE_TRACE.update(
            {
                "headers": {k: v for k, v in headers.items() if k.lower() in ("traceparent", "tracestate")},
                "idempotency_key": idempotency_key,
                "expires_at": time.time() + max(60, ttl),
            }
        )


def get_active_incident_trace_headers() -> dict[str, str]:
    """Return active incident trace headers from in-process store (grafana-bridge)."""
    with _ACTIVE_TRACE_LOCK:
        if not _ACTIVE_TRACE:
            return {}
        if time.time() > float(_ACTIVE_TRACE.get("expires_at") or 0):
            _ACTIVE_TRACE.clear()
            return {}
        return dict(_ACTIVE_TRACE.get("headers") or {})


def active_incident_trace_json() -> dict[str, str]:
    """JSON payload for GET /mlflow/trace-context."""
    headers = get_active_incident_trace_headers()
    with _ACTIVE_TRACE_LOCK:
        key = str(_ACTIVE_TRACE.get("idempotency_key") or "")
    out = dict(headers)
    if key:
        out["idempotency_key"] = key
    return out


def fetch_incident_trace_headers() -> dict[str, str]:
    """Resolve distributed trace context: env → bridge HTTP → in-process."""
    traceparent = os.environ.get("MLFLOW_TRACEPARENT", "").strip()
    if traceparent:
        return {"traceparent": traceparent}

    local = get_active_incident_trace_headers()
    if local:
        return local

    url = os.environ.get("MLFLOW_TRACE_CONTEXT_URL", _TRACE_CONTEXT_URL).strip()
    if not url:
        return {}
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode() or "{}")
        headers = {k: str(v) for k, v in data.items() if k.lower() in ("traceparent", "tracestate") and v}
        return headers
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError, OSError):
        return {}


def _span_set_inputs(span: Any, data: Any) -> None:
    if hasattr(span, "set_inputs"):
        span.set_inputs(data)
    elif hasattr(span, "set_attribute"):
        span.set_attribute("mlflow.spanInputs", json.dumps(data, default=str)[:4000])


def _span_set_outputs(span: Any, data: Any) -> None:
    if hasattr(span, "set_outputs"):
        span.set_outputs(data)
    elif hasattr(span, "set_attribute"):
        span.set_attribute("mlflow.spanOutputs", json.dumps(data, default=str)[:4000])


def _span_set_attributes(span: Any, attrs: dict[str, Any]) -> None:
    if hasattr(span, "set_attributes"):
        span.set_attributes({k: str(v) for k, v in attrs.items()})
        return
    if hasattr(span, "set_attribute"):
        for k, v in attrs.items():
            span.set_attribute(k, str(v))


@contextmanager
def incident_trace_context(headers: dict[str, str] | None = None) -> Iterator[None]:
    """Attach child spans to an active incident trace when traceparent is available."""
    hdrs = headers if headers is not None else fetch_incident_trace_headers()
    if hdrs and set_tracing_context_from_http_request_headers is not None:
        with set_tracing_context_from_http_request_headers(hdrs):
            yield
    else:
        yield


@contextmanager
def trace_span(
    name: str,
    *,
    inputs: Any = None,
    attributes: dict[str, Any] | None = None,
    link_incident: bool = True,
) -> Iterator[Any]:
    """Create an MLflow span, optionally nested under the active Grafana incident trace."""
    if not init_mlflow():
        yield None
        return

    import mlflow

    ctx = incident_trace_context() if link_incident else nullcontext()
    with ctx:
        with mlflow.start_span(name) as span:
            if attributes:
                _span_set_attributes(span, attributes)
            if inputs is not None:
                _span_set_inputs(span, inputs)
            try:
                yield span
            except Exception as exc:
                _span_set_attributes(span, {"error": type(exc).__name__})
                raise


@contextmanager
def incident_root_span(
    name: str,
    *,
    inputs: Any = None,
    attributes: dict[str, Any] | None = None,
    idempotency_key: str = "",
) -> Iterator[dict[str, str]]:
    """
    Root investigate/heal span; publishes traceparent for downstream LLM/MCP spans.
    Keep this context open for the duration of the incident turn.
    """
    if not init_mlflow():
        yield {}
        return

    import mlflow

    with mlflow.start_span(name) as span:
        attrs = {"audit.source": "netobserv-aiops", **(attributes or {})}
        if idempotency_key:
            attrs["incident.idempotency_key"] = idempotency_key
        _span_set_attributes(span, attrs)
        if inputs is not None:
            _span_set_inputs(span, inputs)

        headers: dict[str, str] = {}
        if get_tracing_context_headers_for_http_request is not None:
            try:
                headers = dict(get_tracing_context_headers_for_http_request())
            except Exception:
                headers = {}
        if headers:
            set_active_incident_trace(headers, idempotency_key=idempotency_key)
        yield headers


def _log_tool_run(
    name: str,
    attrs: dict[str, Any],
    duration_ms: float,
    result: Any,
    *,
    error: str | None = None,
) -> None:
    """Mirror each MCP tool call into MLflow Runs (Traces tab uses start_span)."""
    try:
        import mlflow

        with mlflow.start_run(run_name=f"tool.{name}", nested=False, log_system_metrics=False):
            mlflow.set_tags({k: str(v) for k, v in attrs.items()})
            mlflow.log_metric("duration_ms", duration_ms)
            if error:
                mlflow.set_tag("error", error)
            if isinstance(result, str):
                mlflow.log_metric("result_chars", len(result))
                for tag in ("VERDICT", "EVIDENCE_READY", "POLICY_VERDICT", "HEAL_RESULT"):
                    if tag in result:
                        mlflow.set_tag(f"has_{tag.lower()}", "true")
    except Exception:
        pass  # Traces remain primary; Runs mirror is best-effort


def trace_mcp_tool(name: str) -> Callable[[F], F]:
    """Log each MCP tool invocation as an MLflow trace span."""

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not init_mlflow():
                return fn(*args, **kwargs)

            start = time.perf_counter()
            attrs: dict[str, Any] = {
                "audit.source": "netobserv-mcp",
                "tool.name": name,
            }
            if kwargs.get("confirmed") is not None:
                attrs["confirmed"] = str(bool(kwargs["confirmed"]))
            if kwargs.get("namespace"):
                attrs["namespace"] = str(kwargs["namespace"])

            tool_inputs: dict[str, Any] = {}
            if kwargs.get("duration_seconds") is not None:
                tool_inputs["duration_seconds"] = int(kwargs["duration_seconds"])
            if kwargs.get("burst_load") is not None:
                tool_inputs["burst_load"] = bool(kwargs["burst_load"])

            with trace_span(f"tool.{name}", inputs=tool_inputs or None, attributes=attrs) as span:
                try:
                    result = fn(*args, **kwargs)
                    duration_ms = round((time.perf_counter() - start) * 1000, 2)
                    if span is not None:
                        _span_set_attributes(span, {"duration_ms": duration_ms})
                        if isinstance(result, str):
                            _span_set_attributes(span, {"result_chars": len(result)})
                            out_preview = result[:2000] if len(result) <= 2000 else result[:2000] + "…"
                            _span_set_outputs(span, {"preview": out_preview})
                            for tag in (
                                "VERDICT",
                                "EVIDENCE_READY",
                                "POLICY_VERDICT",
                                "HEAL_RESULT",
                            ):
                                if tag in result:
                                    _span_set_attributes(span, {f"has_{tag.lower()}": "true"})
                    _log_tool_run(name, attrs, duration_ms, result)
                    return result
                except Exception as exc:
                    duration_ms = round((time.perf_counter() - start) * 1000, 2)
                    if span is not None:
                        _span_set_attributes(span, {"error": type(exc).__name__})
                    _log_tool_run(name, attrs, duration_ms, None, error=type(exc).__name__)
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator


def trace_llm_chat(
    *,
    model: str,
    user_texts: list[str],
    tool_count: int,
    blocked: bool,
    forward: Callable[[], tuple[int, dict[str, str], bytes]],
) -> tuple[int, dict[str, str], bytes]:
    """Wrap an LLM chat/completions forward with an llm_call span."""
    if not init_mlflow():
        return forward()

    preview = "\n---\n".join(t[:500] for t in user_texts[:3] if t)
    attrs = {
        "audit.source": "netobserv-llm-guard-proxy",
        "llm.model": model or "unknown",
        "llm.tool_count": str(tool_count),
        "llm.blocked": str(blocked).lower(),
    }
    with trace_span("llm_call", inputs={"user_preview": preview}, attributes=attrs) as span:
        start = time.perf_counter()
        status, hdrs, body = forward()
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        if span is not None:
            _span_set_attributes(span, {"duration_ms": duration_ms, "http_status": status})
            if not blocked and body:
                try:
                    data = json.loads(body.decode())
                    usage = data.get("usage") or {}
                    if usage:
                        _span_set_attributes(
                            span,
                            {
                                "tokens.prompt": usage.get("prompt_tokens", 0),
                                "tokens.completion": usage.get("completion_tokens", 0),
                                "tokens.total": usage.get("total_tokens", 0),
                            },
                        )
                    choice = (data.get("choices") or [{}])[0]
                    msg = (choice.get("message") or {}).get("content") or ""
                    if msg:
                        _span_set_outputs(span, {"assistant_preview": msg[:2000]})
                except (json.JSONDecodeError, UnicodeDecodeError, IndexError, TypeError):
                    pass
        return status, hdrs, body
