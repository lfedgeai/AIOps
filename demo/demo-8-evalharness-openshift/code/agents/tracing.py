"""
MLflow tracing integration for agent tool-calling loops.

Provides span wrappers for LLM calls and tool executions so each agent run
produces a trace visible in the MLflow Traces UI.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator


def _tracing_enabled() -> bool:
    """Check if MLflow tracing should be active (needs MLFLOW_RUN_ID + tracking URI)."""
    return bool(os.environ.get("MLFLOW_RUN_ID") and os.environ.get("MLFLOW_TRACKING_URI"))


@contextmanager
def trace_llm_call(round_idx: int, model: str, messages: list[dict]) -> Generator[dict, None, None]:
    """Context manager that wraps an LLM API call in an MLflow span."""
    span_data: dict[str, Any] = {}
    if not _tracing_enabled():
        yield span_data
        return
    try:
        import mlflow
        with mlflow.start_span(name=f"llm_round_{round_idx}", span_type="LLM") as span:
            span.set_inputs({"model": model, "message_count": len(messages)})
            yield span_data
            span.set_outputs(span_data)
    except Exception:
        yield span_data


@contextmanager
def trace_tool_call(tool_name: str, arguments: dict) -> Generator[dict, None, None]:
    """Context manager that wraps a tool execution in an MLflow span."""
    span_data: dict[str, Any] = {}
    if not _tracing_enabled():
        yield span_data
        return
    try:
        import mlflow
        with mlflow.start_span(name=f"tool_{tool_name}", span_type="TOOL") as span:
            span.set_inputs({"tool": tool_name, "arguments": {k: str(v)[:200] for k, v in arguments.items()}})
            yield span_data
            if span_data:
                span.set_outputs({"result_preview": str(span_data.get("result_preview", ""))[:500]})
    except Exception:
        yield span_data


def start_agent_trace(agent_name: str, scenario: str, fault_flag: str) -> Any:
    """Start a top-level trace for the entire agent run. Returns the span (or None)."""
    if not _tracing_enabled():
        return None
    try:
        import mlflow
        span = mlflow.start_span(name=f"agent_{agent_name}", span_type="AGENT")
        span.set_inputs({"agent": agent_name, "scenario": scenario, "fault_flag": fault_flag})
        return span
    except Exception:
        return None


def end_agent_trace(span: Any, result: dict[str, Any]) -> None:
    """End the top-level agent trace."""
    if span is None:
        return
    try:
        span.set_outputs({"detected": result.get("detected"), "confidence": result.get("confidence")})
        span.end()
    except Exception:
        pass
