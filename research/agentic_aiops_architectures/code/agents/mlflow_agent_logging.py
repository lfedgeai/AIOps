"""
MLflow artifacts + **Datasets** (Inputs) for LLM agents.

- Artifacts: ``MlflowClient.log_dict`` / ``log_artifact`` → run artifact files.
- Datasets UI: ``MlflowClient.log_inputs`` with :py:class:`mlflow.data.pandas_dataset.PandasDataset`
  so **Prompts** and **Tool calls** appear under the experiment/run **Datasets** / inputs section.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ART_PROMPTS = "agent_llm_prompts.json"
ART_TOOL_CALLS = "agent_llm_tool_calls.json"
ART_ROUNDS = "agent_llm_rounds.json"
ART_THINKING = "agent_llm_thinking.json"
ART_LOG_ERROR = "agent_mlflow_log_error.json"

# MLflow Datasets / inputs: context tag values (shown in UI)
CTX_AGENT_PROMPTS = "agent_prompts"
CTX_AGENT_TOOL_CALLS = "agent_tool_calls"
CTX_AGENT_LLM_ROUNDS = "agent_llm_rounds"
CTX_HARNESS_SUMMARY = "harness_run_summary"


def serialize_messages_for_mlflow(
    messages: list[dict[str, Any]],
    *,
    content_max_chars: int = 16_000,
) -> list[dict[str, Any]]:
    """JSON-safe copy of OpenAI-style messages; truncate long tool/assistant text."""
    out: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.get("role")}
        c = m.get("content")
        if c is not None:
            if isinstance(c, str) and len(c) > content_max_chars:
                c = c[: content_max_chars - 80] + "\n... [truncated for MLflow artifact size]"
            d["content"] = c
        if m.get("tool_calls"):
            d["tool_calls"] = serialize_tool_calls_for_mlflow(m["tool_calls"])
        if m.get("tool_call_id"):
            d["tool_call_id"] = m["tool_call_id"]
        out.append(d)
    return out


def serialize_tool_calls_for_mlflow(tool_calls: Any) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            serialized.append(
                {
                    "id": tc.get("id"),
                    "type": tc.get("type"),
                    "function": {
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments"),
                    },
                }
            )
        else:
            fn = getattr(tc, "function", None)
            serialized.append(
                {
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", None),
                    "function": {
                        "name": getattr(fn, "name", None) if fn else None,
                        "arguments": getattr(fn, "arguments", None) if fn else None,
                    },
                }
            )
    return serialized


def _ensure_mlflow_tls_env() -> None:
    """Honor MLFLOW_TRACKING_INSECURE_TLS for HTTPS tracking (e.g. OpenShift routes)."""
    v = (os.environ.get("MLFLOW_TRACKING_INSECURE_TLS") or "").strip().lower()
    if v in ("1", "true", "yes"):
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"


def _client_log_dict_or_artifact(client: Any, run_id: str, data: dict[str, Any], artifact_path: str) -> None:
    """Upload JSON using log_dict when available, else tempfile + log_artifact."""
    if hasattr(client, "log_dict"):
        client.log_dict(run_id, data, artifact_path)
        return
    td = tempfile.mkdtemp()
    try:
        name = Path(artifact_path).name
        parent = str(Path(artifact_path).parent)
        apath = None if parent in (".", "") else parent
        fp = Path(td) / name
        fp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        client.log_artifact(run_id, str(fp), artifact_path=apath)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _prompts_to_dataframe(prompts: dict[str, Any], *, max_cell: int = 8_000) -> Any:
    """One row per conversation message so prompts are readable in the MLflow UI."""
    import pandas as pd

    model = str(prompts.get("openai_model") or "")
    api_base = str(prompts.get("openai_api_base") or "")
    tools = ", ".join(prompts.get("registered_tool_names", []))

    rows: list[dict[str, Any]] = []
    rows.append({
        "order": 0,
        "role": "config",
        "content": (
            f"model: {model}\n"
            f"api_base: {api_base}\n"
            f"since_ts: {prompts.get('since_ts', '')}\n"
            f"tools: {tools}"
        ),
    })

    conv = prompts.get("conversation_messages") or []
    for i, msg in enumerate(conv):
        role = msg.get("role", "unknown")
        text_parts: list[str] = []

        content = msg.get("content")
        if content:
            text_parts.append(str(content)[:max_cell])

        tc_list = msg.get("tool_calls") or []
        for tc in tc_list:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                text_parts.append(
                    f"[tool_call] {fn.get('name', '?')}({fn.get('arguments', '')})"
                )

        tcid = msg.get("tool_call_id")
        if tcid:
            role = f"tool (call_id={tcid})"

        rows.append({
            "order": i + 1,
            "role": role,
            "content": "\n".join(text_parts)[:max_cell] if text_parts else "",
        })

    if not conv:
        rows.append({
            "order": 1,
            "role": "system",
            "content": str(prompts.get("system_prompt", ""))[:max_cell],
        })
        rows.append({
            "order": 2,
            "role": "user",
            "content": str(prompts.get("user_prompt", ""))[:max_cell],
        })

    return pd.DataFrame(rows)


def _tool_calls_to_dataframe(tool_calls: list[dict[str, Any]], *, max_cell: int = 8_000) -> Any:
    """One row per tool call with readable argument formatting."""
    import pandas as pd

    if not tool_calls:
        return pd.DataFrame([{"order": 0, "tool_name": "(none)", "arguments": "", "result": ""}])

    rows = []
    for i, tc in enumerate(tool_calls):
        args = tc.get("arguments", {})
        if isinstance(args, dict):
            arg_lines = [f"  {k} = {json.dumps(v, default=str)}" for k, v in args.items()]
            args_text = "\n".join(arg_lines)
        else:
            args_text = str(args)

        result = str(tc.get("result_preview", ""))
        if len(result) > max_cell:
            result = result[:max_cell - 40] + "\n... [truncated]"

        rows.append({
            "order": i + 1,
            "tool_name": str(tc.get("name", "")),
            "arguments": args_text[:max_cell],
            "result": result,
        })
    return pd.DataFrame(rows)


def _llm_rounds_to_dataframe(llm_rounds: list[dict[str, Any]], *, max_cell: int = 8_000) -> Any:
    """One row per LLM round with readable fields instead of a JSON blob."""
    import pandas as pd

    if not llm_rounds:
        return pd.DataFrame([{"round": 0, "model": "", "summary": "(no rounds)"}])

    rows = []
    for r in llm_rounds:
        parts: list[str] = []

        if r.get("api_error"):
            parts.append(f"API ERROR: {r['api_error']}")

        assistant = r.get("assistant_content", "")
        if assistant:
            preview = str(assistant)[:2000]
            parts.append(f"Assistant:\n{preview}")

        native_tcs = r.get("native_tool_calls_from_api") or r.get("parsed_tool_calls") or []
        if native_tcs:
            for tc in native_tcs:
                fn = tc.get("function", tc) if "function" in tc else tc
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                parts.append(f"-> {name}({args})")

        execs = r.get("tool_executions") or []
        for ex in execs:
            res = str(ex.get("result_preview", ""))[:1500]
            parts.append(f"<- {ex.get('name', '?')}: {res}")

        note = r.get("note", "")
        if note:
            parts.append(f"[{note}]")

        rows.append({
            "round": r.get("round", 0),
            "model": str(r.get("model", "")),
            "summary": "\n".join(parts)[:max_cell] if parts else "(empty round)",
        })
    return pd.DataFrame(rows)


def log_mlflow_dataset_inputs_for_agent_trace(
    client: Any,
    run_id: str,
    *,
    prompts: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    llm_rounds: list[dict[str, Any]] | None,
) -> None:
    """
    Register prompts, tool calls, and (optional) LLM rounds as **run inputs / datasets**
    so they show in the MLflow **Datasets** (inputs) UI for the experiment.

    Uses ``MlflowClient.log_inputs`` — does **not** start or end runs.
    """
    try:
        import mlflow.data
        import pandas as pd
        from mlflow.entities.dataset_input import DatasetInput
        from mlflow.entities.input_tag import InputTag
        from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT
    except ImportError as e:
        print(f"[agent] MLflow datasets: skip (missing dependency: {e})", file=sys.stderr, flush=True)
        return

    rid = (run_id or "").strip()
    if not rid or not hasattr(client, "log_inputs"):
        return

    td = tempfile.mkdtemp()
    try:
        inputs: list[Any] = []

        df_p = _prompts_to_dataframe(prompts)
        p_src = str((Path(td) / "agent_prompts.csv").resolve())
        df_p.to_csv(p_src, index=False)
        ds_p = mlflow.data.from_pandas(
            df_p,
            source=p_src,
            name="agent_llm_prompts",
        )
        inputs.append(
            DatasetInput(
                dataset=ds_p._to_mlflow_entity(),
                tags=[InputTag(key=MLFLOW_DATASET_CONTEXT, value=CTX_AGENT_PROMPTS)],
            )
        )

        df_t = _tool_calls_to_dataframe(tool_calls)
        t_src = str((Path(td) / "agent_tool_calls.csv").resolve())
        df_t.to_csv(t_src, index=False)
        ds_t = mlflow.data.from_pandas(
            df_t,
            source=t_src,
            name="agent_llm_tool_calls",
        )
        inputs.append(
            DatasetInput(
                dataset=ds_t._to_mlflow_entity(),
                tags=[InputTag(key=MLFLOW_DATASET_CONTEXT, value=CTX_AGENT_TOOL_CALLS)],
            )
        )

        if llm_rounds is not None:
            df_r = _llm_rounds_to_dataframe(llm_rounds)
            r_src = str((Path(td) / "agent_llm_rounds.csv").resolve())
            df_r.to_csv(r_src, index=False)
            ds_r = mlflow.data.from_pandas(
                df_r,
                source=r_src,
                name="agent_llm_rounds",
            )
            inputs.append(
                DatasetInput(
                    dataset=ds_r._to_mlflow_entity(),
                    tags=[InputTag(key=MLFLOW_DATASET_CONTEXT, value=CTX_AGENT_LLM_ROUNDS)],
                )
            )

        client.log_inputs(run_id=rid, datasets=inputs)
        print(
            f"[agent] MLflow Datasets UI: log_inputs ({CTX_AGENT_PROMPTS}, {CTX_AGENT_TOOL_CALLS}"
            + (f", {CTX_AGENT_LLM_ROUNDS}" if llm_rounds is not None else "")
            + f") run={rid[:8]}…",
            file=sys.stderr,
            flush=True,
        )
    except Exception as e:
        print(f"[agent] MLflow log_inputs (datasets) failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def log_mlflow_dataset_inputs_for_harness_summary(client: Any, run_id: str, summary: dict[str, Any]) -> None:
    """Single-row dataset for harness outcome (Datasets / inputs section)."""
    try:
        import mlflow.data
        from mlflow.entities.dataset_input import DatasetInput
        from mlflow.entities.input_tag import InputTag
        from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT
    except ImportError as e:
        print(f"[harness] MLflow datasets: skip ({e})", flush=True)
        return

    rid = (run_id or "").strip()
    if not rid or not hasattr(client, "log_inputs"):
        return

    td = tempfile.mkdtemp()
    try:
        import pandas as pd

        flat = {k: json.dumps(v, default=str) if isinstance(v, (list, dict)) else str(v) for k, v in summary.items()}
        df = pd.DataFrame([flat])
        src = str((Path(td) / "harness_run_summary.csv").resolve())
        df.to_csv(src, index=False)
        ds = mlflow.data.from_pandas(df, source=src, name="harness_run_summary")
        inp = DatasetInput(
            dataset=ds._to_mlflow_entity(),
            tags=[InputTag(key=MLFLOW_DATASET_CONTEXT, value=CTX_HARNESS_SUMMARY)],
        )
        client.log_inputs(run_id=rid, datasets=[inp])
        print(f"[harness] MLflow Datasets UI: log_inputs ({CTX_HARNESS_SUMMARY}) run={rid[:8]}…", flush=True)
    except Exception as e:
        print(f"[harness] MLflow log_inputs failed: {type(e).__name__}: {e}", flush=True)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _render_prompts_text(prompts: dict[str, Any]) -> str:
    """Render prompts/conversation as a readable chat transcript."""
    lines: list[str] = []
    model = prompts.get("openai_model") or ""
    api_base = prompts.get("openai_api_base") or ""
    tools = prompts.get("registered_tool_names", [])
    lines.append("=" * 70)
    lines.append("AGENT CONFIGURATION")
    lines.append("=" * 70)
    lines.append(f"Model:    {model}")
    lines.append(f"API Base: {api_base}")
    lines.append(f"Since:    {prompts.get('since_ts', '')}")
    lines.append(f"Tools:    {', '.join(tools)}")
    lines.append("")

    conv = prompts.get("conversation_messages") or []
    if conv:
        lines.append("=" * 70)
        lines.append("CONVERSATION")
        lines.append("=" * 70)
        for i, msg in enumerate(conv):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content")
            tcid = msg.get("tool_call_id")

            if tcid:
                lines.append(f"\n--- [{i+1}] TOOL RESULT (call_id={tcid}) ---")
                if content:
                    lines.append(str(content)[:20_000])
            else:
                lines.append(f"\n--- [{i+1}] {role} ---")
                if content:
                    lines.append(str(content)[:20_000])
                tc_list = msg.get("tool_calls") or []
                for tc in tc_list:
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        lines.append(f"  >> TOOL CALL: {fn.get('name', '?')}({fn.get('arguments', '')})")
    else:
        lines.append("=" * 70)
        lines.append("PROMPTS (no conversation logged yet)")
        lines.append("=" * 70)
        lines.append(f"\n--- SYSTEM ---\n{prompts.get('system_prompt', '')[:20_000]}")
        lines.append(f"\n--- USER ---\n{prompts.get('user_prompt', '')[:20_000]}")

    return "\n".join(lines)


def _render_tool_calls_text(tool_calls: list[dict[str, Any]]) -> str:
    """Render tool calls as a readable list."""
    if not tool_calls:
        return "(no tool calls recorded)"
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"TOOL CALLS ({len(tool_calls)} total)")
    lines.append("=" * 70)
    for i, tc in enumerate(tool_calls):
        name = tc.get("name", "?")
        args = tc.get("arguments", {})
        result = str(tc.get("result_preview", ""))

        lines.append(f"\n--- Tool Call #{i+1}: {name} ---")
        if isinstance(args, dict):
            for k, v in args.items():
                lines.append(f"  {k} = {json.dumps(v, default=str)}")
        else:
            lines.append(f"  arguments: {args}")
        lines.append(f"\n  Result ({len(result)} chars):")
        lines.append(f"  {result[:8000]}")
    return "\n".join(lines)


def _render_llm_rounds_text(llm_rounds: list[dict[str, Any]]) -> str:
    """Render LLM rounds as a readable summary."""
    if not llm_rounds:
        return "(no LLM rounds recorded)"
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"LLM ROUNDS ({len(llm_rounds)} total)")
    lines.append("=" * 70)
    for r in llm_rounds:
        rnum = r.get("round", "?")
        model = r.get("model", "")
        lines.append(f"\n{'─' * 50}")
        lines.append(f"Round {rnum}  (model: {model})")
        lines.append(f"{'─' * 50}")

        if r.get("api_error"):
            lines.append(f"  API ERROR: {r['api_error']}")

        assistant = r.get("assistant_content", "")
        if assistant:
            lines.append(f"\n  Assistant response ({len(assistant)} chars):")
            lines.append(f"  {str(assistant)[:4000]}")

        for tc in r.get("native_tool_calls_from_api") or r.get("parsed_tool_calls") or []:
            fn = tc.get("function", tc) if "function" in tc else tc
            name = fn.get("name", "?")
            args = fn.get("arguments", "")
            lines.append(f"\n  >> TOOL CALL: {name}({args})")

        for ex in r.get("tool_executions") or []:
            res = str(ex.get("result_preview", ""))[:3000]
            lines.append(f"  << TOOL RESULT ({ex.get('name', '?')}): {res}")

        note = r.get("note", "")
        if note:
            lines.append(f"  [{note}]")

    return "\n".join(lines)


def _log_text_artifact(client: Any, run_id: str, text: str, artifact_path: str) -> None:
    """Upload a plain-text artifact."""
    td = tempfile.mkdtemp()
    try:
        name = Path(artifact_path).name
        parent = str(Path(artifact_path).parent)
        apath = None if parent in (".", "") else parent
        fp = Path(td) / name
        fp.write_text(text, encoding="utf-8")
        client.log_artifact(run_id, str(fp), artifact_path=apath)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def log_agent_trace_to_mlflow(
    *,
    run_id: str,
    tracking_uri: str,
    prompts: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    thinking: list[dict[str, Any]] | None,
    llm_rounds: list[dict[str, Any]] | None,
    artifact_subdir: str = "",
) -> None:
    """
    Log prompts, tool_calls, optional thinking, and llm_rounds as:
    - JSON artifacts (machine-readable)
    - Plain-text artifacts (human-readable in MLflow Artifacts viewer)
    - Dataset inputs (schema in MLflow Datasets UI)
    """
    from mlflow.tracking import MlflowClient

    _ensure_mlflow_tls_env()
    uri = (tracking_uri or "").strip()
    rid = (run_id or "").strip()
    if not uri or not rid:
        return

    client = MlflowClient(uri)
    sub = (artifact_subdir or "").strip().strip("/")

    def path(leaf: str, root_name: str) -> str:
        if sub:
            return f"{sub}/{leaf}"
        return root_name

    _client_log_dict_or_artifact(client, rid, prompts, path("prompts.json", ART_PROMPTS))
    _client_log_dict_or_artifact(
        client, rid, {"tool_calls": tool_calls}, path("tool_calls.json", ART_TOOL_CALLS)
    )
    if thinking:
        _client_log_dict_or_artifact(
            client, rid, {"thinking": thinking}, path("agent_thinking.json", ART_THINKING)
        )
    if llm_rounds is not None:
        _client_log_dict_or_artifact(
            client, rid, {"rounds": llm_rounds}, path("llm_rounds.json", ART_ROUNDS)
        )

    try:
        _log_text_artifact(client, rid, _render_prompts_text(prompts), "agent_prompts.txt")
        _log_text_artifact(client, rid, _render_tool_calls_text(tool_calls), "agent_tool_calls.txt")
        if llm_rounds is not None:
            _log_text_artifact(client, rid, _render_llm_rounds_text(llm_rounds), "agent_llm_rounds.txt")
    except Exception as e:
        print(f"[agent] MLflow text artifacts failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    log_mlflow_dataset_inputs_for_agent_trace(
        client,
        rid,
        prompts=prompts,
        tool_calls=tool_calls,
        llm_rounds=llm_rounds,
    )

    print(
        f"[agent] MLflow: wrote artifacts + text (prompts, tool_calls"
        + (", rounds" if llm_rounds is not None else "")
        + f") run {rid[:8]}…",
        file=sys.stderr,
        flush=True,
    )


def log_agent_trace_error_to_mlflow(
    *,
    run_id: str,
    tracking_uri: str,
    error_payload: dict[str, Any],
) -> None:
    """Small error artifact when main trace logging fails."""
    from mlflow.tracking import MlflowClient

    _ensure_mlflow_tls_env()
    uri = (tracking_uri or "").strip()
    rid = (run_id or "").strip()
    if not uri or not rid:
        return
    client = MlflowClient(uri)
    _client_log_dict_or_artifact(client, rid, error_payload, ART_LOG_ERROR)
