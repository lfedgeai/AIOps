"""
Per-round AI metrics: TTFT, token counts, latency, tool execution time.

Usage in agents:
    from code.agents.ai_metrics import RoundMetrics, aggregate_metrics

    rm = RoundMetrics(round_idx=1, model="qwen3-14b")
    rm.mark_request_start()
    response = client.chat.completions.create(...)
    rm.mark_response_end()
    rm.record_usage(response.usage)   # prompt_tokens, completion_tokens
    rm.record_tool_exec("search_logs", elapsed_sec)
    ...
    # After all rounds:
    agg = aggregate_metrics(all_round_metrics)
    # agg is a flat dict ready for mlflow.log_metrics()
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoundMetrics:
    round_idx: int
    model: str

    # Timing
    request_start: float = 0.0
    first_token_time: float = 0.0
    response_end: float = 0.0

    # Token counts (from API usage)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Tool execution
    tool_execs: list[dict[str, Any]] = field(default_factory=list)

    def mark_request_start(self) -> None:
        self.request_start = time.monotonic()

    def mark_first_token(self) -> None:
        if self.first_token_time == 0.0:
            self.first_token_time = time.monotonic()

    def mark_response_end(self) -> None:
        self.response_end = time.monotonic()

    def record_usage(self, usage: Any) -> None:
        if usage is None:
            return
        self.prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.total_tokens = getattr(usage, "total_tokens", 0) or 0

    def record_tool_exec(self, name: str, elapsed_sec: float) -> None:
        self.tool_execs.append({"name": name, "elapsed_sec": elapsed_sec})

    @property
    def ttft_sec(self) -> float | None:
        if self.first_token_time > 0 and self.request_start > 0:
            return self.first_token_time - self.request_start
        return None

    @property
    def llm_latency_sec(self) -> float:
        if self.response_end > 0 and self.request_start > 0:
            return self.response_end - self.request_start
        return 0.0

    @property
    def generation_sec(self) -> float:
        if self.response_end > 0 and self.first_token_time > 0:
            return self.response_end - self.first_token_time
        return 0.0

    @property
    def tokens_per_second(self) -> float:
        gen = self.generation_sec
        if gen > 0 and self.completion_tokens > 0:
            return self.completion_tokens / gen
        return 0.0

    @property
    def total_tool_exec_sec(self) -> float:
        return sum(t["elapsed_sec"] for t in self.tool_execs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_idx,
            "model": self.model,
            "ttft_sec": self.ttft_sec,
            "llm_latency_sec": round(self.llm_latency_sec, 3),
            "generation_sec": round(self.generation_sec, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_second": round(self.tokens_per_second, 1),
            "tool_count": len(self.tool_execs),
            "tool_exec_sec": round(self.total_tool_exec_sec, 3),
            "tool_details": self.tool_execs,
        }


def aggregate_metrics(rounds: list[RoundMetrics]) -> dict[str, float]:
    """Aggregate per-round metrics into a flat dict for mlflow.log_metrics()."""
    if not rounds:
        return {}

    total_prompt = sum(r.prompt_tokens for r in rounds)
    total_completion = sum(r.completion_tokens for r in rounds)
    total_tokens = sum(r.total_tokens for r in rounds)
    total_llm_latency = sum(r.llm_latency_sec for r in rounds)
    total_tool_time = sum(r.total_tool_exec_sec for r in rounds)
    total_tool_calls = sum(len(r.tool_execs) for r in rounds)

    ttfts = [r.ttft_sec for r in rounds if r.ttft_sec is not None]
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else -1.0

    tps_vals = [r.tokens_per_second for r in rounds if r.tokens_per_second > 0]
    avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0.0

    return {
        "ai_rounds": float(len(rounds)),
        "ai_avg_ttft_sec": round(avg_ttft, 3),
        "ai_total_llm_latency_sec": round(total_llm_latency, 3),
        "ai_total_tool_exec_sec": round(total_tool_time, 3),
        "ai_total_tool_calls": float(total_tool_calls),
        "ai_prompt_tokens": float(total_prompt),
        "ai_completion_tokens": float(total_completion),
        "ai_total_tokens": float(total_tokens),
        "ai_avg_tokens_per_sec": round(avg_tps, 1),
    }


def render_ai_metrics_text(rounds: list[RoundMetrics]) -> str:
    """Human-readable text for MLflow artifact."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("AI METRICS")
    lines.append("=" * 60)

    agg = aggregate_metrics(rounds)
    lines.append(f"\nRounds:              {int(agg.get('ai_rounds', 0))}")
    lines.append(f"Avg TTFT:            {agg.get('ai_avg_ttft_sec', -1):.3f}s")
    lines.append(f"Total LLM latency:   {agg.get('ai_total_llm_latency_sec', 0):.3f}s")
    lines.append(f"Total tool exec:     {agg.get('ai_total_tool_exec_sec', 0):.3f}s")
    lines.append(f"Total tool calls:    {int(agg.get('ai_total_tool_calls', 0))}")
    lines.append(f"Prompt tokens:       {int(agg.get('ai_prompt_tokens', 0))}")
    lines.append(f"Completion tokens:   {int(agg.get('ai_completion_tokens', 0))}")
    lines.append(f"Total tokens:        {int(agg.get('ai_total_tokens', 0))}")
    lines.append(f"Avg tokens/sec:      {agg.get('ai_avg_tokens_per_sec', 0):.1f}")

    for rm in rounds:
        d = rm.to_dict()
        lines.append(f"\n{'─' * 40}")
        lines.append(f"Round {d['round']}  (model: {d['model']})")
        lines.append(f"  TTFT:           {d['ttft_sec']:.3f}s" if d['ttft_sec'] is not None else "  TTFT:           N/A (non-streaming)")
        lines.append(f"  LLM latency:    {d['llm_latency_sec']:.3f}s")
        lines.append(f"  Generation:     {d['generation_sec']:.3f}s")
        lines.append(f"  Tokens/sec:     {d['tokens_per_second']:.1f}")
        lines.append(f"  Prompt tokens:  {d['prompt_tokens']}")
        lines.append(f"  Compl. tokens:  {d['completion_tokens']}")
        if d['tool_count'] > 0:
            lines.append(f"  Tool calls:     {d['tool_count']}  ({d['tool_exec_sec']:.3f}s total)")
            for t in d['tool_details']:
                lines.append(f"    - {t['name']}: {t['elapsed_sec']:.3f}s")

    return "\n".join(lines)
