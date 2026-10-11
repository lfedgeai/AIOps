"""
Context engineering for AIOps harness (C0–C4 corpora, explicit multi-select).

C0: no RAG context tools.
C1–C4: separate RAG corpora, enabled only when listed in CONTEXT_C (e.g. "1,4").
compare_telemetry: standard telemetry tool (not a corpus); gated by tool profile.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_URL = "https://github.com/open-telemetry/opentelemetry-demo.git"
DEFAULT_REPO_DIR = ROOT / "data" / "context_repos" / "opentelemetry-demo"

CONTEXT_TOOLS_BY_C: dict[int, str] = {
    1: "rag_search_source",
    2: "rag_search_docs",
    3: "rag_search_architecture",
    4: "rag_search_dependencies",
}

CORPUS_NAME_BY_C: dict[int, str] = {
    1: "source",
    2: "docs",
    3: "architecture",
    4: "dependencies",
}

# Always offered when the tool profile includes telemetry (not part of CONTEXT_C).
STANDARD_TELEMETRY_TOOLS = ["compare_telemetry"]

from code.tools.rag_context import EmptyCorpusError
from code.tools.rag_context import build_index as build_rag_index
from code.tools.rag_context import retrieve as rag_retrieve

_LAST_CONTEXT_META: dict[str, Any] = {}


def last_context_metadata() -> dict[str, Any]:
    return dict(_LAST_CONTEXT_META)


def parse_context_c(raw: str | None = None) -> frozenset[int]:
    """Parse CONTEXT_C like '1,4' or 'C2' into {1,4} / {2}. Empty or '0' => no corpora."""
    s = (raw if raw is not None else os.environ.get("CONTEXT_C", "0")).strip()
    if not s or s in ("0", "C0", "c0"):
        return frozenset()
    out: set[int] = set()
    for part in s.replace(" ", "").split(","):
        part = part.upper().lstrip("C")
        if not part:
            continue
        try:
            n = int(part)
        except ValueError:
            continue
        if 1 <= n <= 4:
            out.add(n)
    return frozenset(out)


def active_context_c() -> frozenset[int]:
    return parse_context_c()


def context_c_label(corpora: frozenset[int] | None = None) -> str:
    c = active_context_c() if corpora is None else corpora
    if not c:
        return "0"
    return ",".join(str(x) for x in sorted(c))


def context_tool_names(corpora: frozenset[int] | None = None) -> list[str]:
    c = active_context_c() if corpora is None else corpora
    return [CONTEXT_TOOLS_BY_C[n] for n in sorted(c) if n in CONTEXT_TOOLS_BY_C]


def profile_has_telemetry(profile: str) -> bool:
    from code.tools.tool_profiles import TOOL_PROFILES

    tools = set(TOOL_PROFILES.get(profile, []))
    return bool(
        tools
        & {
            "search_logs",
            "search_traces",
            "search_metrics",
            "query_clickhouse",
        }
    )


def repo_path() -> Path:
    explicit = (os.environ.get("CONTEXT_REPO_PATH") or "").strip()
    if explicit:
        return Path(explicit)
    return DEFAULT_REPO_DIR


def ensure_context_repo() -> Path:
    path = repo_path()
    if (path / ".git").is_dir() or (path / "src").is_dir():
        os.environ.setdefault("CONTEXT_REPO_PATH", str(path))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (os.environ.get("CONTEXT_REPO_URL") or DEFAULT_REPO_URL).strip()
    ref = (os.environ.get("CONTEXT_REPO_REF") or "main").strip()
    if path.exists() and not any(path.iterdir()):
        path.rmdir()
    print(f"[context] cloning {url} (ref={ref}) -> {path}", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", ref, url, str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    os.environ["CONTEXT_REPO_PATH"] = str(path)
    return path


def _rag_search_corpus(corpus_id: int, query: str, limit: int = 5) -> str:
    corpus = CORPUS_NAME_BY_C[corpus_id]
    repo = ensure_context_repo()
    build_rag_index(repo, corpus)
    hits = rag_retrieve(repo, query, corpus, limit=limit)
    return json.dumps(
        {
            "corpus": corpus,
            "corpus_id": corpus_id,
            "query": query,
            "hits": [
                {
                    "path": h.get("path"),
                    "chunk_id": h.get("chunk_id"),
                    "score": h.get("score"),
                    "text": h.get("text"),
                }
                for h in hits
            ],
        },
        indent=2,
    )


def rag_search_source(query: str, limit: int = 5) -> str:
    return _rag_search_corpus(1, query, limit)


def rag_search_docs(query: str, limit: int = 5) -> str:
    return _rag_search_corpus(2, query, limit)


def rag_search_architecture(query: str, limit: int = 5) -> str:
    return _rag_search_corpus(3, query, limit)


def rag_search_dependencies(query: str, limit: int = 5) -> str:
    return _rag_search_corpus(4, query, limit)


def build_user_prompt(fault_injection_ts: str) -> str:
    """Harness user prompt — canonical text lives in ``code.agents.prompts``."""
    from code.agents.prompts import build_harness_user_prompt

    rag = context_tool_names() if active_context_c() else None
    return build_harness_user_prompt(fault_injection_ts, rag_tools=rag)

def ensure_context_corpora_ready(
    corpora: frozenset[int] | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """
    Clone repo if needed and build indexes for each enabled corpus.

    Aborts with EmptyCorpusError when any requested corpus has zero chunks —
    empty corpora are invalid for context-engineering evaluation (no soft-fallback).
    """
    c = active_context_c() if corpora is None else corpora
    meta: dict[str, Any] = {
        "context_c": context_c_label(c),
        "context_corpora": sorted(c),
        "indexes": {},
    }
    if not c:
        return meta
    repo = ensure_context_repo()
    meta["repo_path"] = str(repo)
    for cid in sorted(c):
        name = CORPUS_NAME_BY_C[cid]
        try:
            idx_meta = build_rag_index(repo, name, force=force)
        except EmptyCorpusError:
            raise
        except Exception as e:
            raise EmptyCorpusError(
                f"Failed to index corpus C{cid} ({name}) under {repo}: {e}"
            ) from e
        count = int(idx_meta.get("chunk_count") or 0)
        if count <= 0:
            raise EmptyCorpusError(
                f"No indexable chunks for corpus=C{cid} ({name}) under {repo}. "
                "Aborting: empty corpora are invalid for context-engineering evaluation."
            )
        meta["indexes"][str(cid)] = {
            "corpus": name,
            "chunk_count": count,
            "repo_sha": idx_meta.get("repo_sha"),
        }
        print(f"[context] C{cid} ({name}) ready: {count} chunks", flush=True)
    return meta


def apply_context_to_agent_env(
    env: dict[str, str],
    *,
    fault_injection_ts: str,
    scenario_id: str = "a",
    role_id: str | None = None,
    fault_flag: str = "",
    fault_variant: str = "",
) -> dict[str, Any]:
    global _LAST_CONTEXT_META
    corpora = active_context_c()
    label = context_c_label(corpora)
    env["CONTEXT_C"] = label
    env["FAULT_INJECTION_TIME"] = fault_injection_ts
    if role_id:
        env["SCENARIO_ROLE"] = role_id
    # Fail hard on empty corpora (do not warn-and-continue)
    meta = ensure_context_corpora_ready(corpora)
    meta["context_tools"] = context_tool_names(corpora)
    env["AGENT_USER_PROMPT_OVERRIDE"] = build_user_prompt(fault_injection_ts)
    _LAST_CONTEXT_META = meta
    return meta


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _ch_format(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _error_counts_by_service(
    ch_http: str,
    start: datetime,
    end: datetime,
) -> dict[str, int]:
    from code.tools.agent_tools import query_clickhouse

    sql = f"""
    SELECT ServiceName, count() AS cnt
    FROM otel.otel_logs
    WHERE Timestamp >= '{_ch_format(start)}' AND Timestamp < '{_ch_format(end)}'
      AND (
        positionCaseInsensitive(Body, 'error') > 0
        OR positionCaseInsensitive(Body, 'fail') > 0
        OR positionCaseInsensitive(Body, 'exception') > 0
      )
    GROUP BY ServiceName
    ORDER BY cnt DESC
    LIMIT 40
    FORMAT TabSeparated
    """
    out = query_clickhouse(sql, ch_http)
    if out.startswith("Error:") or not out.strip():
        return {}
    counts: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            try:
                counts[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return counts


def build_telemetry_delta(ch_http: str | None, fault_injection_ts: str, minutes: int = 5) -> str:
    ch = ch_http or os.environ.get("CLICKHOUSE_HTTP", "http://127.0.0.1:8123")
    try:
        t0 = _parse_iso(fault_injection_ts)
    except ValueError:
        return f"(invalid fault_injection_ts: {fault_injection_ts})"
    pre_start = t0 - timedelta(minutes=minutes)
    post_end = t0 + timedelta(minutes=minutes)
    pre = _error_counts_by_service(ch, pre_start, t0)
    post = _error_counts_by_service(ch, t0, post_end)
    all_services = sorted(set(pre) | set(post))
    rows: list[dict[str, Any]] = []
    for svc in all_services:
        b, a = pre.get(svc, 0), post.get(svc, 0)
        delta = a - b
        if b > 0 or a > 0:
            rows.append({"service": svc, "pre": b, "post": a, "delta": delta})
    rows.sort(key=lambda r: (-abs(r["delta"]), -r["post"]))
    if not rows:
        return (
            f"No error/fail/exception log counts in windows "
            f"[{pre_start.isoformat()}, {t0.isoformat()}) vs [{t0.isoformat()}, {post_end.isoformat()})."
        )
    lines = [
        f"T0={t0.isoformat()}",
        f"Pre-window: {pre_start.isoformat()} → T0 | Post-window: T0 → {post_end.isoformat()}",
        "Error-like log counts by ServiceName (pre → post, delta):",
    ]
    for r in rows[:15]:
        marker = " **NEW/SPIKE**" if r["delta"] > 0 and r["post"] > r["pre"] else ""
        lines.append(f"- {r['service']}: {r['pre']} → {r['post']} (Δ{r['delta']:+d}){marker}")
    return "\n".join(lines)


def compare_telemetry(
    fault_injection_ts: str,
    minutes: int = 5,
    ch_http: str | None = None,
) -> str:
    text = build_telemetry_delta(ch_http, fault_injection_ts, minutes=minutes)
    return json.dumps({"summary": text}, indent=2)


COMPARE_TELEMETRY_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "compare_telemetry",
        "description": (
            "Compare pre- vs post-injection error-like log counts per service around T0. "
            "Use to see what changed after the fault was injected."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fault_injection_ts": {
                    "type": "string",
                    "description": "ISO8601 T0; defaults to harness fault injection time.",
                },
                "minutes": {"type": "integer", "default": 5, "description": "Window size each side of T0."},
            },
        },
    },
}

CONTEXT_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "rag_search_source",
            "description": "Search application source code (src/**) for implementation and service logic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search_docs",
            "description": "Search product documentation (docs/**, README) for behavior and operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search_architecture",
            "description": "Search architecture diagrams and mermaid/topology docs in the repo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search_dependencies",
            "description": "Search docker-compose and Kubernetes manifests for service dependencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]