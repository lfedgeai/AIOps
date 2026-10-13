"""Canonical agent prompts shared by every detector agent.

Matrix comparisons are invalid if agents use different task framing.
All agents MUST build system/user messages from this module (plus
profile overlays from ``tool_profiles.system_prompt_for_profile``).

DeepSeek may append tool-call *syntax* instructions (prompt-based tools),
but the job description, skills, and default user prompt stay identical.

Do not embed application-specific architecture, service names, or product
topology here — agents discover workload identity from tools / optional RAG.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_AGENTS_DIR = Path(__file__).resolve().parent
_SKILLS_DIR = _AGENTS_DIR / "skills"

# Fixed order — identical for every matrix agent.
DEFAULT_SKILL_NAMES: tuple[str, ...] = ("detect", "diagnose", "remediate", "verify")

# Shared task framing — keep identical across all agents / matrix runs.
# Intentionally generic: no product, service, or architecture hints.
_CORE_SYSTEM_PROMPT = """You are an autonomous AIOps agent.
Your job is to DETECT faults, DIAGNOSE root cause, FIX the problem, and REPORT what you did.

Use ONLY the tools listed in your profile. Do not invent tools you cannot call.

Output final JSON (no markdown):
{"detected": true|false, "confidence": 0.0-1.0, "suggested_root_cause": "workload", "suggested_remediations": ["steps taken or recommended"]}

Rules:
- detected=false only when your available signals look normal for the fault window.
- Call log_action to log steps you took or recommend.
- When remediation write tools are available and a fault is found, call them — do not only suggest.
- If detected=true and remediation tools exist, you must call a write tool before final JSON
  (or explicitly state you cannot remediate). The harness enforces this execute gate.
- Infer workload names and topology only from tool results (and optional RAG tools if enabled).
  Do not assume a particular application or service inventory.
"""

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def _skills_enabled() -> bool:
    """AGENT_SKILLS=0/off/false disables; unset or any other value enables (default on)."""
    raw = (os.environ.get("AGENT_SKILLS") or "1").strip().lower()
    return raw not in ("0", "off", "false", "no", "none")


def _skill_names_from_env() -> tuple[str, ...]:
    raw = (os.environ.get("AGENT_SKILL_NAMES") or "").strip()
    if not raw:
        return DEFAULT_SKILL_NAMES
    names = tuple(n.strip() for n in raw.split(",") if n.strip())
    return names or DEFAULT_SKILL_NAMES


def _strip_frontmatter(text: str) -> str:
    return _FRONTMATTER_RE.sub("", text, count=1).strip()


def load_shared_skills(names: tuple[str, ...] | None = None) -> str:
    """Load skill markdown bodies (no YAML frontmatter), joined in fixed order."""
    ordered = names if names is not None else _skill_names_from_env()
    blocks: list[str] = []
    for name in ordered:
        path = _SKILLS_DIR / name / "SKILL.md"
        if not path.is_file():
            raise FileNotFoundError(f"Missing shared AIOps skill: {path}")
        body = _strip_frontmatter(path.read_text(encoding="utf-8"))
        if body:
            blocks.append(body)
    if not blocks:
        return ""
    return "## Shared AIOps skills\n\nFollow these skills in order.\n\n" + "\n\n".join(blocks)


def build_system_prompt_base() -> str:
    """Core framing + shared skills. Identical for all agents."""
    parts = [_CORE_SYSTEM_PROMPT.strip()]
    if _skills_enabled():
        parts.append(load_shared_skills())
    return "\n\n".join(p for p in parts if p).strip()


# Imported by every agent as BASE_SYSTEM_PROMPT — composed once at import.
BASE_SYSTEM_PROMPT = build_system_prompt_base()

DEFAULT_USER_PROMPT_TEMPLATE = (
    "Analyze telemetry since {since_ts}. Use the tools to search logs, traces, "
    "and metrics. Determine if there is a failure, remediate if you can, and "
    "verify the fix before finishing. Respond with JSON when done."
)

# Used by harness via AGENT_USER_PROMPT_OVERRIDE (CONTEXT_C runs).
# Keep wording aligned with DEFAULT_USER_PROMPT_TEMPLATE intent.
HARNESS_USER_PROMPT_TEMPLATE = (
    "Analyze telemetry since {{since_ts}} (fault injection at T0={fault_injection_ts}). "
    "Detect the injected fault, diagnose root cause, fix if you have remediation tools, "
    "and respond with JSON when done."
)


def default_user_prompt(since_ts: str) -> str:
    return DEFAULT_USER_PROMPT_TEMPLATE.format(since_ts=since_ts)


def resolve_user_prompt(since_ts: str) -> str:
    """Return harness override if set, else the shared default user prompt."""
    override = (os.environ.get("AGENT_USER_PROMPT_OVERRIDE") or "").strip()
    if override:
        return override.replace("{since_ts}", since_ts)
    return default_user_prompt(since_ts)


def build_harness_user_prompt(fault_injection_ts: str, *, rag_tools: list[str] | None = None) -> str:
    """Harness-side user prompt (set as AGENT_USER_PROMPT_OVERRIDE)."""
    text = HARNESS_USER_PROMPT_TEMPLATE.format(fault_injection_ts=fault_injection_ts)
    if rag_tools:
        tools = ", ".join(rag_tools)
        text += (
            f" Start with the enabled RAG tool(s) ({tools}) to query the application repo, "
            "then use telemetry and cluster tools."
        )
    return text
