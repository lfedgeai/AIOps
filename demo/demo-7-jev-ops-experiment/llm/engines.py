"""Engine selection for run.py: jev (default), claude-haiku, claude-sonnet, claude-opus, ollama:<model>."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

from jev.client import ConfigError, JevClient, _env_float
from llm.claude import EFFORTS, MODELS, ClaudeAPIEngine, ClaudeCLIEngine
from llm.ollama import DEFAULT_URL as OLLAMA_DEFAULT_URL, OllamaEngine

ENGINE_CHOICES = ("jev", *MODELS, "ollama:<model>")


def parse_engine(value: str) -> str:
    """Validate an --engine value. Raises ValueError with the valid choices."""
    if value == "jev" or value in MODELS:
        return value
    if value.startswith("ollama:") and value[len("ollama:"):].strip():
        return value
    raise ValueError(f"unknown engine {value!r}; valid: {', '.join(ENGINE_CHOICES)}")


def engine_slug(name: str) -> str:
    """Engine name as used in result file names."""
    return name.replace(":", "-").replace("/", "-")


def engine_info(engine) -> dict:
    """Identity of an engine for records and meta. Anything without .info is Jev."""
    info = getattr(engine, "info", None)
    if isinstance(info, dict):
        return dict(info)
    cfg = getattr(engine, "config", None)
    return {"name": "jev", "transport": "jev-api", "model_requested": getattr(cfg, "model", None)}


def fallback_notice(model: str) -> str:
    return (f"ANTHROPIC_API_KEY not set: using the claude CLI (claude -p) for {model}. "
            "Results are labelled transport=claude-cli and are not directly comparable "
            "with API runs.")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _prices(name: str) -> tuple[float | None, float | None]:
    prefix = name.upper().replace("-", "_")
    return (_env_float(f"{prefix}_PRICE_INPUT_PER_MTOK"), _env_float(f"{prefix}_PRICE_OUTPUT_PER_MTOK"))


def _cli_version(claude: str) -> str | None:
    try:
        return subprocess.run([claude, "--version"], capture_output=True, text=True,
                              timeout=15).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_engine(name: str, *, which=shutil.which, err=sys.stderr, prompt=None):
    """Build the engine for a validated name, asking the given prompt variant (the
    Tier 1 baseline if None). Raises ConfigError before any call."""
    _load_dotenv()
    if name == "jev":
        return JevClient(prompt=prompt)
    if name.startswith("ollama:"):
        return OllamaEngine(name[len("ollama:"):], url=os.environ.get("OLLAMA_URL", OLLAMA_DEFAULT_URL),
                            prompt=prompt)

    effort = os.environ.get("LLM_EFFORT", "medium").strip() or "medium"
    if effort not in EFFORTS:
        raise ConfigError(f"LLM_EFFORT must be one of {', '.join(EFFORTS)}, got {effort!r}")
    prices = _prices(name)
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return ClaudeAPIEngine(name, effort=effort, prices=prices, prompt=prompt)

    claude = which("claude")
    if not claude:
        raise ConfigError(
            f"{name} needs either ANTHROPIC_API_KEY (Anthropic API) or the claude CLI on PATH "
            "(fallback). Neither was found.")
    notice = fallback_notice(MODELS[name])
    bar = "*" * 78
    print(f"\n{bar}\n*** {notice}\n{bar}\n", file=err)
    engine = ClaudeCLIEngine(name, effort=effort, claude=claude, prices=prices,
                             cli_version=_cli_version(claude), prompt=prompt)
    engine.info["notice"] = notice
    return engine
