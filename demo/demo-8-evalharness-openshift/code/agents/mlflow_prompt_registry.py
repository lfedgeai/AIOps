"""Store exact agent prompts as ordered, reusable MLflow prompt parts."""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any

# The MLflow model-version tag holding a prompt is limited to 8,000 characters.
PART_SIZE = 6000
_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_name(model: str, role: str, part: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", model).strip("-")[:80] or "unknown"
    return f"aiops-{slug}-{_digest(model)[:8]}-{role}-part-{part:03d}"


def _find_part(client: Any, name: str, text: str, digest: str) -> str | None:
    from mlflow.exceptions import MlflowException

    # OSS MLflow excludes prompts from model-version searches. A content alias
    # supports direct lookup without scanning or depending on search results.
    try:
        prompt = client.get_prompt_version_by_alias(name, f"sha256_{digest}")
    except MlflowException as e:
        if e.error_code == "RESOURCE_DOES_NOT_EXIST" or (
            e.error_code == "INVALID_PARAMETER_VALUE" and "not found" in str(e).lower()
        ):
            return None
        raise
    if prompt.template != text:
        raise ValueError(f"Prompt alias collision for {name}")
    return str(prompt.version)


def register_agent_prompts(client: Any, run_id: str, prompts: dict[str, Any]) -> dict[str, Any]:
    """Register system and user text; return an ordered reconstruction manifest.

    Versions are reused only after both the digest and the full text match.
    The manifest, not a version tag, links every run to reused versions.
    """
    model = str(prompts.get("openai_model") or "unknown")
    manifest: dict[str, Any] = {"format": "aiops-prompt-parts-v1", "model": model, "roles": {}}
    for role in ("system", "user"):
        text = prompts.get(f"{role}_prompt")
        if not isinstance(text, str) or not text:
            continue
        chunks = [text[i:i + PART_SIZE] for i in range(0, len(text), PART_SIZE)]
        parts = []
        for index, chunk in enumerate(chunks, 1):
            name = _prompt_name(model, role, index)
            digest = _digest(chunk)
            cache_key = (str(getattr(client, "tracking_uri", "")), os.environ.get("MLFLOW_WORKSPACE", ""), name, digest)
            cached = _cache.get(cache_key)
            version = cached["version"] if cached and cached["text"] == chunk else None
            if version is None:
                version = _find_part(client, name, chunk, digest)
            if version is None:
                created = client.register_prompt(
                    name=name,
                    template=chunk,
                    commit_message=f"Agent {role} prompt, part {index}",
                    tags={
                        "aiops.sha256": digest,
                        "aiops.source_run_id": run_id,
                        "aiops.model": model,
                        "aiops.role": role,
                        "aiops.part": str(index),
                        "aiops.context_c": str(prompts.get("context_c") or "0"),
                        "aiops.tool_profile": str(prompts.get("tool_profile") or ""),
                    },
                )
                version = str(created.version)
                client.set_prompt_alias(name, f"sha256_{digest}", int(version))
            _cache[cache_key] = {"version": version, "text": chunk}
            parts.append({"name": name, "version": version, "sha256": digest})
        manifest["roles"][role] = {"sha256": _digest(text), "parts": parts}
    if not manifest["roles"]:
        raise ValueError("No system or user prompt was supplied")
    return manifest
