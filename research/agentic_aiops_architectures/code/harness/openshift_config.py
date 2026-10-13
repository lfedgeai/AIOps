"""Load OpenShift cluster settings from config/openshift.yaml with local/env overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OPENSHIFT_CONFIG = ROOT / "config" / "openshift.yaml"
OPENSHIFT_LOCAL_CONFIG = ROOT / "config" / "openshift.local.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_openshift_config() -> dict[str, Any]:
    """
    Merge config/openshift.yaml + config/openshift.local.yaml (if present).

    Environment overrides (highest precedence):
      OPENSHIFT_CONTEXT  -> cluster.context
      OPENSHIFT_API_URL  -> cluster.api_url
      OPENSHIFT_NAMESPACE -> namespace
    """
    cfg = _load_yaml(OPENSHIFT_CONFIG)
    local = _load_yaml(OPENSHIFT_LOCAL_CONFIG)
    if local:
        cfg = _deep_merge(cfg, local)

    cluster = dict(cfg.get("cluster") or {})
    if os.environ.get("OPENSHIFT_CONTEXT", "").strip():
        cluster["context"] = os.environ["OPENSHIFT_CONTEXT"].strip()
    if os.environ.get("OPENSHIFT_API_URL", "").strip():
        cluster["api_url"] = os.environ["OPENSHIFT_API_URL"].strip()
    cfg["cluster"] = cluster

    if os.environ.get("OPENSHIFT_NAMESPACE", "").strip():
        cfg["namespace"] = os.environ["OPENSHIFT_NAMESPACE"].strip()

    return cfg


def cluster_context(cfg: dict[str, Any] | None = None) -> str:
    return str((cfg or load_openshift_config()).get("cluster", {}).get("context") or "").strip()


def cluster_api_url(cfg: dict[str, Any] | None = None) -> str:
    return str((cfg or load_openshift_config()).get("cluster", {}).get("api_url") or "").strip()


def openshift_namespace(cfg: dict[str, Any] | None = None) -> str:
    return str((cfg or load_openshift_config()).get("namespace") or "agentic-aiops").strip()


if __name__ == "__main__":
    import json

    cfg = load_openshift_config()
    print(json.dumps(cfg, indent=2))
