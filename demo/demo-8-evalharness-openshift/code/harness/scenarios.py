"""Load experiment scenario definitions from config/scenarios.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_CONFIG = ROOT / "config" / "scenarios.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def list_scenario_ids() -> list[str]:
    data = _load_yaml(SCENARIOS_CONFIG)
    scenarios = data.get("scenarios") or {}
    return sorted(scenarios.keys())


def get_scenario(scenario_id: str) -> dict[str, Any]:
    data = _load_yaml(SCENARIOS_CONFIG)
    scenarios = data.get("scenarios") or {}
    if scenario_id not in scenarios:
        raise ValueError(
            f"Unknown scenario '{scenario_id}'. Available: {sorted(scenarios.keys())}"
        )
    return dict(scenarios[scenario_id])
