"""Tier 3 scenario model and YAML loader.

Ground truth lives here and in the scenario files only. Nothing in this module
is imported by the observation builder (see tests/tier3/test_observation.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from tier3.vocab import (ACTIONS, APP_NODES, COMPONENTS, DIAGNOSES, FAULT_TYPES, NO_COMPONENT,
                         SERVICES)

SCENARIO_DIR = Path(__file__).parent / "scenarios"


class ScenarioError(ValueError):
    pass


@dataclass(frozen=True)
class FaultSpec:
    """Harness-side only. Never serialized into an observation."""
    type: str
    target: str  # a deployment name, or an application node for node_taint


@dataclass(frozen=True)
class GroundTruth:
    incident: bool
    affected_component: str  # a COMPONENTS entry, or NO_COMPONENT for the control
    diagnosis: str
    acceptable_actions: tuple[str, ...]


@dataclass(frozen=True)
class Scenario:
    experiment_id: str
    application: str
    description: str
    rationale: str
    fault: FaultSpec | None  # None for the healthy control
    target_component: str
    duration_s: int  # single mode: fault held this long before the one decision
    detection_window_s: int  # poll mode: polling stops when this has passed since inject
    poll_interval_s: int
    workload_users: int | None
    observation_window_s: int
    baseline_window_s: int
    expected: GroundTruth


_REQUIRED = ("experiment_id", "application", "description", "rationale", "fault",
             "target_component", "duration_s", "detection_window_s", "poll_interval_s",
             "observation_window_s", "baseline_window_s", "expected")
_ALLOWED = (*_REQUIRED, "workload")
_EXPECTED = ("incident", "affected_component", "diagnosis", "acceptable_actions")


def _check_target(fault_type: str, target: str, source: str) -> None:
    if fault_type == "node_taint":
        if target not in APP_NODES:
            raise ScenarioError(f"{source}: node_taint may only target an application worker "
                                f"({', '.join(APP_NODES)}), not {target!r}")
    elif target not in SERVICES:
        raise ScenarioError(f"{source}: fault target {target!r} is not a demo deployment")


def parse_scenario(data: dict, source: str = "<scenario>") -> Scenario:
    if not isinstance(data, dict):
        raise ScenarioError(f"{source}: not a mapping")
    unknown = sorted(set(data) - set(_ALLOWED))
    if unknown:
        raise ScenarioError(f"{source}: unknown field(s) {', '.join(unknown)}")
    for k in _REQUIRED:
        if k not in data:
            raise ScenarioError(f"{source}: missing field '{k}'")
    exp = data["expected"]
    if not isinstance(exp, dict):
        raise ScenarioError(f"{source}: 'expected' must be a mapping")
    unknown = sorted(set(exp) - set(_EXPECTED))
    if unknown:
        raise ScenarioError(f"{source}: unknown field(s) expected.{', expected.'.join(unknown)}")
    for k in _EXPECTED:
        if k not in exp:
            raise ScenarioError(f"{source}: missing field 'expected.{k}'")
    if not str(data["rationale"] or "").strip():
        raise ScenarioError(f"{source}: rationale is empty")
    if exp["diagnosis"] not in DIAGNOSES:
        raise ScenarioError(f"{source}: expected.diagnosis {exp['diagnosis']!r} not in vocabulary "
                            f"({', '.join(DIAGNOSES)})")
    acts = tuple(exp["acceptable_actions"] or ())
    if not acts:
        raise ScenarioError(f"{source}: expected.acceptable_actions is empty")
    for a in acts:
        if a not in ACTIONS:
            raise ScenarioError(f"{source}: acceptable action {a!r} not in vocabulary "
                                f"({', '.join(ACTIONS)})")
    comp = exp["affected_component"]
    if comp != NO_COMPONENT and comp not in COMPONENTS:
        raise ScenarioError(f"{source}: expected.affected_component {comp!r} is not a known component")
    tgt = data["target_component"]
    if tgt != NO_COMPONENT and tgt not in COMPONENTS:
        raise ScenarioError(f"{source}: target_component {tgt!r} is not a known component")

    fault = None
    if data["fault"]:
        f = data["fault"]
        if not isinstance(f, dict) or set(f) != {"type", "target"}:
            raise ScenarioError(f"{source}: fault needs exactly 'type' and 'target'")
        if f["type"] not in FAULT_TYPES:
            raise ScenarioError(f"{source}: unknown fault type {f['type']!r} "
                                f"(allowed: {', '.join(FAULT_TYPES)})")
        _check_target(f["type"], str(f["target"]), source)
        fault = FaultSpec(str(f["type"]), str(f["target"]))

    incident = exp["incident"]
    if not isinstance(incident, bool):
        raise ScenarioError(f"{source}: expected.incident must be true or false")
    if fault is None:
        if incident:
            raise ScenarioError(f"{source}: expects an incident but defines no fault")
        if exp["diagnosis"] != "none" or comp != NO_COMPONENT:
            raise ScenarioError(f"{source}: control must expect diagnosis 'none' and component 'none'")
    # A fault with incident: false is allowed: task 1 may show a fault leaves the
    # system healthy, and the rationale must then say why (design D7).

    window = int(data["detection_window_s"])
    interval = int(data["poll_interval_s"])
    if interval <= 0 or window < interval:
        raise ScenarioError(f"{source}: need 0 < poll_interval_s <= detection_window_s")
    wl = data.get("workload") or {}
    return Scenario(
        experiment_id=str(data["experiment_id"]),
        application=str(data["application"]),
        description=str(data["description"]),
        rationale=str(data["rationale"]).strip(),
        fault=fault,
        target_component=tgt,
        duration_s=int(data["duration_s"]),
        detection_window_s=window,
        poll_interval_s=interval,
        workload_users=wl.get("users"),
        observation_window_s=int(data["observation_window_s"]),
        baseline_window_s=int(data["baseline_window_s"]),
        expected=GroundTruth(incident=incident, affected_component=comp,
                             diagnosis=exp["diagnosis"], acceptable_actions=acts),
    )


def load_scenario(path: Path) -> Scenario:
    return parse_scenario(yaml.safe_load(Path(path).read_text()), str(path))


def load_all(directory: Path = SCENARIO_DIR) -> dict[str, Scenario]:
    out: dict[str, Scenario] = {}
    for p in sorted(Path(directory).glob("*.yaml")):
        s = load_scenario(p)
        if s.experiment_id in out:
            raise ScenarioError(f"{p}: duplicate experiment_id {s.experiment_id}")
        if p.stem != s.experiment_id:
            raise ScenarioError(f"{p}: file name does not match experiment_id {s.experiment_id}")
        out[s.experiment_id] = s
    return out


def get_scenario(experiment_id: str, directory: Path = SCENARIO_DIR) -> Scenario:
    all_ = load_all(directory)
    if experiment_id not in all_:
        raise ScenarioError(f"unknown experiment {experiment_id!r}; valid: {', '.join(all_)}")
    return all_[experiment_id]


def format_list(directory: Path = SCENARIO_DIR) -> str:
    rows = []
    for s in load_all(directory).values():
        fault = f"{s.fault.type} {s.fault.target}" if s.fault else "none"
        rows.append(f"{s.experiment_id}  {fault:<30} {s.description}")
    return "\n".join(rows)
