"""Tier 2 scenario model and YAML loader.

Ground truth lives here and in the scenario files only. Nothing in this module
is imported by the observation builder (see tests/tier2/test_observation.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from tier2.vocab import ACTIONS, DIAGNOSES, NO_SERVICE, SERVICES

SCENARIO_DIR = Path(__file__).parent / "scenarios"


class ScenarioError(ValueError):
    pass


@dataclass(frozen=True)
class FaultSpec:
    """Harness-side only. Never serialized into an observation."""
    flag: str
    variant: str


@dataclass(frozen=True)
class GroundTruth:
    incident: bool
    affected_service: str  # a SERVICES entry, or NO_SERVICE for the control
    diagnosis: str
    acceptable_actions: tuple[str, ...]


@dataclass(frozen=True)
class Scenario:
    experiment_id: str
    application: str
    description: str
    rationale: str
    fault: FaultSpec | None  # None for the healthy control
    target_service: str
    fault_type: str
    duration_s: int
    workload_users: int | None
    observation_window_s: int
    baseline_window_s: int
    expected: GroundTruth


_REQUIRED = ("experiment_id", "application", "description", "rationale", "fault_type",
             "target_service", "duration_s", "observation_window_s", "baseline_window_s",
             "expected")
_EXPECTED_REQUIRED = ("incident", "affected_service", "diagnosis", "acceptable_actions")


def parse_scenario(data: dict, source: str = "<scenario>") -> Scenario:
    if not isinstance(data, dict):
        raise ScenarioError(f"{source}: not a mapping")
    for k in _REQUIRED:
        if k not in data:
            raise ScenarioError(f"{source}: missing field '{k}'")
    exp = data["expected"]
    if not isinstance(exp, dict):
        raise ScenarioError(f"{source}: 'expected' must be a mapping")
    for k in _EXPECTED_REQUIRED:
        if k not in exp:
            raise ScenarioError(f"{source}: missing field 'expected.{k}'")
    if not str(data["rationale"]).strip():
        raise ScenarioError(f"{source}: rationale is empty")
    if exp["diagnosis"] not in DIAGNOSES:
        raise ScenarioError(f"{source}: expected.diagnosis {exp['diagnosis']!r} not in vocabulary")
    acts = tuple(exp["acceptable_actions"])
    if not acts:
        raise ScenarioError(f"{source}: expected.acceptable_actions is empty")
    for a in acts:
        if a not in ACTIONS:
            raise ScenarioError(f"{source}: acceptable action {a!r} not in vocabulary")
    svc = exp["affected_service"]
    if svc != NO_SERVICE and svc not in SERVICES:
        raise ScenarioError(f"{source}: expected.affected_service {svc!r} is not a known service")
    tgt = data["target_service"]
    if tgt != NO_SERVICE and tgt not in SERVICES:
        raise ScenarioError(f"{source}: target_service {tgt!r} is not a known service")
    incident = bool(exp["incident"])
    fault = None
    if data.get("fault"):
        f = data["fault"]
        if "flag" not in f or "variant" not in f:
            raise ScenarioError(f"{source}: fault needs 'flag' and 'variant'")
        fault = FaultSpec(str(f["flag"]), str(f["variant"]))
    if incident and fault is None:
        raise ScenarioError(f"{source}: expects an incident but defines no fault")
    if not incident:
        if fault is not None:
            raise ScenarioError(f"{source}: control scenario must not define a fault")
        if exp["diagnosis"] != "none" or svc != NO_SERVICE:
            raise ScenarioError(f"{source}: control must expect diagnosis 'none' and service 'none'")
    wl = data.get("workload") or {}
    return Scenario(
        experiment_id=str(data["experiment_id"]),
        application=str(data["application"]),
        description=str(data["description"]),
        rationale=str(data["rationale"]).strip(),
        fault=fault,
        target_service=tgt,
        fault_type=str(data["fault_type"]),
        duration_s=int(data["duration_s"]),
        workload_users=wl.get("users"),
        observation_window_s=int(data["observation_window_s"]),
        baseline_window_s=int(data["baseline_window_s"]),
        expected=GroundTruth(incident=incident, affected_service=svc,
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
        out[s.experiment_id] = s
    return out


def get_scenario(experiment_id: str, directory: Path = SCENARIO_DIR) -> Scenario:
    all_ = load_all(directory)
    if experiment_id not in all_:
        raise ScenarioError(f"unknown experiment {experiment_id!r}; valid: {', '.join(all_)}")
    return all_[experiment_id]


def validate_flags(scenario: Scenario, flag_file: Path) -> None:
    """Check the scenario's flag and variant exist in the pinned demo flag file."""
    if scenario.fault is None:
        return
    flags = json.loads(Path(flag_file).read_text())["flags"]
    f = scenario.fault
    if f.flag not in flags:
        raise ScenarioError(f"{scenario.experiment_id}: flag {f.flag!r} is not in the flag file")
    if f.variant not in flags[f.flag]["variants"]:
        raise ScenarioError(f"{scenario.experiment_id}: variant {f.variant!r} is not defined "
                            f"for flag {f.flag!r}")


def format_list(directory: Path = SCENARIO_DIR) -> str:
    return "\n".join(f"{s.experiment_id}  {s.description}" for s in load_all(directory).values())
