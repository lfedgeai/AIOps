import json
from pathlib import Path

import pytest

from tier2.scenarios import (ScenarioError, format_list, get_scenario, load_all, parse_scenario,
                             validate_flags)
from tier2.vocab import ACTIONS, DIAGNOSES, SERVICES

VALID = {
    "experiment_id": "X1", "application": "otel-demo-3.1.0", "description": "d", "rationale": "because",
    "fault": {"flag": "cartFailure", "variant": "100%"}, "target_service": "cart",
    "fault_type": "failure", "duration_s": 240, "observation_window_s": 180,
    "baseline_window_s": 180, "workload": {"users": 10},
    "expected": {"incident": True, "affected_service": "cart", "diagnosis": "application_failure",
                 "acceptable_actions": ["investigate"]},
}


def mod(**kw):
    d = json.loads(json.dumps(VALID))
    for k, v in kw.items():
        if "." in k:
            a, b = k.split(".")
            d[a][b] = v
        else:
            d[k] = v
    return d


def test_valid_loads_frozen():
    s = parse_scenario(VALID)
    assert s.fault.flag == "cartFailure" and s.expected.affected_service == "cart"
    with pytest.raises(Exception):
        s.experiment_id = "other"


@pytest.mark.parametrize("data,needle", [
    ({k: v for k, v in VALID.items() if k != "rationale"}, "rationale"),
    (mod(**{"expected.diagnosis": "gremlins"}), "gremlins"),
    (mod(**{"expected.acceptable_actions": ["reboot_the_world"]}), "reboot_the_world"),
    (mod(**{"expected.affected_service": "nope"}), "nope"),
    (mod(target_service="nope"), "nope"),
    (mod(rationale="  "), "rationale is empty"),
    (mod(fault=None), "no fault"),
])
def test_invalid_rejected(data, needle):
    with pytest.raises(ScenarioError, match=needle):
        parse_scenario(data)


def test_missing_expected_field_named():
    d = mod()
    del d["expected"]["diagnosis"]
    with pytest.raises(ScenarioError, match="expected.diagnosis"):
        parse_scenario(d)


def test_control_rules():
    ctrl = mod(fault=None, **{"expected.incident": False, "expected.diagnosis": "none",
                              "expected.affected_service": "none"})
    assert parse_scenario(ctrl).fault is None
    with pytest.raises(ScenarioError, match="control"):
        parse_scenario(mod(**{"expected.incident": False}))


def test_catalogue_loads_and_is_consistent():
    cat = load_all()
    assert "F000" in cat
    c = cat["F000"]
    assert c.fault is None and c.expected.incident is False and c.expected.diagnosis == "none"
    faults = [s for s in cat.values() if s.fault]
    assert len(faults) >= 5
    for s in cat.values():
        assert s.rationale
        assert s.expected.diagnosis in DIAGNOSES
        assert set(s.expected.acceptable_actions) <= set(ACTIONS)
        assert s.expected.affected_service in (*SERVICES, "none")
    # coverage of the reasoning problems named in the spec
    assert {s.expected.diagnosis for s in faults} >= {
        "application_failure", "latency_degradation", "dependency_unreachable",
        "database_contention"}


def test_list_prints_every_id():
    out = format_list()
    for eid in load_all():
        assert eid in out


def test_get_unknown_lists_valid_ids():
    with pytest.raises(ScenarioError, match="F000"):
        get_scenario("F999")


def test_validate_flags(tmp_path: Path):
    from tests.tier2.helpers import FLAGS
    f = tmp_path / "flags.json"
    f.write_text(json.dumps(FLAGS))
    validate_flags(parse_scenario(VALID), f)
    with pytest.raises(ScenarioError, match="nonexistent"):
        validate_flags(parse_scenario(mod(**{"fault.flag": "nonexistent"})), f)
    with pytest.raises(ScenarioError, match="200%"):
        validate_flags(parse_scenario(mod(**{"fault.variant": "200%"})), f)


def test_catalogue_flags_exist_in_pinned_demo():
    flag_file = (Path(__file__).parents[2] / "tier2/testbed/opentelemetry-demo/src/flagd/demo.flagd.json")
    if not flag_file.exists():
        pytest.skip("demo not checked out (run ./setup.sh)")
    for s in load_all().values():
        validate_flags(s, flag_file)
