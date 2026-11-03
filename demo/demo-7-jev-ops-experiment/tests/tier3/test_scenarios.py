import copy

import pytest
import yaml

from tier3 import vocab
from tier3.scenarios import SCENARIO_DIR, ScenarioError, load_all, parse_scenario

K001 = yaml.safe_load((SCENARIO_DIR / "K001.yaml").read_text())


def with_(**over) -> dict:
    d = copy.deepcopy(K001)
    for k, v in over.items():
        if "." in k:
            a, b = k.split(".")
            d[a][b] = v
        else:
            d[k] = v
    return d


def test_catalogue_loads_k000_to_k010():
    cat = load_all()
    assert list(cat) == [f"K{i:03d}" for i in range(11)]


def test_fault_names_match_the_group_and_cover_all_ten():
    types = [s.fault.type for s in load_all().values() if s.fault]
    assert sorted(types) == sorted(vocab.FAULT_TYPES)
    assert vocab.FAULT_TYPES == ("scale_zero", "kill_pod", "memory_limit", "network_partition",
                                 "readiness_probe_fail", "config_corruption", "dependency_removal",
                                 "replica_overload", "node_taint", "pvc_full")


def test_healthy_control_shape():
    k = load_all()["K000"]
    assert k.fault is None and k.expected.incident is False
    assert (k.expected.affected_component, k.expected.diagnosis) == ("none", "none")
    assert k.expected.acceptable_actions == ("no_action",)


def test_targets_as_in_the_design():
    cat = load_all()
    assert cat["K007"].fault.target == "valkey-cart"
    assert cat["K009"].fault.target == "jev-worker2"
    others = [s for k, s in cat.items() if s.fault and k not in ("K007", "K009")]
    assert {s.fault.target for s in others} == {"cart"}


def test_unknown_fault_type_named_in_error():
    with pytest.raises(ScenarioError, match="cpu_hog"):
        parse_scenario(with_(fault={"type": "cpu_hog", "target": "cart"}))


@pytest.mark.parametrize("node", ["jev-worker", "jev-control-plane"])
def test_taint_only_on_application_workers(node):
    with pytest.raises(ScenarioError, match="application worker"):
        parse_scenario(with_(fault={"type": "node_taint", "target": node}))


def test_non_deployment_target_rejected():
    with pytest.raises(ScenarioError, match="not a demo deployment"):
        parse_scenario(with_(fault={"type": "scale_zero", "target": "jev-worker2"}))


@pytest.mark.parametrize("value", ["", "   ", None])
def test_missing_rationale(value):
    with pytest.raises(ScenarioError, match="rationale"):
        parse_scenario(with_(rationale=value))


def test_rationale_field_required():
    d = with_()
    del d["rationale"]
    with pytest.raises(ScenarioError, match="rationale"):
        parse_scenario(d)


def test_out_of_vocabulary_diagnosis_names_value_and_allowed_set():
    with pytest.raises(ScenarioError) as e:
        parse_scenario(with_(**{"expected.diagnosis": "resource_exhaustion"}))
    assert "resource_exhaustion" in str(e.value) and "service_unavailable" in str(e.value)


def test_out_of_vocabulary_action():
    with pytest.raises(ScenarioError, match="reboot"):
        parse_scenario(with_(**{"expected.acceptable_actions": ["reboot"]}))


def test_unknown_field_rejected():
    with pytest.raises(ScenarioError, match="severity"):
        parse_scenario(with_(severity="high"))
    with pytest.raises(ScenarioError, match="expected.notes"):
        parse_scenario(with_(**{"expected.notes": "x"}))


def test_control_cannot_expect_an_incident():
    with pytest.raises(ScenarioError, match="no fault"):
        parse_scenario(with_(fault=None))


def test_poll_interval_must_fit_the_window():
    with pytest.raises(ScenarioError, match="poll_interval_s"):
        parse_scenario(with_(poll_interval_s=600))


def test_vocabularies_are_separate_from_tier2():
    from tier2 import vocab as v2
    assert vocab.DIAGNOSES != v2.DIAGNOSES
    assert set(vocab.NODES) <= set(vocab.COMPONENTS) and "none" not in vocab.COMPONENTS
    assert "unknown" in vocab.DIAGNOSES and "no_action" in vocab.ACTIONS
