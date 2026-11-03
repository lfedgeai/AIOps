import pytest

from scenarios.definitions import (ACTIONS, CAUSES, HUMAN_REVIEW, REQUIRED_SCENARIOS, SCENARIOS, SEVERITIES,
                                   Expected, get_scenario, scenario_names)


def test_required_scenarios_present():
    for n in REQUIRED_SCENARIOS:
        assert get_scenario(n).name == n


def test_unknown_scenario_lists_valid_names():
    with pytest.raises(KeyError) as e:
        get_scenario("nope")
    assert "healthy" in str(e.value)


def test_expected_values_in_allowed_sets():
    for s in SCENARIOS.values():
        assert s.expected.severity in SEVERITIES
        assert s.expected.action in ACTIONS
        assert s.expected.human_review in HUMAN_REVIEW
        assert s.expected.probable_cause in CAUSES


def test_expected_rejects_bad_values():
    with pytest.raises(ValueError):
        Expected("meh", "observe", "no", "none")
    with pytest.raises(ValueError):
        Expected("normal", "reboot", "no", "none")
    with pytest.raises(ValueError):
        Expected("normal", "observe", "maybe", "none")
    with pytest.raises(ValueError):
        Expected("normal", "observe", "no", "gremlins")


def test_rationale_and_profiles_present():
    for s in SCENARIOS.values():
        assert s.rationale.strip()
        assert s.profiles


def test_expected_is_static_and_seed_independent():
    from generator.server_state import generate
    before = get_scenario("degraded").expected
    generate("degraded", 1)
    generate("degraded", 2)
    assert get_scenario("degraded").expected == before


def test_severity_escalates_across_scenarios():
    order = {s: i for i, s in enumerate(SEVERITIES)}
    sev = [order[get_scenario(n).expected.severity] for n in ("healthy", "degraded", "critical")]
    assert sev == sorted(sev) and len(set(sev)) == 3


def test_names_listed():
    assert set(REQUIRED_SCENARIOS) <= set(scenario_names())
