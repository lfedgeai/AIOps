from statistics import mean

from generator.server_state import generate
from scenarios.definitions import scenario_names

import json
import random

SEEDS = range(200)


def test_all_fields_present_and_in_range():
    for name in scenario_names():
        for seed in range(50):
            d = generate(name, seed).to_dict()
            for f in ("cpu_pct", "memory_pct", "disk_pct"):
                assert 0 <= d[f] <= 100
            assert d["load_1m"] >= 0
            assert d["network_errors_per_min"] >= 0
            assert 0 <= d["app_error_rate_pct"] <= 100
            assert d["api_latency_ms"] > 0
            assert set(d["services"].values()) <= {"healthy", "degraded", "down"}
            assert d["restarts_last_hour"] >= 0
            assert isinstance(d["recent_events"], list)


def test_same_seed_same_output():
    for name in scenario_names():
        assert generate(name, 42) == generate(name, 42)


def test_different_seed_differs():
    assert generate("degraded", 1) != generate("degraded", 2)


def test_global_rng_untouched():
    random.seed(7)
    expected_next = random.random()
    random.seed(7)
    generate("critical", 1)
    assert random.random() == expected_next


def test_json_round_trip():
    d = generate("ambiguous", 3).to_dict()
    assert json.loads(json.dumps(d)) == d


def _means(name, field):
    return mean(generate(name, s).to_dict()[field] for s in SEEDS)


def test_severity_ordering_of_means():
    for f in ("cpu_pct", "app_error_rate_pct", "api_latency_ms"):
        assert _means("healthy", f) < _means("degraded", f) < _means("critical", f)


def test_healthy_is_healthy():
    for s in SEEDS:
        d = generate("healthy", s)
        assert set(d.services.values()) == {"healthy"}
        assert d.cpu_pct < 40 and d.app_error_rate_pct < 1


def test_critical_has_a_down_service():
    for s in SEEDS:
        assert "down" in generate("critical", s).services.values()


def test_contradictory_signals_conflict():
    seen = set()
    for s in SEEDS:
        d = generate("contradictory", s)
        quiet_but_failing = d.cpu_pct < 40 and (d.app_error_rate_pct > 5 or "down" in d.services.values())
        busy_but_clean = d.cpu_pct > 80 and d.app_error_rate_pct < 1 and set(d.services.values()) == {"healthy"}
        assert quiet_but_failing or busy_but_clean
        seen.add("q" if quiet_but_failing else "b")
    assert seen == {"q", "b"}  # both variants occur


# --- application logs ---------------------------------------------------------

from generator.app_logs import LEVELS
from scenarios.definitions import get_scenario


def _levels(name, seed):
    return {x["level"] for x in generate(name, seed).logs}


def test_logs_present_and_well_formed():
    for name in scenario_names():
        for seed in range(30):
            logs = generate(name, seed).logs
            assert logs
            for x in logs:
                assert set(x) == {"ts", "level", "service", "message"}
                assert x["level"] in LEVELS and x["message"]
            assert [x["ts"] for x in logs] == sorted(x["ts"] for x in logs)


def test_logs_are_seed_deterministic():
    assert generate("critical", 9).logs == generate("critical", 9).logs
    assert generate("critical", 9).logs != generate("critical", 10).logs


def test_logs_match_scenario_severity():
    for seed in SEEDS:
        assert not _levels("healthy", seed) & {"ERROR", "FATAL"}
        assert _levels("critical", seed) & {"ERROR", "FATAL"}


def test_error_logs_name_the_affected_service():
    for seed in SEEDS:
        d = generate("critical", seed)
        bad = {n for n, st in d.services.items() if st != "healthy"}
        crash = [x for x in d.logs if x["level"] == "FATAL" or "exited" in x["message"]]
        assert all(x["service"] in bad for x in crash)


def test_hung_worker_evidence_is_in_logs():
    for seed in SEEDS:
        d = generate("hung_worker", seed)
        assert d.services["worker"] == "degraded"
        assert set(d.services.values()) <= {"healthy", "degraded"}
        assert d.cpu_pct < 50
        assert any(x["service"] == "worker" and x["level"] == "ERROR" for x in d.logs)
        assert not any(x["service"] == "worker" and x["level"] == "INFO" for x in d.logs)


def test_log_only_errors_metrics_look_healthy():
    for seed in SEEDS:
        d = generate("log_only_errors", seed)
        assert set(d.services.values()) == {"healthy"}
        assert d.cpu_pct < 50 and d.app_error_rate_pct < 2 and d.api_latency_ms < 300
        assert sum(1 for x in d.logs if x["level"] == "ERROR") >= 4


def test_logs_do_not_leak_scenario_or_answer():
    words = set(scenario_names()) | {"expected", "scenario", "restart the", "escalate"}
    for name in scenario_names():
        for seed in range(30):
            text = " ".join(x["message"] for x in generate(name, seed).logs).lower()
            assert not any(w in text for w in words), (name, seed)


def test_adding_logs_did_not_change_metrics_for_old_scenarios():
    # Logs are generated after metrics, so metrics for a seed are stable.
    d = generate("degraded", 1234)
    assert (d.cpu_pct, d.memory_pct, d.api_latency_ms) == (62.2, 74.1, 610)
