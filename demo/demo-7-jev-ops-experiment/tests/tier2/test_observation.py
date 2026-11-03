import ast
import inspect
import json

from tests.tier2.helpers import contaminated_telemetry
from tier2 import observation
from tier2.observation import (BYTE_BUDGET, MAX_LOG_LINES_PER_SERVICE, MAX_SERVICES, ServiceStats,
                               Telemetry, build_observation)
from tier2.scenarios import FaultSpec, GroundTruth, Scenario

TERMS = ["cartFailure", "paymentFailure", "flagd", "F003"]
GROUPS = {"window", "baseline_window", "services", "dependencies", "error_logs", "failing_traces"}


def test_field_groups_and_window():
    obs, _ = build_observation(contaminated_telemetry(), TERMS)
    assert set(obs) == GROUPS
    assert obs["window"]["start"] and obs["window"]["end"]
    row = obs["services"][0]
    for k in ("request_rate_per_s", "error_rate", "latency_p95_ms", "error_rate_change"):
        assert k in row


def test_same_shape_healthy_and_faulty():
    healthy = Telemetry("a", "b", services={"cart": ServiceStats(1, 0, 5, 10)})
    a, _ = build_observation(healthy, TERMS)
    b, _ = build_observation(contaminated_telemetry(), TERMS)
    assert set(a) == set(b) == GROUPS


def test_services_ranked_by_change_vs_baseline():
    obs, _ = build_observation(contaminated_telemetry(), TERMS)
    names = [r["service"] for r in obs["services"]]
    assert names.index("payment") < names.index("frontend")


def test_deterministic():
    a, _ = build_observation(contaminated_telemetry(), TERMS)
    b, _ = build_observation(contaminated_telemetry(), TERMS)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_caps_and_trim_notes_outside_request():
    svcs = {f"s{i:02d}": ServiceStats(1, i / 100, 5, 10 + i) for i in range(MAX_SERVICES + 5)}
    logs = [{"ts": str(i), "service": "s00", "level": "ERROR", "message": f"boom {'x' * i}"}
            for i in range(MAX_LOG_LINES_PER_SERVICE + 4)]
    obs, trims = build_observation(Telemetry("a", "b", services=svcs, logs=logs))
    assert len(obs["services"]) == MAX_SERVICES and trims["services"] == 5
    assert len(obs["error_logs"]) == MAX_LOG_LINES_PER_SERVICE and trims["log_lines"] == 4
    assert "trim" not in json.dumps(obs).lower()  # trim notes are never in the request


def test_byte_budget_drops_lowest_ranked_first(monkeypatch):
    monkeypatch.setattr(observation, "BYTE_BUDGET", 1500)
    svcs = {f"s{i:02d}": ServiceStats(1, i / 100, 5, 10 + i) for i in range(12)}
    obs, trims = build_observation(Telemetry("a", "b", services=svcs))
    assert len(json.dumps(obs).encode()) <= 1500
    assert trims.get("budget_services")
    assert obs["services"][0]["service"] == "s11"  # highest deviation survives
    assert BYTE_BUDGET == 60_000


def test_log_dedup_counts_repeats():
    logs = [{"ts": str(i), "service": "cart", "level": "ERROR", "message": f"failed id {i}"}
            for i in range(5)]
    obs, _ = build_observation(Telemetry("a", "b", services={"cart": ServiceStats(1, 1, 1, 1)}, logs=logs))
    assert len(obs["error_logs"]) == 1 and obs["error_logs"][0]["count"] == 5


def test_info_logs_excluded():
    logs = [{"ts": "1", "service": "cart", "level": "INFO", "message": "fine"}]
    obs, _ = build_observation(Telemetry("a", "b", logs=logs))
    assert obs["error_logs"] == []


# --- scrubber and anti-leak -------------------------------------------------
def test_scrubber_removes_feature_flag_keys_and_flag_names():
    t = contaminated_telemetry()
    obs, _ = build_observation(t, TERMS)
    s = json.dumps(obs)
    assert "feature_flag" not in s
    assert "paymentFailure" not in s and "cartFailure" not in s
    assert "F003" not in s
    # ordinary evidence is kept
    assert "Invalid token" in s


def test_flagd_term_is_redacted_wherever_it_appears():
    # The collector drops flagd services and edges; the scrubber is the second line.
    obs, _ = build_observation(contaminated_telemetry(), TERMS)
    assert "flagd" not in json.dumps(obs)


def test_variant_words_are_not_scrubbed_from_ordinary_text():
    t = Telemetry("a", "b", logs=[{"ts": "1", "service": "cart", "level": "ERROR",
                                   "message": "connection refused, retry on next request"}],
                  services={"cart": ServiceStats(1, 1, 1, 1)})
    obs, _ = build_observation(t, TERMS)
    assert "connection refused, retry on next request" in json.dumps(obs)


def test_builder_signature_has_no_scenario_types():
    params = inspect.signature(build_observation).parameters
    forbidden = {Scenario, FaultSpec, GroundTruth}
    for p in params.values():
        assert p.annotation not in forbidden and p.name not in {"scenario", "fault", "ground_truth", "truth"}


def test_observation_module_does_not_import_scenarios():
    tree = ast.parse(inspect.getsource(observation))
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module)
        elif isinstance(n, ast.Import):
            imported.update(a.name for a in n.names)
    assert not any(m.startswith("tier2.scenarios") or m == "tier2.faults" for m in imported)


def test_short_terms_use_word_boundaries():
    from tier2.observation import Scrubber
    s = Scrubber(["F003", "cartFailure"])
    assert s.text("run F003 done") == "run [redacted] done"
    assert s.text("trace 1f0034ab") == "trace 1f0034ab"  # hex id is untouched
    assert s.text("flag cartFailureRate high") == "flag [redacted]Rate high"
