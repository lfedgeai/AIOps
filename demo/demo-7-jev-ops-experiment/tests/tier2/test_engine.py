import ast
import inspect
import json
from pathlib import Path

import pytest

from jev.client import JevClient, JevConfig
from tests.tier2.helpers import FakeEngine, ok_decision
from tier2 import jev_adapter
from tier2.engine import STATUS_ERROR, STATUS_INVALID, STATUS_OK, AIOpsEngine
from tier2.jev_adapter import QUESTIONS, JevAdapter, parse_answers
from tier2.vocab import ACTIONS, DIAGNOSES, NO_SERVICE, SERVICES


def answers(**over):
    a = {
        "incident_detected": {"noul": 0.9},
        "affected_service": {"choice": "payment", "confidence": 0.8},
        "diagnosis": {"choice": "application_failure", "confidence": 0.6},
        "recommended_action": {"choice": "investigate", "confidence": 0.7},
    }
    a.update(over)
    return {"answers": a, "usage": {"input_tokens": 100, "output_tokens": 10}}


class Resp:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def adapter(resp_or_exc, record=None):
    def post(url, headers, payload, timeout):
        if record is not None:
            record.append(payload)
        if isinstance(resp_or_exc, Exception):
            raise resp_or_exc
        return resp_or_exc
    cfg = JevConfig(api_key="k", max_retries=1)
    return JevAdapter(JevClient(cfg, post=post, sleep=lambda s: None))


def test_engine_interface_accepts_any_implementation():
    eng: AIOpsEngine = FakeEngine(ok_decision())
    assert eng.decide({}).status == "ok"


def test_complete_answer_ok_and_raw_kept():
    d = adapter(Resp(200, answers())).decide({"services": []})
    assert d.status == STATUS_OK
    assert (d.incident_detected, d.affected_service, d.diagnosis, d.recommended_action) == (
        True, "payment", "application_failure", "investigate")
    assert d.raw_response == answers()
    assert d.confidence == pytest.approx(0.7)
    assert d.telemetry["input_tokens"] == 100 and d.telemetry["attempts"] == 1


def test_no_confidence_is_none():
    body = answers(affected_service={"choice": "cart"}, diagnosis={"choice": "none"},
                   recommended_action={"choice": "no_action"})
    d = parse_answers(body)
    assert d.status == STATUS_OK and d.confidence is None


def test_incident_threshold():
    assert parse_answers(answers(incident_detected={"noul": 0.5})).incident_detected is True
    assert parse_answers(answers(incident_detected={"noul": 0.49})).incident_detected is False


@pytest.mark.parametrize("over", [
    {"affected_service": {"choice": "not-a-service"}},
    {"diagnosis": {"choice": "gremlins"}},
    {"recommended_action": {"choice": "reboot_the_world"}},
    {"incident_detected": {"noul": 1.5}},
    {"incident_detected": {}},
    {"diagnosis": None},
])
def test_out_of_set_is_invalid_not_coerced(over):
    d = parse_answers(answers(**over))
    assert d.status == STATUS_INVALID and d.raw_response is not None
    assert d.incident_detected is None or d.status == STATUS_INVALID


def test_non_dict_body_invalid_keeps_raw():
    d = parse_answers(["nope"])
    assert d.status == STATUS_INVALID and d.raw_response == ["nope"]


def test_transport_failure_is_error():
    import requests
    d = adapter(requests.ConnectionError("boom")).decide({})
    assert d.status == STATUS_ERROR and "gave up" in d.message


def test_non_json_reply_invalid():
    d = adapter(Resp(200, None, "<html>")).decide({})
    assert d.status == STATUS_INVALID


def test_request_shape_and_fixed_questions():
    sent = []
    adapter(Resp(200, answers()), sent).decide({"services": []})
    p = sent[0]
    assert p["state"] == {"observations": {"services": []}}
    assert list(p["questions"]) == ["incident_detected", "affected_service", "diagnosis",
                                    "recommended_action"]


def test_vocab_exits_always_available():
    assert {NO_SERVICE, *SERVICES} == set(QUESTIONS["affected_service"]["criteria"])
    assert {"unknown", "none"} <= set(QUESTIONS["diagnosis"]["criteria"]) <= set(DIAGNOSES)
    assert "no_action" in QUESTIONS["recommended_action"]["criteria"]
    assert set(QUESTIONS["recommended_action"]["criteria"]) == set(ACTIONS)


def test_tier1_request_unchanged_by_questions_arg():
    from jev.client import QUESTIONS as T1
    c = JevClient(JevConfig(api_key="k"), post=lambda *a: None)
    req = c.build_request({"cpu_pct": 1})
    assert req["questions"] == T1 and req["state"] == {"server_observations": {"cpu_pct": 1}}
    custom = c.build_request({"x": 1}, questions={"q": {}}, state_key="observations")
    assert custom["questions"] == {"q": {}} and custom["state"] == {"observations": {"x": 1}}


def test_recommendation_is_only_recorded():
    # A restart recommendation is stored on the Decision; nothing in tier2 can act on it.
    d = parse_answers(answers(recommended_action={"choice": "restart_service"}))
    assert d.recommended_action == "restart_service"
    for path in Path(__file__).parents[2].glob("tier2/*.py"):
        assert "docker restart" not in path.read_text() and "compose restart" not in path.read_text()


def test_only_the_adapter_imports_jev_client():
    for path in Path(__file__).parents[2].glob("tier2/**/*.py"):
        if path.name == "jev_adapter.py" or "testbed" in path.parts:  # testbed holds the demo checkout
            continue
        tree = ast.parse(path.read_text())
        for n in ast.walk(tree):
            mod = n.module if isinstance(n, ast.ImportFrom) else None
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else []
            assert mod != "jev.client" and "jev.client" not in names and mod != "jev", path
    assert "jev.client" in inspect.getsource(jev_adapter)
