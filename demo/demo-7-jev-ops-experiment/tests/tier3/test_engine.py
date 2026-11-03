import ast
import json
from pathlib import Path

import pytest

from jev.client import JevClient, JevConfig
from tests.tier3.helpers import SeqEngine, ok_decision
from tier3.engine import STATUS_ERROR, STATUS_INVALID, STATUS_OK, AIOpsEngine
from tier3.jev_adapter import QUESTIONS, JevAdapter, parse_answers
from tier3.vocab import ACTIONS, COMPONENTS, DIAGNOSES, NO_COMPONENT


def answers(**over):
    a = {
        "incident_detected": {"noul": 0.9},
        "affected_component": {"choice": "cart", "confidence": 0.8},
        "diagnosis": {"choice": "service_unavailable", "confidence": 0.6},
        "recommended_action": {"choice": "scale_up", "confidence": 0.7},
    }
    a.update(over)
    return {"model": "jev-1.13.0", "answers": a, "usage": {"input_tokens": 100, "output_tokens": 10}}


class Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body
        self.text = json.dumps(body)

    def json(self):
        return self._body


def adapter(resp, record=None):
    def post(url, headers, payload, timeout):
        if record is not None:
            record.append(payload)
        return resp
    return JevAdapter(JevClient(JevConfig(api_key="k", max_retries=1), post=post, sleep=lambda s: None))


def test_engine_interface_accepts_any_implementation():
    eng: AIOpsEngine = SeqEngine(ok_decision())
    assert eng.decide({}).status == "ok"


def test_valid_answer():
    d = adapter(Resp(200, answers())).decide({"deployments": []})
    assert d.status == STATUS_OK
    assert (d.incident_detected, d.affected_component, d.diagnosis, d.recommended_action) == (
        True, "cart", "service_unavailable", "scale_up")
    assert d.confidence == pytest.approx(0.7)
    assert d.telemetry["input_tokens"] == 100 and d.telemetry["model"] == "jev-1.13.0"


def test_node_is_a_valid_component():
    d = parse_answers(answers(affected_component={"choice": "jev-worker2"}))
    assert d.status == STATUS_OK and d.affected_component == "jev-worker2"


def test_out_of_set_component_is_invalid_and_raw_kept():
    body = answers(affected_component={"choice": "node-3"})
    d = parse_answers(body)
    assert d.status == STATUS_INVALID and d.raw_response == body and "node-3" in d.message


@pytest.mark.parametrize("p", [1.5, -0.1, True, None, "0.9"])
def test_bad_probability_is_invalid(p):
    assert parse_answers(answers(incident_detected={"noul": p})).status == STATUS_INVALID


def test_http_error_is_error_not_guessed():
    d = adapter(Resp(500, {"code": -1})).decide({})
    assert d.status == STATUS_ERROR and d.incident_detected is None


def test_reasoning_stored_and_does_not_change_answers():
    body = answers()
    body["reasoning"] = "cart has zero available replicas"
    d = parse_answers(body)
    assert d.reasoning == "cart has zero available replicas"
    assert d.affected_component == "cart"


def test_request_has_observation_and_exactly_four_questions():
    sent = []
    adapter(Resp(200, answers()), sent).decide({"deployments": [{"deployment": "cart"}]})
    req = sent[0]
    assert req["state"] == {"observations": {"deployments": [{"deployment": "cart"}]}}
    assert set(req["questions"]) == {"incident_detected", "affected_component", "diagnosis",
                                     "recommended_action"}


def test_question_options_match_vocabularies():
    assert tuple(QUESTIONS["affected_component"]["criteria"]) == (NO_COMPONENT, *COMPONENTS)
    assert tuple(QUESTIONS["diagnosis"]["criteria"]) == DIAGNOSES
    assert tuple(QUESTIONS["recommended_action"]["criteria"]) == ACTIONS
    assert QUESTIONS["incident_detected"]["type"] == "noul"


def test_questions_do_not_name_a_fault_type():
    from tier3.vocab import FAULT_TYPES
    text = json.dumps(QUESTIONS)
    for f in FAULT_TYPES:
        assert f not in text


def test_only_the_adapter_imports_jev_client():
    root = Path(__file__).resolve().parents[2] / "tier3"
    importers = []
    for f in root.rglob("*.py"):
        for n in ast.walk(ast.parse(f.read_text())):
            mods = [n.module] if isinstance(n, ast.ImportFrom) and n.module else \
                [a.name for a in n.names] if isinstance(n, ast.Import) else []
            if any(m == "jev.client" or m.startswith("jev.") for m in mods):
                importers.append(f.name)
    assert set(importers) == {"jev_adapter.py"}
