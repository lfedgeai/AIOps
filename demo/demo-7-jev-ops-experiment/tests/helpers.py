

from jev.client import Decision


def ok_decision(severity="normal", action="observe", human_review="no", confidence=0.9, latency=100.0,
                probable_cause="none", input_tokens=None, output_tokens=None, cost_usd=None,
                server_elapsed_ms=None, attempts=1, http_status=200):
    return Decision(status="ok", severity=severity, action=action, human_review=human_review,
                    probable_cause=probable_cause, input_tokens=input_tokens,
                    output_tokens=output_tokens, cost_usd=cost_usd,
                    cost_source="estimated" if cost_usd is not None else None,
                    server_elapsed_ms=server_elapsed_ms, attempts=attempts, http_status=http_status,
                    confidence=confidence, human_review_probability=0.2 if human_review == "no" else 0.8,
                    latency_ms=latency, model_version="fake-1")


class FakeClient:
    """Returns scripted decisions in order (last one repeats). Never touches the network."""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def decide(self, observations):
        self.calls.append(observations)
        i = min(len(self.calls) - 1, len(self.decisions) - 1)
        return self.decisions[i]
