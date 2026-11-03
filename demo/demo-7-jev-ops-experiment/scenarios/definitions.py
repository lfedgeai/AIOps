"""Scenario catalogue and ground truth.

Expected outcomes are defined here by hand. They are never produced by Jev or
any other LLM, and they do not depend on the generated numbers or the seed.
"""
from __future__ import annotations

from dataclasses import dataclass

SEVERITIES = ("normal", "degraded", "high", "critical")
ACTIONS = ("observe", "investigate", "restart", "escalate")
HUMAN_REVIEW = ("yes", "no")
# Diagnosis, scored separately from the three operational decisions.
CAUSES = ("none", "resource_exhaustion", "dependency_failure", "application_bug",
          "configuration", "network", "unknown")


@dataclass(frozen=True)
class Expected:
    severity: str
    action: str
    human_review: str
    probable_cause: str

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"bad severity {self.severity!r}")
        if self.action not in ACTIONS:
            raise ValueError(f"bad action {self.action!r}")
        if self.human_review not in HUMAN_REVIEW:
            raise ValueError(f"bad human_review {self.human_review!r}")
        if self.probable_cause not in CAUSES:
            raise ValueError(f"bad probable_cause {self.probable_cause!r}")

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "action": self.action,
            "human_review": self.human_review,
            "probable_cause": self.probable_cause,
        }


@dataclass(frozen=True)
class Profile:
    """How the generator builds one flavour of a scenario.

    Bands are (low, high) "stress" levels in 0..1. Resource metrics (CPU,
    memory, disk, load) follow resource_band; failure metrics (network errors,
    error rate, latency) follow failure_band. They are equal for ordinary
    scenarios and deliberately different for contradictory ones.
    """

    resource_band: tuple[float, float]
    failure_band: tuple[float, float]
    jitter: float = 0.0  # per-metric extra noise on the stress level
    services_down: tuple[int, int] = (0, 0)  # inclusive range
    services_degraded: tuple[int, int] = (0, 0)
    restarts: tuple[int, int] = (0, 0)  # restarts in the last hour
    events: tuple[str, ...] = ()  # pool of recent-event messages
    affected: tuple[str, ...] = ()  # services that fail; empty = any service
    logs: tuple[tuple[str, int, int], ...] = ()  # (log kind, min, max) lines
    label: str = ""


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    expected: Expected
    rationale: str
    profiles: tuple[Profile, ...]


SCENARIOS: dict[str, Scenario] = {
    "healthy": Scenario(
        name="healthy",
        description="Low resource use, no errors, all services up.",
        expected=Expected("normal", "observe", "no", "none"),  # cause: nothing is wrong
        rationale="Every signal is in its normal range. Nothing needs doing.",
        profiles=(
            Profile(
                resource_band=(0.05, 0.25),
                failure_band=(0.0, 0.1),
                events=("scheduled backup completed", "log rotation completed"),
                logs=(("info_ok", 6, 10), ("warn_benign", 0, 1)),
                label="quiet",
            ),
        ),
    ),
    "degraded": Scenario(
        name="degraded",
        description="High load and slow responses, one service degraded.",
        expected=Expected("high", "investigate", "no", "resource_exhaustion"),  # cause: load, slow queries, pool exhausted
        rationale=(
            "Several signals agree that service quality is falling, but nothing "
            "is down. Investigate the cause. A human does not need to be paged yet."
        ),
        profiles=(
            Profile(
                resource_band=(0.55, 0.72),
                failure_band=(0.5, 0.65),
                services_degraded=(1, 1),
                restarts=(0, 1),
                events=("worker queue depth rising", "slow query warning"),
                logs=(("info_ok", 3, 6), ("warn_slow", 3, 6), ("error_timeout", 1, 3)),
                label="loaded",
            ),
        ),
    ),
    "critical": Scenario(
        name="critical",
        description="Resources near exhaustion, errors high, a service down.",
        expected=Expected("critical", "escalate", "yes", "resource_exhaustion"),  # cause: OOM kills, resources near 100%
        rationale=(
            "A service is down, resources are close to exhaustion and errors are "
            "very high. A restart alone is unlikely to be safe or enough, so a "
            "human must take over."
        ),
        profiles=(
            Profile(
                resource_band=(0.88, 1.0),
                failure_band=(0.85, 1.0),
                services_down=(1, 1),
                services_degraded=(1, 1),
                restarts=(3, 8),
                events=("out of memory kill", "service crash loop detected"),
                logs=(("info_ok", 0, 2), ("warn_slow", 1, 3), ("error_timeout", 3, 5),
                      ("error_crash", 2, 4)),
                label="failing",
            ),
        ),
    ),
    "ambiguous": Scenario(
        name="ambiguous",
        description="Middling, mixed signals. No service down, nothing clearly wrong.",
        expected=Expected("degraded", "investigate", "yes", "unknown"),  # cause: evidence too weak or conflicting to name one
        rationale=(
            "The evidence is weak and mixed, so it does not justify a restart or "
            "an escalation, but it is not clearly fine either. Investigate, and ask "
            "a human to look because the machine cannot be sure. This is a "
            "judgment call, not an objectively correct answer."
        ),
        profiles=(
            Profile(
                resource_band=(0.4, 0.6),
                failure_band=(0.25, 0.45),
                jitter=0.15,
                services_degraded=(0, 1),
                restarts=(0, 2),
                events=("deploy finished 2h ago", "cache hit rate dipped"),
                logs=(("info_ok", 4, 8), ("warn_benign", 2, 4), ("warn_slow", 0, 2),
                      ("error_timeout", 0, 1)),
                label="mixed",
            ),
        ),
    ),
    "contradictory": Scenario(
        name="contradictory",
        description="Signals that disagree: quiet hosts with failing service, or busy host with no symptoms.",
        expected=Expected("degraded", "investigate", "yes", "unknown"),  # cause: evidence too weak or conflicting to name one
        rationale=(
            "The data cannot all be true at once, or does not fit a known failure "
            "pattern. It is safest to investigate and get human review. Restart "
            "and escalate are not justified by contradictory evidence. This is a "
            "judgment call, not an objectively correct answer."
        ),
        profiles=(
            Profile(
                resource_band=(0.05, 0.2),
                failure_band=(0.8, 0.95),
                services_down=(1, 1),
                restarts=(0, 1),
                events=("health check failing", "no recent deploys"),
                # The failing service logs crashes AND "health check ok".
                logs=(("info_ok", 3, 5), ("error_crash", 1, 2), ("health_ok", 2, 3)),
                label="quiet_but_failing",
            ),
            Profile(
                resource_band=(0.88, 0.97),
                failure_band=(0.0, 0.05),
                restarts=(0, 0),
                events=("no alerts firing",),
                logs=(("info_ok", 6, 10),),
                label="busy_but_clean",
            ),
        ),
    ),
    "hung_worker": Scenario(
        name="hung_worker",
        description="Background worker deadlocked. Host metrics are quiet; the logs show why.",
        expected=Expected("high", "restart", "no", "application_bug"),  # cause: deadlock in job runner
        rationale=(
            "One stateless worker is stuck in a deadlock and jobs are piling up. "
            "The logs name the cause, nothing else is wrong, and a restart is the "
            "standard, low-risk fix for a hung worker. It is contained, so no "
            "human is needed first. The evidence for this is mostly in the logs."
        ),
        profiles=(
            Profile(
                resource_band=(0.1, 0.25),
                failure_band=(0.2, 0.35),
                services_degraded=(1, 1),
                affected=("worker",),
                events=("worker queue depth rising", "no recent deploys"),
                logs=(("info_api", 2, 4), ("error_hung", 2, 4), ("warn_queue", 1, 2)),
                label="deadlocked_worker",
            ),
        ),
    ),
    "log_only_errors": Scenario(
        name="log_only_errors",
        description="Metrics look healthy, but the logs show checkout failing after a deploy.",
        expected=Expected("high", "investigate", "yes", "application_bug"),  # cause: KeyError after deploy
        rationale=(
            "Host metrics and service statuses look normal, so a metrics-only view "
            "would say all is well. The logs show a repeated code error on "
            "checkout right after a deploy, so users are affected (high). A "
            "restart will not fix a code bug, so investigate, and a human should "
            "decide on a rollback. The evidence for this is only in the logs."
        ),
        profiles=(
            Profile(
                resource_band=(0.1, 0.3),
                failure_band=(0.15, 0.25),
                events=("deploy v2.14.0 finished 20 minutes ago",),
                logs=(("info_ok", 5, 8), ("error_app", 4, 7)),
                label="bad_deploy",
            ),
        ),
    ),
}

REQUIRED_SCENARIOS = ("healthy", "degraded", "critical", "ambiguous", "contradictory",
                      "hung_worker", "log_only_errors")


def scenario_names() -> list[str]:
    return list(SCENARIOS)


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise KeyError(
            f"unknown scenario {name!r}; valid names: {', '.join(SCENARIOS)}"
        ) from None
