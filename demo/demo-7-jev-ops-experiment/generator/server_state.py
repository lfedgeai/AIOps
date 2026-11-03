"""Synthetic server-state generator.

Metrics are derived from a scenario profile, not drawn independently, so they
move together. All randomness comes from a private random.Random(seed).
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from generator.app_logs import generate_logs
from scenarios.definitions import Profile, Scenario, get_scenario

SERVICE_NAMES = ("api", "worker", "database", "cache")


@dataclass(frozen=True)
class ServerState:
    cpu_pct: float
    memory_pct: float
    disk_pct: float
    load_1m: float
    network_errors_per_min: int
    app_error_rate_pct: float
    api_latency_ms: int
    services: dict = field(default_factory=dict)  # name -> healthy|degraded|down
    restarts_last_hour: int = 0
    recent_events: list = field(default_factory=list)
    logs: list = field(default_factory=list)  # application log lines, oldest first

    def to_dict(self) -> dict:
        return asdict(self)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _stress(rng: random.Random, band: tuple[float, float], jitter: float) -> float:
    base = rng.uniform(*band)
    if jitter:
        base += rng.uniform(-jitter, jitter)
    return _clamp(base, 0.0, 1.0)


def _pick_services(rng: random.Random, profile: Profile) -> dict:
    n_down = rng.randint(*profile.services_down)
    n_degraded = rng.randint(*profile.services_degraded)
    pool = profile.affected or SERVICE_NAMES
    names = rng.sample(pool, min(len(pool), n_down + n_degraded))
    status = {n: "healthy" for n in SERVICE_NAMES}
    for n in names[:n_down]:
        status[n] = "down"
    for n in names[n_down:]:
        status[n] = "degraded"
    return status


def generate(scenario: Scenario | str, seed: int) -> ServerState:
    """Build one observation. Same scenario and seed always give the same result."""
    if isinstance(scenario, str):
        scenario = get_scenario(scenario)
    rng = random.Random(seed)
    profile = rng.choice(scenario.profiles)

    def res() -> float:
        return _stress(rng, profile.resource_band, profile.jitter)

    def fail() -> float:
        return _stress(rng, profile.failure_band, profile.jitter)

    cpu = _clamp(8 + 88 * res() + rng.gauss(0, 2), 0, 100)
    mem = _clamp(20 + 75 * res() + rng.gauss(0, 2), 0, 100)
    disk = _clamp(30 + 55 * res() + rng.gauss(0, 1.5), 0, 100)
    load = max(0.0, 0.2 + 7.5 * res() + rng.gauss(0, 0.2))
    net = max(0, round(45 * fail() ** 2 + rng.gauss(0, 0.5)))
    err = _clamp(0.05 + 18 * fail() ** 2 + rng.gauss(0, 0.05), 0, 100)
    latency = max(1, round(90 + 1600 * fail() ** 2 + rng.gauss(0, 10)))

    restarts = rng.randint(*profile.restarts)
    events = []
    if profile.events:
        events = rng.sample(list(profile.events), rng.randint(1, min(2, len(profile.events))))
    if restarts:
        events.append(f"{restarts} service restart(s) in the last hour")

    # Services before logs, so the logs can name the affected services.
    services = _pick_services(rng, profile)
    affected = [n for n, st in services.items() if st != "healthy"]
    logs = generate_logs(rng, profile.logs, affected, SERVICE_NAMES)

    return ServerState(
        cpu_pct=round(cpu, 1),
        memory_pct=round(mem, 1),
        disk_pct=round(disk, 1),
        load_1m=round(load, 2),
        network_errors_per_min=net,
        app_error_rate_pct=round(err, 2),
        api_latency_ms=latency,
        services=services,
        restarts_last_hour=restarts,
        recent_events=events,
        logs=logs,
    )
