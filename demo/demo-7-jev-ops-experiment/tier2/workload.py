"""Locust load-generator control through Envoy (/loadgen/). Verified on 3.1.0:
GET /loadgen/stats/requests -> {state, user_count, ...}; POST /loadgen/swarm user_count, spawn_rate."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request


class WorkloadError(RuntimeError):
    pass


class Workload:
    def __init__(self, base_url: str = "http://localhost:8080", get=None, post=None, timeout=10.0):
        self.base, self.timeout = base_url.rstrip("/"), timeout
        self._get = get or self._http_get
        self._post = post or self._http_post

    def _http_get(self, url: str) -> dict:
        with urllib.request.urlopen(url, timeout=self.timeout) as r:
            return json.load(r)

    def _http_post(self, url: str, data: dict) -> dict:
        body = urllib.parse.urlencode(data).encode()
        with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=self.timeout) as r:
            return json.load(r)

    def status(self) -> dict:
        return self._get(f"{self.base}/loadgen/stats/requests")

    def require_running(self) -> int:
        """Return the user count, or raise if Locust is not generating load."""
        s = self.status()
        if s.get("state") != "running" or not s.get("user_count"):
            raise WorkloadError(f"load generator not running (state={s.get('state')!r}, "
                                f"users={s.get('user_count')!r})")
        return int(s["user_count"])

    def set_users(self, users: int, spawn_rate: int = 5) -> None:
        r = self._post(f"{self.base}/loadgen/swarm",
                       {"user_count": users, "spawn_rate": spawn_rate})
        if not r.get("success"):
            raise WorkloadError(f"swarm rejected: {r!r}")
