"""Dead-man revert (design D6): a transient systemd user timer that runs
`python -m tier3 reset-faults` if the harness has not reset the fault in time.
Armed before inject, disarmed after a successful reset."""
from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
Run = Callable[[list[str]], subprocess.CompletedProcess]


class DeadmanError(RuntimeError):
    pass


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


class Deadman:
    def __init__(self, run: Run = _run, python: str = sys.executable, cwd: Path = ROOT):
        self._run, self.python, self.cwd = run, python, cwd

    def arm(self, seconds: int) -> str:
        unit = f"jev-fault-reset-{uuid.uuid4().hex[:8]}"
        r = self._run(["systemd-run", "--user", f"--unit={unit}", f"--on-active={int(seconds)}s",
                       f"--property=WorkingDirectory={self.cwd}", "--collect",
                       self.python, "-m", "tier3", "reset-faults"])
        if r.returncode != 0:
            raise DeadmanError(f"could not arm the dead-man timer: {r.stderr.strip()[:300]}")
        return unit

    def disarm(self, unit: str) -> None:
        self._run(["systemctl", "--user", "stop", f"{unit}.timer"])
