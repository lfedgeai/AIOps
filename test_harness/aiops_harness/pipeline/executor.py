from __future__ import annotations

from typing import Any, Dict, List, Protocol
from rich.console import Console


console = Console()


class RemediationExecutor(Protocol):
	def name(self) -> str: ...
	def execute(self, steps: List[str], dry_run: bool = True) -> Dict[str, Any]: ...


class NoOpExecutor:
	def __init__(self, **params: Any) -> None:
		self.params = params

	def name(self) -> str:
		return "noop"

	def execute(self, steps: List[str], dry_run: bool = True) -> Dict[str, Any]:
		console.log(f"[NoOpExecutor] dry_run={dry_run} steps={steps}")
		return {"executed": [] if dry_run else steps, "dry_run": dry_run}


def make_executor(executor_type: str, **params: Any) -> RemediationExecutor:
	if executor_type == "noop":
		return NoOpExecutor(**params)
	raise ValueError(f"Unknown executor type: {executor_type}")

