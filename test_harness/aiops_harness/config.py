from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import yaml
import os


class ExperimentConfig(BaseModel):
	name: str = Field(default="aiops-experiment")
	tags: Dict[str, str] = Field(default_factory=dict)


class ProviderConfig(BaseModel):
	type: str
	params: Dict[str, Any] = Field(default_factory=dict)


class CorrelationConfig(BaseModel):
	prompt_template: str = Field(default="Given events, group related ones and identify the incident.")
	max_candidates: int = Field(default=100)


class RCAConfig(BaseModel):
	prompt_template: str = Field(default="Identify the root cause from the correlated context.")


class RemediationConfig(BaseModel):
	prompt_template: str = Field(default="Suggest remediation steps for the identified root cause.")
	max_steps: int = Field(default=5)


class ExecutorConfig(BaseModel):
	type: str = Field(default="noop")
	params: Dict[str, Any] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
	logs_file: str = Field(default="logs.jsonl")
	metrics_file: str = Field(default="metrics.jsonl")
	traces_file: str = Field(default="traces.jsonl")
	incidents_file: str = Field(default="incidents.jsonl")


class PipelineConfig(BaseModel):
	correlation: CorrelationConfig = Field(default_factory=CorrelationConfig)
	rca: RCAConfig = Field(default_factory=RCAConfig)
	remediation: RemediationConfig = Field(default_factory=RemediationConfig)
	executor: ExecutorConfig = Field(default_factory=ExecutorConfig)


class RootConfig(BaseModel):
	experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
	provider: ProviderConfig
	pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
	dataset: DatasetConfig = Field(default_factory=DatasetConfig)


def _expand_env(obj: Any) -> Any:
	"""
	Recursively expand environment variables in strings.
	Unresolved ${VAR} placeholders are converted to None to allow downstream defaults.
	"""
	if isinstance(obj, dict):
		return {k: _expand_env(v) for k, v in obj.items()}
	if isinstance(obj, list):
		return [_expand_env(v) for v in obj]
	if isinstance(obj, str):
		expanded = os.path.expandvars(obj)
		# If still contains ${...}, treat as unresolved → None
		if "${" in expanded and "}" in expanded:
			return None
		return expanded
	return obj


def load_config(path: str) -> RootConfig:
	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f)
	data = _expand_env(data)
	return RootConfig(**data)

