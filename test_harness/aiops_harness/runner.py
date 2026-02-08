from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import mlflow
from rich.console import Console

from .config import load_config, RootConfig
from .llm.base import LLMProvider
from .llm.llamastack import LlamaStackProvider
from .llm.openai_compat import OpenAICompatibleProvider
from .llm.ollama import OllamaProvider
from .data.loaders import load_dataset
from .pipeline.correlation import correlate_signals, summarize_events_for_prompt
from .pipeline.rca import root_cause_analysis
from .pipeline.remediation import suggest_remediation
from .pipeline.executor import make_executor
from .metrics import compute_mttd, compute_mttr, compute_mmtd


console = Console()


def make_provider(cfg: RootConfig) -> LLMProvider:
	ptype = cfg.provider.type.lower()
	params = cfg.provider.params or {}
	if ptype == "llamastack":
		return LlamaStackProvider(**params)
	if ptype == "openai_compat":
		return OpenAICompatibleProvider(**params)
	if ptype == "ollama":
		return OllamaProvider(**params)
	raise ValueError(f"Unknown provider type: {ptype}")


def _log_artifact_text(text: str, artifact_path: str, file_name: str) -> None:
	tmp_dir = ".mlflow_tmp"
	os.makedirs(tmp_dir, exist_ok=True)
	fp = os.path.join(tmp_dir, file_name)
	with open(fp, "w", encoding="utf-8") as f:
		f.write(text)
	mlflow.log_artifact(fp, artifact_path=artifact_path)


def run_experiment(config_path: str, dataset_dir: str, seed: int = 42) -> None:
	cfg = load_config(config_path)
	provider = make_provider(cfg)
	console.log(f"Using provider: {provider.name()}")

	logs, metrics, traces, incidents = load_dataset(
		dataset_dir,
		cfg.dataset.logs_file,
		cfg.dataset.metrics_file,
		cfg.dataset.traces_file,
		cfg.dataset.incidents_file,
	)

	mlflow.set_experiment(cfg.experiment.name)
	with mlflow.start_run(run_name=f"{cfg.experiment.name}-seed{seed}"):
		# Params
		mlflow.log_params({
			"provider": provider.name(),
			"seed": seed,
			"correlation.max_candidates": cfg.pipeline.correlation.max_candidates,
			"executor.type": cfg.pipeline.executor.type,
		})
		for k, v in cfg.experiment.tags.items():
			mlflow.set_tag(k, v)

		# Correlation
		corr = correlate_signals(
			provider=provider,
			prompt_template=cfg.pipeline.correlation.prompt_template,
			logs=logs,
			metrics=metrics,
			traces=traces,
			limit=cfg.pipeline.correlation.max_candidates,
		)
		mlflow.log_dict({"prompt": corr["prompt"], "response": corr["raw"]}, artifact_file="correlation/interaction.json")

		# RCA
		context = summarize_events_for_prompt(logs, metrics, traces, limit=cfg.pipeline.correlation.max_candidates)
		rca = root_cause_analysis(
			provider=provider,
			prompt_template=cfg.pipeline.rca.prompt_template,
			correlated_context=context,
		)
		mlflow.log_dict({"response": rca["raw"]}, artifact_file="rca/interaction.json")

		# Remediation
		rem = suggest_remediation(
			provider=provider,
			prompt_template=cfg.pipeline.remediation.prompt_template,
			rca_summary=rca["content"],
			max_steps=cfg.pipeline.remediation.max_steps,
		)
		mlflow.log_dict({"response": rem["raw"]}, artifact_file="remediation/interaction.json")

		# Execution (safe/no-op)
		executor = make_executor(cfg.pipeline.executor.type, **cfg.pipeline.executor.params)
		exec_result = executor.execute(steps=[rem["content"]], dry_run=True)
		mlflow.log_dict(exec_result, artifact_file="executor/result.json")

		# Metrics
		mttd = compute_mttd(incidents)
		mttr = compute_mttr(incidents)
		# For demo: Model detection and earliest signal timestamps are not resolved; log None MMTD
		mmtd = compute_mmtd(model_detection_ts=[None], earliest_related_signal_ts=[None])

		metrics: Dict[str, float] = {}
		if mttd is not None:
			metrics["MTTD"] = mttd
		if mttr is not None:
			metrics["MTTR"] = mttr
		if mmtd is not None:
			metrics["MMTD"] = mmtd
		if metrics:
			mlflow.log_metrics(metrics)

		console.log(f"Logged metrics: {metrics}")
		console.log("Run complete.")


def run_benchmark(config_path: str, dataset_dir: str, seeds: List[int]) -> None:
	for s in seeds:
		run_experiment(config_path, dataset_dir, seed=s)

