## AIOps Test Harness (LlamaStack + MLflow)

This harness lets you plug different LLMs (via a provider abstraction that includes LlamaStack) to perform:
- Signal correlation across logs/metrics/traces
- Root Cause Analysis (RCA)
- Remediation step suggestions
- Optional execution of remediation actions (safe/no-op by default)

All experiments are tracked in MLflow, including configuration parameters, prompts, outputs, and performance metrics such as MTTD, MMTD, and MTTR.

### Quick Start
1) Python 3.10+
2) Install:

```bash
pip install -e .
```

3) Start or point to an MLflow tracking server. Example (local):

```bash
mlflow ui --backend-store-uri 'sqlite:///mlruns.db' --host 0.0.0.0 --port 5000         ─╯

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

4) Configure an LLM provider. For LlamaStack:
```bash
export LLAMA_STACK_BASE_URL=http://localhost:8080
export LLAMA_STACK_API_KEY=your_token_if_required
```
For OpenAI-compatible endpoints:
```bash
export OPENAI_API_BASE=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...
```

5) Run an example:
```bash
aiops-harness run --config examples/configs/basic.yaml --dataset examples/data
```

This will:
- Load dataset (logs/metrics/traces + incidents)
- Use the configured LLM pipeline for correlation, RCA, remediation
- Compute MTTD/MMTD/MTTR
- Log parameters, metrics, prompts, and outputs to MLflow

### Local LLMs (Ollama)
To run three local models (Llama 3.1, Mistral, Qwen2.5) via Ollama:
```bash
bash scripts/setup_local_llms.sh
export OLLAMA_BASE_URL=http://127.0.0.1:11434
```
Then run per-model configs and compare results in MLflow:
```bash
aiops-harness run --config examples/configs/ollama_llama3.yaml --dataset examples/data
aiops-harness run --config examples/configs/ollama_mistral.yaml --dataset examples/data
aiops-harness run --config examples/configs/ollama_qwen.yaml --dataset examples/data
```
If a model isn’t available on your platform, pull an alternative in Ollama (e.g., `gemma2`) and change `default_model` in the config.

### Concepts
- Providers: `LlamaStackProvider`, `OpenAICompatibleProvider`
- Pipelines: `correlation`, `rca`, `remediation`, `executor`
- Metrics:
  - MTTD: Mean Time To Detect (incident_detected_ts - incident_start_ts)
  - MMTD: Mean Model Time to Detect (model_detection_ts - earliest_related_signal_ts)
  - MTTR: Mean Time To Resolve (incident_resolved_ts - incident_start_ts)

### Configuration
See `examples/configs/basic.yaml` for a complete template. Key fields:
- `experiment.name`, `experiment.tags`
- `provider.type` (e.g., `llamastack` or `openai_compat`), `provider.params`
- `pipeline` sections for `correlation`, `rca`, `remediation`, `executor`
- `dataset` file names for logs, metrics, traces, incidents

### Safety
The default executor is a safe no-op that only logs suggested steps. You can implement and register a concrete executor for your environment, gated by approvals and RBAC.

### Extending
- Add new providers under `aiops_harness/llm/`
- Add new stages under `aiops_harness/pipeline/`
- Register them in your config by name



# What's included
CLI: aiops_harness/cli.py with commands run and benchmark.
Configs: YAML-driven setup (examples/configs/basic.yaml) defining provider, pipeline, and dataset.
Providers:
- LlamaStackProvider (aiops_harness/llm/llamastack.py)
- OpenAICompatibleProvider (aiops_harness/llm/openai_compat.py)

Data schema/loaders:
- aiops_harness/data/schema.py
- aiops_harness/data/loaders.py
- Sample data in examples/data/ for logs, metrics, traces, incidents.

Pipeline stages:
- Correlation: aiops_harness/pipeline/correlation.py
- RCA: aiops_harness/pipeline/rca.py
- Remediation: aiops_harness/pipeline/remediation.py
- Safe executor (no-op): aiops_harness/pipeline/executor.py

Metrics: aiops_harness/metrics.py computes MTTD, MTTR, and placeholder MMTD.

Runner: aiops_harness/runner.py orchestrates stages and logs params, metrics, prompts, and outputs to MLflow.

Project metadata: pyproject.toml and README.md.

# Notes
MTTD/MTTR are computed from examples/data/incidents.jsonl. MMTD MMTD is scaffolded (placeholder) — the function exists but we currently pass None because we don’t yet record when the model “detects” an incident or when the earliest related signal occurred.
The default executor is a safe no-op. You can implement a real executor and register it via config.
- To make MMTD real, log per-incident:
- model_detection_ts: timestamp when your correlation/RCA pipeline first flags the incident.
- earliest_related_signal_ts: min timestamp across the signals the model grouped as related.
Then pass those arrays into compute_mmtd(...) in runner.py.

To switch models/providers, edit examples/configs/basic.yaml (provider.type and params).
Completed: project scaffold, provider abstraction (LlamaStack + OpenAI-compatible), loaders/schema, correlation/RCA/remediation modules, metrics with MLflow logging, YAML configs and example dataset, safe executor, and README.
