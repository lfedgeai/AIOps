# Agentic AIOps Approach Comparison — Architecture Proposal

> **Executive Summary:** A framework for systematically comparing agentic AIOps architectures across anomaly detection, signal correlation, RCA, and remediation—enabling data-driven selection of designs and workflows.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT ORCHESTRATION LAYER                           │
│  (Workflow scheduler, run configs, versioning, reproducibility)                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   DATA SOURCES      │  │   EXPERIMENT         │  │   METRICS &          │
│   & GROUND TRUTH    │  │   TRACKING           │  │   EVALUATION         │
│                     │  │   (MLflow/W&B)       │  │   (MTTD, MTTR, etc.) │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
          │                             │                             │
          └─────────────────────────────┼─────────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      AGENTIC APPROACH UNDER TEST                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                        │
│  │ Anomaly  │→ │ Signal   │→ │   RCA    │→ │Remediation│                        │
│  │Detection │  │Correlation│  │  Agent   │  │  Agent   │                        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TELEMETRY PLATFORM (OTEL)                                 │
│  Logs, Traces, Metrics ← OTEL Demo App + Chaos Engineering (flagd)                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Experiment Tracking

### 2.1 Why Structured Experiment Tracking?

Agentic approaches introduce variability: multi-step reasoning, tool calls, retries, and non-deterministic LLM outputs. You need:

- **Reproducibility**: Exact config (prompts, models, thresholds) for each run
- **Comparability**: Same workload and ground truth across approaches
- **Attribution**: Which design choice drove MTTD/MTTR changes

### 2.2 Recommended: MLflow + Custom Artifacts

Your repo already references MLflow. Extend it for agentic runs:

| Concept        | MLflow Mapping                        | Agentic Extensions                                      |
|----------------|---------------------------------------|---------------------------------------------------------|
| **Experiment** | One per comparison campaign            | e.g. `agentic_aiops_comparison_v1`                      |
| **Run**        | One per (approach × fault scenario)   | Tag: `approach`, `fault_flag`, `variant`                 |
| **Params**     | Model, quantile, etc.                  | Add: `llm_model`, `tool_set`, `max_steps`, `temperature`|
| **Metrics**    | F1, PR-AUC, etc.                      | Add: `mttd_seconds`, `mttr_seconds`, `rca_accuracy`     |
| **Artifacts**  | Plots, reports                         | Add: full trace (tool calls, reasoning steps), cost log |

**Suggested structure:**

```
mlruns/
├── experiments/
│   └── agentic_comparison/
│       ├── run_001_cart_failure_openai_gpt4/
│       ├── run_002_cart_failure_local_llama/
│       └── run_003_payment_failure_openai_gpt4/
```

### 2.3 Alternative: Weights & Biases (W&B)

If you need rich LLM tracing (token counts, latency per step, cost), W&B has strong integrations for agent frameworks (LangChain, LlamaIndex). Trade-off: vendor lock-in vs. out-of-box LLM observability.

### 2.4 Minimum Schema for Each Run

```yaml
run_id: uuid
timestamp: ISO8601
approach:
  name: "single-agent-rca"
  components: ["anomaly:iforest", "rca:gpt4", "remediation:manual"]
  config: { ... }
workload:
  fault_flag: "cartFailure"
  variant: "on"
  injection_time: ISO8601
  recovery_time: ISO8601
ground_truth:
  root_cause: "cart-service"
  dataset_version: "otel_20260204"
metrics:
  mttd_seconds: 45.2
  mttr_seconds: 120.0
  rca_correct: true
  fp_count: 0
  fn_count: 0
  cost_usd: 0.02
artifacts:
  - trace.json
  - alert_timeline.json
```

---

## 3. Data Sources

### 3.1 Primary: OTEL Demo + Chaos Engineering (Existing)

Your `ground_truth_data_collector.py` and OTEL demo are already aligned:

| Source                | Format              | Use Case                            |
|-----------------------|---------------------|-------------------------------------|
| **flagd**             | `flag` + `variant`  | Ground truth: fault identity, timing|
| **OpenSearch**        | Traces, logs        | Telemetry for detection & RCA       |
| **Prometheus**        | Metrics             | SLI/SLO baselines, anomaly signals  |
| **Metadata JSON**     | `root_cause`, etc.  | Evaluation labels                   |

**Keep this as the canonical data source** — it provides:

- Deterministic fault injection (known start/end)
- Labeled root causes per experiment
- Multi-modal telemetry (logs, traces, metrics)

### 3.2 Secondary: Synthetic / Replayed Datasets

For scale and edge cases:

- **Replay**: Export OTEL data to Parquet/Arrow; replay with varied timing to stress detection latency.
- **Synthetic**: Generate synthetic anomalies (e.g., inject noise, delay spikes) for sensitivity analysis.

### 3.3 Data Pipeline Flow

```
flagd (fault on/off) → OTEL Demo app → OTLP → Collector → OpenSearch / Prometheus
                                                              ↓
                                              ground_truth_data_collector.py
                                                              ↓
                                              metadata_*.json + traces_*.json + logs_*.txt
                                                              ↓
                                              feature_pipeline.py → features.csv
                                                              ↓
                                              compare_detectors.py (existing)
                                              OR new agentic eval harness
```

### 3.4 Ground Truth Requirements for MTTD/MTTR

You need **wall-clock timestamps** for:

- `fault_injection_time`: when the fault flag is turned on
- `fault_recovery_time`: when the fault is turned off (or when recovery is confirmed)

These are already available from your experiment config (`EXPERIMENTS` in `ground_truth_data_collector.py`). Store them in metadata:

```json
{
  "flag": "cartFailure",
  "variant": "on",
  "injection_time": "2026-02-04T14:00:00Z",
  "recovery_time": "2026-02-04T14:10:00Z",
  "ground_truth_root_cause": "cart-service"
}
```

---

## 4. Measuring MTTD and MTTR

### 4.1 MTTD (Mean Time To Detect)

**Definition**: Time from fault injection to first correct alert/indication by the system.

| Approach                       | Measurement                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| **Batch evaluation** (current) | Not directly applicable — you have precision/recall on fixed windows.       |
| **Streaming / live**           | `MTTD = first_alert_time - fault_injection_time`                             |

For agentic comparison, you need **streaming or time-ordered evaluation**:

1. **Option A — Simulation**  
   Replay telemetry in time order; record the first timestamp at which the detector (or agent) would have raised an alert. Use your existing windowed features but score windows in chronological order and take the first above-threshold.

2. **Option B — Live harness**  
   Run the OTEL demo with fault injection; pipe telemetry to the agentic system in real time; record `first_alert_ts`. More realistic but harder to reproduce.

3. **Proxy for batch**  
   If you must stay batch: treat each fault window as a separate "incident"; compute **time-to-first-detection-within-window** as the offset from window start to the first positive in a sliding sub-window. Less ideal but usable.

**Formula** (per incident):

```
MTTD_i = first_positive_alert_timestamp - fault_injection_timestamp
```

**Aggregate**:

```
MTTD = mean(MTTD_i) over all incidents where detection occurred
MTTD_missed = count(incidents with no detection)  # report separately
```

### 4.2 MTTR (Mean Time To Resolve)

**Definition**: Time from first detection to confirmed resolution (fault cleared, service healthy).

| Approach   | Measurement                                                                 |
|------------|-----------------------------------------------------------------------------|
| **Automated** | `MTTR = fault_recovery_time - first_alert_time` (if remediation succeeded) |
| **Manual**   | `MTTR = human_resolution_time - first_alert_time` (requires human timestamp)|

For agentic remediation:

1. **Automated remediation**  
   - Agent triggers fix (e.g., rollback, scaling).  
   - Resolution = when metrics/traces return to baseline (or when flag is turned off in controlled experiments).

2. **Semi-automated**  
   - Agent proposes action; human approves. Resolution = approval time or execution time, depending on definition.

**Formula** (per incident):

```
MTTR_i = fault_recovery_timestamp - first_alert_timestamp
```

**Aggregate**:

```
MTTR = mean(MTTR_i) over all resolved incidents
```

### 4.3 Instrumentation Hooks

To measure MTTD/MTTR you need to capture:

| Event                     | Source                   | When to log                         |
|---------------------------|--------------------------|-------------------------------------|
| `fault_injection`         | Experiment orchestrator  | When flag is set                     |
| `first_alert`             | Agentic system           | When anomaly/RCA agent first alarms  |
| `remediation_triggered`   | Remediation agent        | When fix is executed                 |
| `fault_recovery`          | Experiment orchestrator | When flag is cleared                 |

Add a small **event ledger** (e.g., JSONL or SQLite) that the harness writes to, and that your metrics pipeline reads.

---

## 5. Other Considerations

### 5.1 RCA Accuracy

- **Definition**: % of incidents where the identified root cause matches ground truth.
- **Data**: Use `ground_truth_root_cause` from metadata; compare to agent’s RCA output (exact or fuzzy match).
- **Metric**: `rca_accuracy = correct_rca_count / total_incidents`

### 5.2 Action Fidelity (Remediation Success)

- **Definition**: % of automated remediations that succeed without rollback or manual override.
- **Measurement**: In chaos experiments, success = metrics return to baseline after remediation and no secondary failures.
- **Requirement**: Store remediation actions and outcomes in experiment artifacts.

### 5.3 Cost-to-Resolve

- **Token usage**: Log prompt + completion tokens per run.
- **Compute**: CPU/GPU time for inference.
- **Formula**: `cost_per_incident = (tokens * price_per_token) + (compute_hours * price_per_hour)`

### 5.4 Reproducibility

- **Seeds**: Fix random seeds for any stochastic component (LLM temp=0 where possible, or document temp).
- **Versioning**: Pin OTEL demo, agent framework, and model versions in each run.
- **Environment**: Docker or Conda env files per campaign.

### 5.5 Statistical Validity

- **Multiple runs**: Run each (approach, fault) combination ≥ 3 times; report mean and std for MTTD, MTTR, F1.
- **Confidence intervals**: Use bootstrap or t-intervals for small N.

### 5.6 Baseline Comparisons

Compare agentic approaches against:

- **Rule-based**: Simple threshold alerts.
- **Classical ML**: Your current IsolationForest/COPOD/RRCF pipeline.
- **Single-model vs. multi-agent**: e.g., one LLM vs. specialized agents per stage.

### 5.7 Isolation and Contamination

- **Data split**: Strict separation of train/val/test (already in your `in_training` logic).
- **Temporal leakage**: Ensure no future data in training; use expanding or rolling windows for time-series.

---

## 6. Suggested Directory Layout

```
research/agentic_aiops_architectures/
├── data/
│   ├── raw/                    # OTEL demo exports (traces, logs, metrics)
│   ├── processed/              # features.csv, metadata
│   └── ground_truth/           # fault injection timestamps, root causes
├── code/
│   ├── agents/                 # Agent implementations (e.g., RCA, remediation)
│   ├── harness/                # Experiment runner, event ledger
│   └── evaluation/             # MTTD/MTTR/RCA metrics computation
├── scripts/
│   ├── run_experiment.sh       # Single (approach × fault) run
│   ├── run_campaign.sh         # Full comparison campaign
│   └── aggregate_mlflow.py     # Roll up MLflow runs into comparison report
├── results/
│   ├── runs/                   # Per-run outputs (or symlink to mlruns)
│   └── reports/                # Aggregated comparison HTML/JSON
└── docs/
    ├── ARCHITECTURE_PROPOSAL.md # This document
    └── METRICS_DEFINITIONS.md   # Formal MTTD/MTTR/RCA definitions
```

---

## 7. Implementation Roadmap

| Phase | Focus                         | Deliverables                                              |
|-------|-------------------------------|-----------------------------------------------------------|
| **1** | Experiment tracking           | MLflow integration, run schema, basic dashboard          |
| **2** | MTTD/MTTR instrumentation    | Event ledger, metrics computation from ledger             |
| **3** | Agentic harness               | Pluggable RCA/remediation agents, wiring to OTEL data     |
| **4** | Comparison campaigns          | Scripts to run N approaches × M faults, aggregate reports |
| **5** | Dashboard                     | Visual comparison (MTTD, MTTR, RCA, cost) per approach    |

---

## References

- Existing: `research/anomaly_detection/performance_comparison/` — detector comparison, feature pipeline, chaos data collection
- Existing: `test_harness/README.md` — AIOps Scorecard (MTTD, RCA Accuracy, Action Fidelity, Cost)
- OTEL Demo: `research/anomaly_detection/performance_comparison/chaos_engineering/otel-demo`
