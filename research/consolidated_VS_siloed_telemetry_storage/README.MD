# 🔬 Project Title: [Compare telemetry storage design for AIOps performance]

> **Executive Summary:** The goal is to make an informed decision if, and if yes how much siloed VS consolidated telemetry signal storage impacts the performance of an AIOps solution.

---


## 📌 Research Overview
* **Status:** `🏗️ In-Progress`
* **Domain:** Telemetry signal storage and Signal Correlation
* **Core Question:** see exec summary above.
* **Tracking issue:** [lfedgeai/AIOps#45](https://github.com/lfedgeai/AIOps/issues/45)

---

## 🎯 Desired Outcomes
* **Primary Objective:** Give a clear recommendation for the telemetry storage design that best supports AIOps. The study compares two designs with the same telemetry, the same faults, and the same agents:
    - Consolidated storage: ClickHouse keeps logs, metrics, and traces in one OLAP database.
    - Siloed storage: Thanos keeps metrics, Grafana Loki keeps logs, and Grafana Tempo keeps traces.

  The study measures the effect of the storage design on four tasks:
    - Anomaly detection, measured as MTTD (mean time to detect).
    - Signal correlation. The consolidated design uses one SQL query. In the siloed design, the agent sends separate queries to Thanos, Loki, and Tempo, and then correlates the results itself.
    - RCA (root cause analysis), measured as the share of runs with the correct root cause.
    - Remediation, measured as actual remediation of the fault and MTTR (mean time to repair).

  The null hypothesis is that the storage design has no measurable effect on these four tasks. A result that confirms the null hypothesis is a valid outcome.
* **Key Deliverables:**
    - [X ] **Code** A performance evaluation harness
    - [X ] **Report** Formal documentatation of the performance evalution findings
    - [X ] **Visuals** Charts showing the differences in performance across signal storage, retrieval and AIOps MTTD/MTTR.

---

## 🛠 Methodology & Framework

* **Approach:** A evaluation harness comparing different signal storage design approaches. The work has three parts:
    1. Ingest. One OpenTelemetry Collector sends the same telemetry to both designs at the same time. The harness records the ingest rate, dropped data, disk use, CPU, and memory for each design.
    2. Retrieval and correlation, without an agent. This part extends the existing [telemetry storage backend benchmark](../telemetry_storage_backend/) instead of building a new one. That benchmark already has ClickHouse, Loki, the canonical query set, and a federated design that merges per-signal results in Python ([findings](../telemetry_storage_backend/docs/BENCHMARK_FINDINGS.md)). This study adds Thanos and Tempo, and a federated Thanos + Loki + Tempo design. The harness records the latency (p50 and p95), the number of backend calls, and the completeness of the result.

       In the siloed design, the correlation runs in code, so its speed depends on the algorithm and on the implementation language. To keep these effects apart from the storage design, the harness records the backend query time and the correlation time separately. The correlation step is implemented in Python and in one compiled language (Go), with the same algorithm.
    3. AIOps evaluation. The study reuses the [agentic AIOps evaluation harness](../agentic_aiops_architectures/), its Kubernetes fault types, and its scorecard (detection, RCA, remediation, MTTD, MTTR, tokens, cost). That harness already reads telemetry from ClickHouse. Each model runs as two agents:
        - Siloed agent. It has one tool each for PromQL (Thanos), LogQL (Loki), and TraceQL (Tempo). It must correlate the signals itself.
        - Consolidated agent. It queries ClickHouse with SQL, as in the existing harness.

      Only the telemetry tools differ between the two agents. Both agents get the same system prompt, the same context (source code and documentation), the same faults, and the same remediation permissions. The agents perform actual remediation in the cluster. The study measures the effect of the storage design on each phase: anomaly detection, signal correlation, RCA, and remediation.
* **Models:** The study starts with the top two executors from the [October 2026 evaluation](../agentic_aiops_architectures/docs/EVALUATION_RESULTS_Oct26.md): `prelude_glm-53-flash` and `litemaas_qwen3.8-27b`. It then adds one frontier model, Claude Opus 5.5 (`claude-opus-5-5`). The open models run through OpenRouter (`z-ai/glm-5.3-flash` and `qwen/qwen3.8-27b`), with one pinned provider for each model, so that both storage designs use the same model serving.
* **Runtime:** A local kind cluster with the OTel Demo App.
* **Tech Stack:** OpenTelemetry Collector, ClickHouse, Thanos (with Prometheus remote write), Grafana Loki, Grafana Tempo, Kubernetes, Python. Go for the compiled correlation step. Every run is logged to MLflow, the same as the agentic AIOps evaluation harness.

---

## 📊 Data Management & Transparency
1.  **Source:** Live telemetry data will be created/injested through OTEL, OTLP and the Otel Demo App. The study starts with the OTel Demo App and can add more demo applications later.
2.  **Processing:** The two designs correlate signals in different ways:
    - Consolidated: one SQL query joins logs, traces, and metrics on the trace ID, the service name, and a time window.
    - Siloed: one PromQL query goes to Thanos, one LogQL query to Loki, and one TraceQL query to Tempo. In the benchmark, code joins the results on the same keys (Python and Go). In the AIOps evaluation, the agent correlates the results itself.

---

## 📂 Repository Structure
```text
├── data/               # Raw and processed datasets
├── code/               # code incl Jupyter or RMarkdown files for EDA
├── scripts/            # Reproducible scripts for training/analysis runs eg make files
├── docs/               # In-depth documentation and literature review
└── README.md           # This file
