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
    - Signal correlation. The consolidated design uses one SQL query. The siloed design sends separate queries to Thanos, Loki, and Tempo, and then joins the results in code.
    - RCA (root cause analysis), measured as the share of runs with the correct root cause.
    - Remediation, measured as recommendations, actual remediation, or a judge score, and MTTR (mean time to repair).

  The null hypothesis is that the storage design has no measurable effect on these four tasks. A result that confirms the null hypothesis is a valid outcome.
* **Key Deliverables:**
    - [X ] **Code** A performance evaluation harness
    - [X ] **Report** Formal documentatation of the performance evalution findings
    - [X ] **Visuals** Charts showing the differences in performance across signal storage, retrieval and AIOps MTTD/MTTR.

---

## 🛠 Methodology & Framework

* **Approach:** A evaluation harness comparing different signal storage design approaches. The work has three parts:
    1. Ingest. One OpenTelemetry Collector sends the same telemetry to both designs at the same time. The harness records the ingest rate, dropped data, disk use, CPU, and memory for each design.
    2. Retrieval and correlation, without an agent. The harness runs a fixed set of queries against both designs. Each query set builds the context for one incident, for example error spans, then the logs for those spans, then the metrics for the affected pods. The harness records the latency (p50 and p95), the number of backend calls, and the completeness of the result.
    3. AIOps evaluation. The study reuses the [agentic AIOps evaluation harness](../agentic_aiops_architectures/), its Kubernetes fault types, and its scorecard (detection, RCA, remediation, MTTD, MTTR, tokens, cost). That harness already reads telemetry from ClickHouse. The agent, the prompts, and the faults stay the same in both designs. The study runs two variants:
        - Variant A, same tools. Both designs give the agent the same tools (`search_logs`, `search_traces`, `search_metrics`). In the siloed design, Loki, Tempo, and Thanos are behind these tools. The tools return the same format in both designs. This variant measures the effect of the storage alone, mostly query latency.
        - Variant B, native queries. Each design gives the agent the query language of its own backends. The consolidated design keeps `query_clickhouse`, so the agent can join logs, traces, and metrics in one SQL query. The siloed design gives the agent one tool each for PromQL, LogQL, and TraceQL, so the agent must join the results itself. This variant measures the effect of the correlation approach on RCA, MTTD, and tokens.

      Variant A alone can show no difference only because the tools hide the storage design. Variant B shows the difference that the storage design makes to the work of the agent.
* **Tech Stack:** OpenTelemetry Collector, ClickHouse, Thanos (with Prometheus remote write), Grafana Loki, Grafana Tempo, Kubernetes, Python. Experiment tracking uses MLflow, the same as the agentic AIOps evaluation harness.

---

## 📊 Data Management & Transparency
1.  **Source:** Live telemetry data will be created/injested through OTEL, OTLP and the Otel Demo App. The study starts with the OTel Demo App and can add more demo applications later.
2.  **Processing:** The two designs correlate signals in different ways:
    - Consolidated: one SQL query joins logs, traces, and metrics on the trace ID, the service name, and a time window.
    - Siloed: the harness sends one PromQL query to Thanos, one LogQL query to Loki, and one TraceQL query to Tempo. Python code then joins the results on the same keys.

---

## 📂 Repository Structure
```text
├── data/               # Raw and processed datasets
├── code/               # code incl Jupyter or RMarkdown files for EDA
├── scripts/            # Reproducible scripts for training/analysis runs eg make files
├── docs/               # In-depth documentation and literature review
└── README.md           # This file
