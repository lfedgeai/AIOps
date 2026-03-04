# Linux Foundation Request for Proposals (RFP)  
**Development of an AIOps Test Harness** 

**RFP Number:** LF-EDGE-AIOPS-2026-001  
**Project Umbrella:** LF Edge  
**Issue Date:** March 4, 2026  
**Submission Deadline:** May 1, 2026, 5:00 PM PT  
**Contact:** proposals@lfedge.org

## 1. Overview

As IT environments transition from static monitoring to **Autonomous Operations**, the industry faces a "Paradox of Choice." With hundreds of open-source and proprietary tools for ingestion, anomaly detection, root cause analysis (RCA), and remediation, there is no standardized way to determine which component combinations yield the highest reliability at the lowest cost—especially within the complexity of distributed computing from microservices architectures to globally distributed edge computing infrastructure.

The LF Edge organization is soliciting proposals for the design and development of the **AIOps Component Permutation & Benchmark Harness (ACPBH)**. This project aims to build a modular, "plug-and-play" testbed that can programmatically swap AIOps components to establish gold-standard architectural patterns for distributed environments. With components we mean AIOps capabilities from anomaly detection, signal correlation, root-cause-analysis and remediation.

## 2. Project Objectives

The primary goal is to create a vendor-neutral framework that:

- **Permutates Components** — Automatically swaps modules (e.g., swapping  GPT-5 for a local Llama-4 model) within a live pipeline.
- **Injects Deterministic Stress** — Uses chaos engineering to simulate multi-layered outages across the environment
- **Evaluates Efficacy** — Measures the "Operational IQ" of a specific toolchain using standardized metrics.
- **Identifies Pareto Frontiers** — Establishes which combinations offer the best balance between latency, accuracy, and infrastructure cost.

## 3. Technical Scope: The "Four-Slot" Architecture

The harness must support a modular architecture divided into four primary pluggable interfaces:

| Slot        | Description                              | Examples                              |
|-------------|------------------------------------------|---------------------------------------|
| Ingestion   | Data collection and normalization        | OpenTelemetry or others               |
| Reasoning   | Analysis and Decision-making engine      | LLM Agents, Heuristics, ML Clustering |
| Enrichment  | Contextual data providing "ground truth" | Vector DBs, Graph DBs, CMDBs          |
| Action      | Execution of remediation steps           | Agentic Tools, Ansible, K8s Operators |

### Benchmark Suite

The harness must output a standardized **AIOps Scorecard**:

- **MTTD (Mean Time to Detect)**: Precision vs. Recall
- **RCA Accuracy**: Percentage of correct "Root Cause" identifications
- **Action Fidelity**: Success rate of automated remediations without human intervention
- **Cost-to-Resolve**: Token usage and compute overhead per incident

## 4. Submission Requirements

Submissions should be provided as a **PDF** or **Markdown** file (max 15 pages) and include:

1. **Architecture Diagram** — Isolation strategies to prevent data cross-contamination
2. **Integration Strategy** — Support for production deployment in brown and greenfield environments
3. **Governance Model** — Proposed maintainer structure and open-source contribution plan
4. **PoC (Optional)** — Link to a GitHub repository or video demonstration of the simulated/working approach

## 5. Evaluation Criteria

| Criteria         | Weight | Description                                                       |
|------------------|--------|-------------------------------------------------------------------|
| Modularity       | 35%    | Ease of adding new tool "slots" or interfaces                     |
| Metric Robustness| 25%    | Statistical validity and depth of the benchmarking engine         |
| Interoperability | 20%    | Alignment with MCP, OpenTelemetry, and other open source projects |
| Feasibility      | 20%    | Realistic timeline, resource requirements, and project roadmap    |

## 6. Important Dates

- **March 10 – March 20, 2026** — Q&A Period via LFEdge slack
- **May 1, 2026 (5:00 PM PT)** — Submission Deadline
- **May 20 – May 30, 2026** — Shortlist Interviews
- **June 2026** — Winner Announcement (Live at first June LFEdge TAC Meeting)

## 7. Contact & Support

For clarification on requirements or technical constraints, please reach out to the LF Edge Program Office:

- **Email:** proposals@lfedge.org
- **Slack:** #aiops-th-rfp on the LF Edge Workspace

---

*Copyright © 2026 The Linux Foundation. All rights reserved.*
