
# 🔬 Project Title: [Compare telemetry storage design for AIOps performance]

> **Executive Summary:** The goal is to make an informed decision if, and if yes how much siloed VS consolidated telemetry signal storage impacts the performance of an AIOps solution.

---


## 📌 Research Overview
* **Status:** `🏗️ In-Progress`
* **Domain:** Telemetry signal storage and Signal Correlation
* **Core Question:** see exec summary above.

---

## 🎯 Desired Outcomes
* **Primary Objective:** [e.g., To prove that $X$ is more efficient than $Y$ under constraints $Z$].
* **Key Deliverables:**
    - [X ] **Code** A performance evaluation harness
    - [X ] **Report** Formal documentatation of the performance evalution findings
    - [X ] **Visuals** Charts showing the differences in performance across signal storage, retrieval and AIOps MTTD/MTTR.

---

## 🛠 Methodology & Framework

* **Approach:** A evaluation harness comparing different signal storage design approaches.
* **Tech Stack:** tbd.

---

## 📊 Data Management & Transparency
1.  **Source:** Live telemetry data will be created/injested through OTEL, OTLP and the Otel Demo App.
2.  **Processing:** signal correlation approach tbd.

---

## 📂 Repository Structure
```text
├── data/               # Raw and processed datasets
├── code/               # code incl Jupyter or RMarkdown files for EDA
├── scripts/            # Reproducible scripts for training/analysis runs eg make files
├── docs/               # In-depth documentation and literature review
└── README.md           # This file
