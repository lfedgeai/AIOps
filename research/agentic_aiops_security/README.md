# 🔬 Project Title: Scope & History-Aware Agentic Security (SHAS)

> **Executive Summary:** A wrapper on SPIFFE/SPIRE that binds an agent's formulation, scope, local history, and commission window, so a context attack carried across a trust boundary cannot become the next brief or the primary store.

---

## 📌 Research Overview
* **Status:** `🏗️ In-Progress`
* **Domain:** Agentic AIOps Security
* **Core Question:** When does the SHAS architecture entail that direct prompt injection, indirect prompt injection, context hacking of shared memory, and temporal reuse of memory or commission stay inside the minted task?

Working papers by O'Neill Joseph, November 2026. Empirical tests are left for a later stage.

---

## 🎯 Desired Outcomes
* **Primary Objective:** Reconstruct the architecture as an argument, name the joints where a conclusion is not yet entailed, and supply the connections that close those joints.
* **Key Deliverables:**
    - [x] **Architecture note:** Distributed framework, with the consolidated flow and the trust, minting, and execution diagrams.
    - [x] **Validation paper:** Which conclusions follow from the note, and under which assumptions.
    - [x] **Critique:** Which of those assumptions the note does not yet supply.
    - [x] **Repairs:** The policy, enforcement, and effect-inspection connections that close the remaining joint.
    - [ ] **Bibliography:** A checked entry for every external citation key.
    - [ ] **Empirical evaluation:** Tests of the closed architecture, once the argument is stable.
    - [ ] **White paper:** Formal write-up after review.

---

## 🛠 Methodology & Framework
The work is a theoretical reconstruction in three passes. The architecture note states the components. The validation paper asks whether each conclusion is entailed by those components. The critique grants the repairs already in the note and marks the joints that still need a premise. The repairs paper supplies those premises from mechanisms published since 2017: an external authorization decision before a profile is published, a policy enforcement point on the tool call, and an inspection of staged effects before reconcile.

* **Approach:** Argument reconstruction. Each conclusion has to follow from a connection the architecture actually states.
* **Core Logic:** SPIFFE/SPIRE answers who is calling and where it runs. SHAS adds the formulation (a published check value), the profile (scope and grain), the history (reputation and the Recorder), and the clock (the Bellman countdown). A finished report enters workload memory only after drift, sanitization, a direct-injection check, and a chain audit.
* **Tech Stack:** `Markdown`, `Pandoc`, `SPIFFE/SPIRE` (the identity layer under study). Mechanisms named in the repairs: `Cedar`, `Zanzibar`, `Progent`, `in-toto`, `CaMeL`.

Build the working papers from the repository root:

```bash
./scripts/build-papers.sh
```

HTML is written to `results/papers/`. That directory is generated and is not kept in git. Pandoc resolves `bibliography: ../references/references.bib` from `docs/papers/`, which is where the script runs. External citation keys are listed in the bibliography and still need checked entries, so a build will warn on those cites.

---

## 📊 Data Management & Transparency
1. **Source:** No dataset in this stage. The inputs are the architecture note and the literature cited from the working papers.
2. **Processing:** Papers are Markdown with one shared bibliography at `docs/references/references.bib`. Companion papers cite each other as `@validation` and `@critique`.
3. **Ethical Considerations:** The study describes containment of context attacks. It does not include personal data, production credentials, or procedures for carrying out those attacks.

---

## 📂 Repository Structure
```text
├── data/                  # Later datasets for the empirical stage
├── code/                  # Later prototypes and evaluation code
├── scripts/               # Paper build
├── results/               # Generated paper builds
├── docs/
│   ├── architecture/      # SHAS framework note
│   ├── figures/           # Flow, trust, minting, and execution diagrams
│   ├── papers/            # Validation, critique, and repairs
│   └── references/        # Shared bibliography
└── README.md
```

## Reading order

1. [Architecture note](docs/architecture/shas-framework.md), with the [consolidated flow](docs/figures/shas-flow.svg) and the diagrams for [trust boundaries](docs/figures/shas-trust-boundaries.svg), [minting](docs/figures/shas-minting.svg), and [execution](docs/figures/shas-execution.svg).
2. [The Logic of Scope-and-History Agentic Security](docs/papers/scope-history-validation.md) — whether the conclusions follow.
3. [Where the Scope-and-History Argument Does Not Follow](docs/papers/scope-history-critique.md) — which joints are still open.
4. [Supplying the Missing Connections](docs/papers/scope-history-repairs.md) — the connections that close the remaining joint.
