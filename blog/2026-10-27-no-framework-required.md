# No Framework Required: What 328 Agent Trials Taught Us About Building an AIOps Runtime

*October 27, 2026*

Every agent project now starts with the same question: "Which framework — LangChain? LangGraph? Strands?" We asked it, answered it twice with *no*, and this post explains why — with data rather than taste. The short version: frameworks are the right answer for most agent projects, and we can show that ours isn't one.

We run an AIOps program with two artifacts. An **evaluation harness** that has now scored **328 trials** (a 288-cell matrix plus a 40-cell follow-up: six model slots × sixteen context configurations × three injected Kubernetes faults). And a **runtime harness** — the production-shaped composition of the winners — a closed loop that self-adopts workloads from telemetry, detects, confirms root cause, remediates behind safety gates, and verifies recovery. Both are pure Python. Neither uses an agent framework.

## First, the honest case for frameworks

If you're building the thing most teams are building — a product copilot, a RAG assistant, a chat experience — use a framework. Seriously:

- **Greenfield speed.** Frameworks get you from idea to demo in an afternoon, which matters when you're still discovering the product.
- **Ecosystem leverage.** Dozens of prebuilt integrations — document loaders, vector stores, tool catalogs — that you'd otherwise write yourself.
- **Dynamic agent graphs.** When the *model* decides what happens next and the topology varies per request, LangGraph-style orchestration earns its complexity.
- **Team scale and hiring.** Market-standard abstractions mean new engineers contribute on day one.

That's a majority of agent projects. Frameworks are a majority solution for a majority problem. We checked whether ours was one. It wasn't — and our own evaluation data is the reason.

## Five findings that decided it

### 1. Tool-schema wording is load-bearing

In our October matrix, three "diagnoser" models attempted 73 write-tool calls. **All 73 failed** — not because the models couldn't act, but because every call passed `namespace: "default"` while the workload lived in `otel-demo`. The tool schema said *"namespace (default from harness env)"* — hinting at a default without naming it. The models filled in the strongest prior from training data. Meanwhile the one model that passed the correct namespace went 48-for-48, and the models that *omitted* the parameter (falling back to the harness default) succeeded on every call.

Three lessons. Interface details — the exact wording of a tool schema — decided 73/73 outcomes. A model's "0% execution" turned out to be partly an interface bug, not a capability gap. And a framework that generates and reformats tool schemas to its own conventions would have made our most instructive finding invisible, while adding a schema dialect of its own we'd need to re-validate.

### 2. The execution substrate is an experimental variable

We accidentally ran the same model — Qwen3.8-27B — through two different serving gateways as two matrix slots. Same weights, identical prompts, faults, judge, and budget. The result: a **21–23 point swing** in detection, RCA, and execution rates between the gateways. The slow gateway exhausted the agent's round budget on exactly the faults that needed more rounds.

If serving infrastructure moves outcomes that much, so does an agent framework. Frameworks change tool-call formatting, retry semantics, loop boundaries — the substrate our numbers were produced on. Swap the loop for a framework and every number you've banked becomes non-comparable until you re-run the matrix. We would have been paying a re-validation study to rent abstractions.

### 3. Metrics live inside the loop

Our MTTD is stamped at the first detection claim that *survives gate-blocked rounds*. Our MTTR and our "remediation executed" flag share a single selector — `first_effective_remediation_write` — so they cannot diverge by construction. We got those definitions right only after finding **six metric bugs**, several of which came from a control layer printing diagnostics to stdout — corrupting the very channel the metrics parsed — and from parallel code paths for the same concept drifting apart.

Here's the uncomfortable part: framework callback, tracing, and telemetry layers are exactly this risk class, with more features. In a system where latency metrics are product features, hidden I/O between the model and the measurement is not an abstraction convenience. It's a liability.

### 4. Our control flow has no framework equivalent

Our execute gate blocks any final answer that claims an action without calling a tool — the models that "confabulate" (*"Restarted deployment 'cart'…"*, past tense, no tool call) get nudged up to three times with an explicit rejection of past-tense claims, then the gate yields and the run scores honestly as no-execution. Write tools carry guardrails validated in-run: patches may only override *observed* state (a hallucinated `CORRECT_VAR` once leaked onto a live deployment and scored as a remediation — that class of bug is now structurally impossible), no-op patches return errors, and 409 conflicts retry with backoff.

None of this exists in any framework we evaluated. In LangGraph you'd build it as custom nodes around bare LLM calls — at which point the framework is contributing a state-machine DSL for a graph that is already static and already implemented in our slot pipeline.

### 5. The loop is small

Median run: 18 tool rounds, capped at 20. The whole pattern — tools described in the prompt, parse the model's tool calls, execute, feed results back, gate, nudge — is roughly **300 lines of Python**. The frameworks we considered are thousands of lines that we would have to *audit anyway* for the safety properties we care about. "No framework" isn't not-invented-here pride; it's a smaller trusted codebase.

## Why the evaluation harness especially can't afford one

A benchmark harness is a measuring instrument. Its job is to be *more* controlled than the systems it measures. Every abstraction sitting between the model and the metric is an uncontrolled variable in an experiment whose entire purpose is control.

The substrate lesson cut deep here: for three batches, one of our slots labeled `maas_qwen36-27b` had actually been serving Qwen3.8-27B through a different gateway than the slot labeled `maas_qwen38-27b` — a label-versus-reality gap we only caught by verifying served models at preflight. Our runs now record **served model + gateway per agent** as first-class metadata. If we can't trust a label, we certainly can't trust a framework's upgrade notes to preserve tool-call formatting across versions.

There's also comparability over time. When we fixed the metric pipeline, we re-derived prior batches rather than mixing derivations. A framework version bump that changes how tool calls are formatted or parsed would force the same re-baselining across the whole matrix — for zero scientific gain.

## What "no framework" actually means here

We didn't reject dependencies; we rejected *renting the loop*. The stack:

| Concern | Choice |
|---|---|
| API, webhooks, UI | FastAPI |
| Response schemas, structured output | pydantic + our own JSON extraction and one repair retry |
| Multi-gateway LLM client | httpx, per-agent endpoint/env mapping |
| K8s tools, verifier, fault injection | the `kubernetes` Python client |
| Tool surface | FastMCP (`list_services`, `search_logs/traces/metrics`, `get_deployment_config`, gated writes) |
| Audit | MLflow — same experiment as the eval, so runtime numbers stay comparable |
| The agent loop | ~300 lines, ours |

And one hedge worth emphasizing: **MCP is our interop layer**. LangChain and Strands agents can consume our tools exactly as they are. Ecosystem at the edge, control at the core. If someone wants to prototype a Strands agent against our telemetry tomorrow, nothing blocks them — they just don't get to own the loop that writes to the cluster.

Note also what the serious coding agents — the ones people use every day — are built on: hand-rolled loops, not frameworks. Framework adoption clusters in product and chatbot territory. Evaluated, safety-gated control loops are a different species.

## When we'd switch

We keep this list explicit, because "no framework" is a position, not an identity:

- **The pipeline becomes dynamic** — agents choosing stage order, loops between stages → LangGraph's model earns its complexity.
- **Tool count explodes** into ecosystem territory — dozens of prebuilt integrations → framework integration catalogs pay off.
- **Team scale** — enough engineers on agent code that market-standard abstractions beat in-house conventions.
- **Durable, human-approval workflows at scale** → a workflow engine (Temporal-class), which is a different tool than an agent framework anyway.

None of those have fired. Until one does, the rule that got us here keeps holding: **the models are the experiment; the loop is the instrument.** You don't build the instrument out of something you can't control.

---

*Evidence for the numbers in this post: [Run A — Option-2 matrix, 288 cells](../research/agentic_aiops_architectures/docs/EVALUATION_RESULTS_22oct26.md) · [Run B — winnability matrix, 40 cells](../research/agentic_aiops_architectures/docs/EVALUATION_RESULTS_ccw_23oct26.md) · [October consolidated report](../research/agentic_aiops_architectures/docs/EVALUATION_RESULTS_Oct26.md).*
