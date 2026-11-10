# Evaluation Harness — Design Rationale and Learnings

**What this document is:** the design document for the AIOps agent evaluation harness — `code/harness/`, `code/agents/`, `code/tools/`, and the matrix runner scripts. It explains **why the harness is built the way it is**: every non-obvious design decision traces to a measured failure, confound, or scoring hole encountered between August and October 2026.

**Companions:** [Learnings_across_runs_report.md](Learnings_across_runs_report.md) (model-behavior learnings) · dated [EVALUATION_RESULTS_*.md](EVALUATION_RESULTS_Oct26.md) reports (evidence) · `.cursor/rules/evaluation-fairness.mdc` (change-control rule). Run-specific design notes (cell lists, batch parameters) live in the dated results reports, not here.

---

## 1. What the harness measures

Each **cell** is one trial on a live OpenShift cluster: inject a Kubernetes fault into a target workload, give an LLM agent a time window with telemetry/K8s/RAG tools, then score the outcome. A **batch** (matrix) is the cross-product of agents × fault types × context configurations.

| Metric | Definition | Invariant / note |
|---|---|---|
| `detected` | Agent reported detection within the window | |
| `rca_correct` | Right failing component identified | OR of declared · inferred · executed-target · oracle-judge layers |
| `remediation_correct` (OR) | Appropriate fix present in declared text | Execution not required |
| `remediation_executed` | A write that was **successful ∧ targeted at the expected component ∧ action-appropriate for the fault** | Selected by the single shared `first_effective_remediation_write` selector |
| `recovery_verified` | Harness verified the fault cleared **before** its own auto-recovery | Only the agent's own fix can flip this |
| `mttd_seconds` | Injection → the agent's own detection timestamp | Tolerant parser; harness clock only as fallback |
| `mttr_seconds` | Injection → first effective write completion | **Exists ⟺ `remediation_executed`**, by construction |
| `mttr_verified_seconds` | Injection → recovery verified | |
| `agent_error` | Harness-recorded agent/API failure for the cell | Reported alongside rates — never silently folded into "no detect" |

The core premise is **fairness**: a cross-agent comparison is only valid when agents share task framing, tools, scoring, and infra health. Everything below exists to protect that premise or to fix a violation of it.

## 2. Architecture

Four layers with defined boundaries (subprocess + JSON protocol; gated function calls):

```
BATCH   matrix runner script (e.g. run_c_matrix_scenario_a_c_permutations.sh)
        ├─ preflight: probe every agent API (abort batch on failure)
        ├─ per-cell infra refresh: oc login* → CH port-forward* → valkey config → health gate ×3
        └─ cell loop (agent → context → fault) with resume skip predicate (§5)
                              │  one subprocess per cell
                              ▼
CELL    code/harness/run_harness.py
        ├─ SETUP     run_id · MLflow run · event ledger
        ├─ INJECT    fault on the target workload
        ├─ SETTLE    15 s
        ├─ DETECT    polling loop inside a wall-clock window (§4)
        │              └─ agent subprocess, JSON-over-stdout protocol
        ├─ VERIFY    recovery check — BEFORE any harness recovery (§3, P6)
        ├─ RECOVER   harness undoes the fault
        ├─ SCORE     metrics + oracle judge (§1)
        └─ PERSIST   MLflow logging → result JSON write (atomic, final step)
                              │  stdout = result JSON, stderr = diagnostics
                              ▼
AGENT   code/agents/<name>/agent.py   (identical scaffolding for every agent)
        ├─ shared system prompt + AIOps skills (prompts.py, skills/)
        ├─ one tool profile for all agents (tool_profiles.py)
        ├─ forced RAG round-0 when context corpora are enabled
        ├─ execute gate: final JSON blocked while detected=true and no write yet
        └─ ≤ MAX_TOOL_ITERATIONS rounds
                              │  gated tool calls, each recorded
                              ▼
TOOLS   code/tools/agent_tools.py + tool_profiles.py
        ├─ reads:  ClickHouse telemetry · K8s state · deployment config · RAG corpora
        └─ writes: restart · scale · delete · patch  (gated + recorded + guardrailed)
```

| Component | Owns |
|---|---|
| Matrix runner | Cell ordering, infra health, API preflight, resume state, supervision entrypoint |
| Cell harness | Fault lifecycle, detection window, verification, scoring, persistence |
| Agent | LLM tool loop, result JSON — no per-agent customization beyond the model endpoint |
| Tools | Read/query + gated writes + write recording |
| Infrastructure | MLflow (metrics + per-round artifacts), ClickHouse (telemetry), OpenShift (fault target) |

## 3. Design principles — and the failures that produced them

### P1 — One task framing for every agent
Shared prompt module, shared skills, shared tool profile, shared gate, shared judge, shared iteration budget. The only per-agent variable is the model endpoint.
**Origin:** cloned agents drift — a copied agent that keeps its own prompt copy silently diverges from the lineup and invalidates comparisons. Codified as a change-control rule (`.cursor/rules/evaluation-fairness.mdc`): prompt/profile/scoring changes require documenting the invalidation of prior reports.

### P2 — Protocol discipline: stdout is data, stderr is diagnostics
The agent↔harness channel is a JSON-over-stdout protocol. Every diagnostic (gate nudges, RAG notices, usage reports) must go to stderr; the harness parses stdout only with the tolerant `parse_agent_signals` parser.
**Origin (Oct 10–12, 3 bugs):** the execute-gate nudge, the forced-RAG notice, and the context-window report all `print()`ed to stdout — the default output channel — corrupting the protocol on *every* run. The strict `json.loads` then failed silently and MTTD fell back to the subprocess-exit clock, inflating one agent's MTTD by ~60 s (exit clock includes post-loop MLflow uploads). Six derivation bugs total, fixed 19 Oct by replay over recorded data — [EVALUATION_RESULTS_12oct26_corrected.md](EVALUATION_RESULTS_12oct26_corrected.md).

### P3 — One selector per concept
`remediation_executed` and `mttr_seconds` are derived from the *same* `first_effective_remediation_write` selector, so **MTTR exists if and only if exec is true** — by construction, not by testing.
**Origin:** MTTR was originally keyed on *any* successful write while exec required *targeted + action-appropriate* — two parallel code paths for one concept that drifted apart, producing phantom MTTR (present with exec=false) and medians skewed late by wrong-target writes.

### P4 — Agent-authored timestamps, not harness clocks
MTTD uses the agent's own `detection_timestamp`: the first mid-loop `detected=true` claim that survives gate-blocked rounds, else the first successful write.
**Origin:** detection was stamped only at the first successful *write* — for declare-then-act agents, detection ≡ remediation; for act-first agents, claims were never stamped at all. Combined with P2's exit-clock fallback, latency columns were untrustworthy across agents.

### P5 — Bounded wall-clock everywhere
The detection window is a wall-clock deadline; the agent subprocess can never outlive it; polls that cannot usefully complete are skipped.

```
W = detection window    A = agent budget     P = poll interval
u = min useful remaining = max(60, min(0.35·A, 300))

while remaining > 0:
    if remaining < u: stop                     # a guaranteed-timeout poll is noise
    sub_timeout = min(A, remaining)
    run agent subprocess (killed at sub_timeout)
    if detected: break
    sleep P
```

**Origin:** the loop originally counted poll intervals while a single agent invocation could run to its full timeout — total wait was unbounded. Separately, qwen36-27b lost 32/48 cells to a 600 s agent budget (Sep 22–23), setting the current `AGENT_TIMEOUT=900` / window 1200 s. The sub-900 s "timeout after Ns" errors seen since are the *window cap* working as designed — the remaining-window clamp, not a new bug.

### P6 — Credit only what the agent did
Recovery is verified **before** the harness auto-recovers the fault, so platform self-healing cannot flip `recovery_verified`. Every write-tool call is recorded (tool, arguments, timestamps, success) and scoring consumes those records.
**Origin:** `kill_pod` is largely self-healing — the deployment controller recreates the pod — so unguarded "recovery" measurement credits the platform, not the agent. The same records exposed that one agent's Oct 10–12 writes targeted a *crash-looping kafka decoy* rather than the injected fault (see P9).

### P7 — Writes are dangerous: guardrail on every generative write tool
A write tool whose arguments are free-form LLM strings must constrain its effects to *observed* state:
- **override-only `set_env`** — may change env vars that already exist; refuses to add new ones;
- **no-op rejection** — a patch that changes nothing is a *failed* write, not a success;
- **conflict retry** — read-modify-write retried on 409 (a prior patch's rollout bumps `resourceVersion`).

**Origin (Oct 22):** an agent hallucinated `set_env={"CORRECT_VAR": "value"}` — the junk variable was *added to the live deployment*, survived the harness's fault recovery (which only removes vars it knows), contaminated subsequent cells, and scored as a successful remediation. Hallucinated arguments must not be able to create cluster state or earn exec credit.

### P8 — A fault must be discoverable and fixable by design
Every injected fault needs (a) an observable signal, (b) a discovery path through the read tools, and (c) a fix path through the write tools. A fault failing any of these measures the harness, not the agents.
**Origin:** `config_corruption` injects a bad env var into the deployment *spec*; the app never reads it (pod stays Running, nothing in the logs), and the write tools (restart/scale/delete) recreate pods with the same corrupt spec. Result: 0/96 recovery across two batches — structurally impossible, invisible to agents (0/96 outputs referenced it), and silently capping the whole matrix's recovery ceiling. Fixed with a deployment-config read tool and a spec-patching write tool; verified winnable afterwards (16/40 in the follow-up matrix). The same design question is open for `memory_limit` and `readiness_probe_fail`.

### P9 — The environment is part of the experiment
Cluster health and telemetry volume are first-class confounds: they must be gated (pre-run health checks), recorded (run notes), and reported alongside results — including environment-drift tables when comparing batches. Endpoint reliability is separated from capability via error-adjusted (clean-run) rates.
**Origin:** the Oct 10–12 batch ran with a crash-looping kafka broker (decoy error signals) and a starved load-generator (~2–25 checkout spans/hour vs ~1,300–3,200 healthy). Act-first agents chased the decoy (glm's writes: kafka 59 / cart 17 → with a clean cluster: kafka 0 / cart 34) — tens of percentage points of apparent "capability" swing. Separately, one agent labeled `qwen36-27b` turned out to serve **Qwen3.8-27B** on a different, faster platform than `qwen38-27b` — model identity must be verified against the serving endpoint (`/v1/models`), not the agent's name.

### P10 — Progress must be atomic, deterministic, and supervisable
Cell completion is one atomic JSON write (the final statement of the cell program); cell identity is a deterministic file path; batches resume via a skip predicate; long batches run detached from any interactive session (systemd user unit preferred, `setsid` minimum).
**Origin:** multi-day batches died to (a) concurrent `oc login` writes tearing `~/.kube/config` mid-read (Oct 10–12 and two Oct 20 aborted kickoffs — fixed with a long-lived ServiceAccount, a read-only dedicated kubeconfig for matrix children, and login-only-when-dead), (b) a health-gate pod-scheduling race (fixed with 3× retry), and (c) a batch launched as a session background shell being reaped when the interactive session refreshed (Oct 22 — the resume model recovered it with no gaps or duplicates, and supervision was moved out of the session).

## 4. Cell lifecycle

| Step | What happens | Notes |
|---|---|---|
| 0. Infra refresh | login only if kubeconfig dead; port-forward only if `SELECT 1` fails; valkey `stop-writes-on-bgsave-error no`; health gate ≤ 3× | abort batch on 3rd gate failure |
| 1. Setup | `run_id`, MLflow run, event ledger | ledger: `fault_injection` → `first_alert` → `fault_recovery` |
| 2. Inject | fault-type-specific mechanism on the target workload | |
| 3. Settle | 15 s | let telemetry reflect the fault |
| 4. Detect | polling loop (§3 P5 math) | agent subprocess per poll |
| 5. Verify | if the agent wrote anything: poll fault resolution ≤ 90 s, **before** auto-recovery | P6 |
| 6. Recover | harness undoes the fault (~90 s after injection if the agent hasn't) | cluster clean for the next cell |
| 7. Score | metrics from agent signals + write records + judge | P3, P4 |
| 8. Persist | MLflow metrics + artifacts (prompts, tool calls, per-round data), then the result JSON — **last statement, single write** | P10 |

## 5. Resume model

```
   PENDING --( skip: JSON exists AND mtime >= kickoff )--------> DONE
      |
      | run cell
      v
   RUNNING --( result JSON written - final statement )---------> DONE
      |
      | killed / failed BEFORE the write  (no JSON exists)
      v
   PENDING   =>  whole cell re-runs on the next launch
```

| Property | Rule |
|---|---|
| Completion marker | The result JSON is the last statement of the cell program. Killed anywhere earlier ⇒ no file ⇒ re-run. |
| Cell identity | One deterministic output path per (agent × fault × context). |
| Skip predicate | `MATRIX_RESUME=1` ∧ file exists ∧ `mtime ≥ kickoff epoch` — the mtime term excludes stale files from earlier attempts. |
| Failed cells | Write no JSON ⇒ retried on the next resume (retry-safe). |
| Fresh start | Without `MATRIX_RESUME`, a new kickoff timestamp is written and nothing is skipped. Missing kickoff + resume degrades to a fresh start — wasteful, never incorrect. |
| Invalidation | A completed-but-invalid cell is invalidated by renaming its JSON; the missing path forces a re-run. |

**Deliberate limit:** resume guarantees step-level correctness (no cell skipped, none double-counted, no partial results) but not cluster-state hygiene — a mid-cell kill can leave a fault injected or agent-written state behind. The operational procedure is: scrub the fault target, mark orphaned MLflow runs FAILED, relaunch with `MATRIX_RESUME=1`.

## 6. Supervision model

| Option | Survives session churn | Dependency ordering | Role |
|---|---|---|---|
| Session background shell | **No** — child of the session's process runner | none | interactive/short commands only |
| Detached process (`setsid nohup`) | Yes | none | ad-hoc batches |
| **systemd user unit** | Yes | `After=`/`Wants=` MLflow + ClickHouse port-forward; dedicated KUBECONFIG | **matrix runs** |

The matrix scripts are supervision-agnostic — the same script and resume semantics run under any of the three. The systemd unit is the production pattern (see `aiops-c-matrix-sa-c-permutations.service`).

## 7. Chronological learning log

| When | Incident | Root cause | Design change |
|---|---|---|---|
| Aug 11 | Phantom MTTR (values with exec=false); weak detection across lineup | early derivation keyed on any write | (later superseded by P3) |
| Sep 14 | Aborted 177-run attempt, undocumented | infra abort mid-matrix | resume model + documented run notes become standard |
| Sep 22–23 | qwen36-27b 32/48 timeouts; llama31-70b 1/48 detect; 203 runs detected-but-never-executed | agent budget too small; dead endpoint; execution gap | `AGENT_TIMEOUT` 600→900, window 720→1200; llama dropped; skills experiment (Sep 25) and later the execute gate |
| Sep 25–27 | `score_remediation` crash on nested `suggested_remediations` (one unscored cell); MLflow orphan run | `" ".join` on list-of-lists | flatten before join (shared gate module); orphans documented, matched by `run_id` |
| Sep 25–27 | RCA +25 pp after changing skills + corpus + timeout together | confounded experiment axes | axes must be isolated and documented (run-notes convention) |
| Oct 10–12 | kafka crash-loop + starved telemetry; act-first agents remediated the decoy | environment degradation unrecorded | environment gates + drift tables + write-target forensics (P9) |
| Oct 10–12 | MTTD inflated (~60 s for one agent), phantom/divergent MTTR, detection ≡ write | 6 derivation bugs: stdout pollution ×3, strict parse + silent fallback, write-only stamping, dual selectors | P2–P4 fixes, applied 19 Oct and validated by replay ([corrected report](EVALUATION_RESULTS_12oct26_corrected.md)) |
| Oct 10–12, Oct 20 | kubeconfig torn mid-read; two aborted kickoffs on OAuth 401s | concurrent `oc login` writes | long-lived ServiceAccount, read-only dedicated kubeconfig for matrix children, login-only-when-dead |
| Oct 20 | health-gate abort on a pod-scheduling race | transient capacity race | health gate 3× retry; PriorityClass on fault targets |
| Oct 20–22 | `config_corruption` recovery 0/96; corrupted env invisible to every agent | spec-level fault with no read/write path (P8) | deployment-config read tool + spec-patch write tool; verified winnable Oct 22–23 |
| Oct 22 | Hallucinated `set_env` added a junk var to the live cluster and scored exec | unconstrained generative write tool | P7 guardrails (override-only, no-op rejection, conflict retry) |
| Oct 22 | Batch killed mid-cell by session refresh | session-scoped supervision | P10: systemd/setsid supervision; resume recovered the batch |

## 8. Limitations and open items

- `restart`/`scale` count as action-appropriate writes for spec-level faults even though they cannot recover them — exec measures *intent*, recovery measures *effectiveness*; keep reading them as separate signals.
- `kill_pod` recovery still partially measures the platform even with pre-recovery verification (the controller may recreate the pod during the verification window). Open: score against an agent-controlled criterion.
- `memory_limit` / `readiness_probe_fail` are spec-level faults with the same P8 problem as `config_corruption` — candidates for the patch-tool path.
- The oracle-judge pool is `{deepseek-r1-distill-qwen-14b, qwen3-14b}`, cross-paired so no agent is judged by its own model (deepseek ⇄ qwen3 judge each other). Both judge models run on the shared RHDP platform alongside three matrix agents — resource contention during batches, not scoring bias.
- Agent labels can outlive the model they point to (`maas_qwen36-27b` served Qwen3.8-27B). **Resolved 23 Oct 26:** labels now embed the serving infra and exact model (`{infra}_{model}`, e.g. `litemaas_qwen3.8-27b`); the analyzer maps pre-rename ids found in historical data, and lineup identity should still be verified from `/v1/models` when endpoints change.
- Pre-22-Oct latency figures (Aug/Sep/early Oct) were produced by the buggy pipeline; re-derive before any cross-month trend claim.
