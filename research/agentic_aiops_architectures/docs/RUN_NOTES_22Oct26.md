# Run Notes — 22 Oct 26 (Option 2 permutations, 288 cells, fixed metrics)

**Kickoff:** 2026-10-20 23:44 AEDT (resumed after two aborted starts same day)
**Completed:** 2026-10-22 17:22 AEDT — **288/288, failures=0, skipped=4** (4 cells from the first
aborted kickoff were resumed, not re-run)
**Report:** [EVALUATION_RESULTS_22oct26.md](EVALUATION_RESULTS_22oct26.md)

First full matrix run on the **fixed MTTD/MTTR metric pipeline** (see
[EVALUATION_RESULTS_12oct26_corrected.md](EVALUATION_RESULTS_12oct26_corrected.md) for the bug
analysis). MTTR now exists exactly when a targeted + action-appropriate write executed
(n = exec count, verified: 44/27/16 per executor agent = 87 total), and MTTD uses the agent's
own detection timestamp via the tolerant parser.

## Deliberate changes vs the Oct 10–12 batch (interpretation caveats)

| Change | Oct 10–12 | This run | Direction of effect |
|--------|-----------|----------|---------------------|
| **MTTD/MTTR metric fixes** | buggy (exit-clock MTTD, any-write MTTR) | fixed | Latency columns now cross-agent comparable; MTTR n = exec count |
| **stdout protocol hardening** | `[execute_gate]`/`[rag_force]`/`[context window]` lines polluted agent JSON stdout | all on stderr | Robust parse; first-claim detection stamping added |
| **kafka broker health** | **crash-looping** (~122 lifetime pod restarts; checkout errored 42% — 684/1628 spans) | **healthy** (KRaft quorum env fix; 0 restarts) | October telemetry contained loud decoy errors pointing at kafka/checkout; eager executors chased them |
| **Telemetry volume (load-generator traffic)** | **~2–25 checkout spans/hour** (starved — fault signal nearly invisible) | **~1,300–3,200/hour** (fresh load-generator) | 100–1000× more traffic: every injected fault produces an immediate, unmistakable checkout→cart error burst |
| `max_tool_iterations` | 20 (median rounds used: 18) | 20 | **Not a confound** — the yaml bump predates the Oct batch |
| Demo services | full set | full set by batch time (ad/image-provider/quote restored after a transient Wednesday scale-down; observability UIs + postgresql/accounting/fraud-detection remain down) | Minor |
| **Cluster auth** | kubeadmin OAuth (kubeconfig corruption mid-run) | long-lived `aiops-matrix` ServiceAccount (read-only kubeconfig, priority class on cart/valkey) | Zero auth-related aborts |
| **ACM stack** | present (~51 pods) | removed before this run | More node headroom; no effect on telemetry |
| Prompts / skills / faults / judge / forced-RAG / execute gate / model lineup | — | **unchanged** (same judge: deepseek-r1-distill-qwen-14b, 48/48 runs both batches) | — |

### Why glm-53-flash jumped (48%→98% RCA, 35%→92% exec) — write-targeting evidence

- **Oct 10–12:** 122 writes → **kafka 59, checkout 10, cart 17**. The sick kafka made
  kafka/checkout the loudest error signal; act-first glm remediated the decoy.
  → exec 17/48, RCA 23/48 (inferred 17).
- **Oct 20–22:** 48 writes → **cart 34, valkey-cart 13, kafka 0**. Healthy kafka + 100×
  traffic means the only loud signal is the injected cart fault.
  → 47 runs with a targeted successful write, exec 44, RCA 47 (inferred 47, declared 26).
- Scoring nuance: `valkey-cart` writes count as cart-targeted (substring match in
  `write_targeted_expected`).
- Conclusion: glm's capability did not change — the October environment (decoy noise +
  starved fault signal) specifically punished act-first agents. Diagnosers (deepseek
  85%→90% RCA) were far less affected, which is why the jump is agent-specific.

## Slot-identity finding (verified 23 Oct 26)

`maas_qwen36-27b` and `maas_qwen38-27b` are the **same model** (Qwen 3.8 27B) on two
different gateways — the "qwen36" name is a misnomer (its env/code default/gateway have
pointed at `Qwen3.8-27B` since creation; litemaas serves no Qwen 3.6). Verified via MLflow
round records (`Qwen3.8-27B` vs `qwen38-27b`), `config/.env`, and both gateways' `/models`
inventories. **Report labeling (EVALUATION_RESULTS_22oct26.md and future reports):** the
slots are labeled by what they actually ran — `qwen3.8-27b_maas-litemaas` (raw id
`maas_qwen36-27b`) and `qwen3.8-27b_maas-rhdp` (raw id `maas_qwen38-27b`); raw JSONs and
MLflow keep the original approach ids. The report carries a dedicated
**"Serving infrastructure experiment"** section with the head-to-head. Implications: the
6-agent lineup is 5 distinct models; the same-model A/B shows gateway effects (litemaas
outperformed maas-rhdp: 56% vs 33% exec, 14 vs 21 timeouts — serving-stack effects, not
model differences); and cross-batch trend lines treating these slots as different models
(Sep/Oct longitudinal comparisons) need this caveat. See the slot-identity note in the
longitudinal report for the history of qwen-slot relabeling (Sep "Qwen38" was actually
`qwen35-9b`).

## Notable results (see report for full tables)

- **glm-53-flash: 98% detect / 98% RCA / 92% exec** (Oct: 94/48/35) — first near-complete
  detect→diagnose→remediate agent in the series; MTTR med 112s (n=44).
- **Exec 87/288 (30%)** vs 50/288 (17%) in Oct — concentrated in glm + qwen36/38.
- **C0 context cliff persisted but softened:** C0 RCA 33% vs 61–83% non-empty (Oct: 17% vs 61–72%).
- **`agent_error` rose 21→39** (qwen38 21, qwen36 14) — more API failures at 20 iterations/run.
- deepseek/granite/qwen3 remain diagnosers (0/48 exec) with RCA 90/85/71%.

## Infra events during the window

- Two aborted kickoffs on 20 Oct (OAuth token-churn 401s; kubeconfig torn write) before the
  SA-token + read-only-kubeconfig fix.
- `aiops-fault-target` PriorityClass added to cart/valkey-cart after a pod-scheduling race
  aborted a start; health gate now retries 3×.
- The 4 resumed cells (deepseek C0/C1 from kickoff 1) were written before the ACM removal;
  no measurable impact (see corrected-metrics sanity checks in the report's MLflow section).
