# Context Engineering Evaluation Report

*Generated 2026-07-25 — Option 1 re-run only (75 runs, mtime ≥ 2026-07-24 22:50)*  
*Full narrative: [EVALUATION_RESULTS_25July26.md](EVALUATION_RESULTS_25July26.md)*

## Executive summary

- **Matrix**: scenario A × C0–C4 × 5 agents × 3 faults; forced RAG on C1–C4.
- **Corpora** (non-empty): C1=source (1784 chunks), C2=docs (130), C3=architecture (129), C4=dependencies (166).
- **Empty-corpus aborts**: 0 (gate armed; indexes valid).
- **compare_telemetry**: available in scenario A; used in 33/75 runs.

### Detection / RCA by context

| Context | Detect | RCA | Remediation | MTTD median |
|---------|--------|-----|-------------|-------------|
| C0 | 11/15 (73%) | 6/15 (40%) | 6/15 (40%) | 64s |
| **C1** | **12/15 (80%)** | 9/15 (60%) | **10/15 (67%)** | 82s |
| C2 | 10/15 (67%) | 9/15 (60%) | 8/15 (53%) | 65s |
| **C3** | 11/15 (73%) | **10/15 (67%)** | 8/15 (53%) | 96s |
| C4 | 8/15 (53%) | 7/15 (47%) | 6/15 (40%) | 120s |

### By fault

| Fault | Detect | RCA | Remediation |
|-------|--------|-----|-------------|
| scale_zero | 18/25 (72%) | 14/25 (56%) | 12/25 (48%) |
| config_corruption | 18/25 (72%) | 14/25 (56%) | 13/25 (52%) |
| kill_pod | 16/25 (64%) | 13/25 (52%) | 13/25 (52%) |

### By agent

| Agent | Detect | RCA | Remediation |
|-------|--------|-----|-------------|
| nemotron-nano-3 | 14/15 (93%) | 13/15 (87%) | 13/15 (87%) |
| maas_deepseek | 13/15 (87%) | 12/15 (80%) | 9/15 (60%) |
| maas_qwen3 | 13/15 (87%) | 9/15 (60%) | 9/15 (60%) |
| maas_llama-scout | 6/15 (40%) | 2/15 (13%) | 2/15 (13%) |
| maas_gpt-oss-120b | 6/15 (40%) | 5/15 (33%) | 5/15 (33%) |

## Takeaways

1. With fixed C3/C4 corpora, **C1 leads detection/remediation; C3 leads RCA** — C0 is no longer “best.”
2. Forced RAG fired on **60/60** C1–C4 runs; no `corpus_empty`.
3. Agent skill still dominates context choice.

## Regenerate

```bash
# Narrative report is hand-written for the Option 1 slice:
# docs/EVALUATION_RESULTS_25July26.md
#
# Broader auto table (mixes historical out/ files):
python scripts/analyze_context_matrix.py --out-dir out --report docs/CONTEXT_ENGINEERING_EVAL_REPORT.md
```
