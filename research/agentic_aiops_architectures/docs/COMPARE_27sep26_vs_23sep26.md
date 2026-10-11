# Option 2 kickoff compare

- Baseline: `2026-09-22 00:02:00` → `2026-09-25 14:06:25` (n=336)
- New: `2026-09-25 14:06:25` → `now` (n=335)

Columns: detect / RCA / exec / recovery (baseline%→new% (Δ)) then agent_error counts.

```
ALL                    |    73%→   79% (+6) |    34%→   59% (+24) |    13%→   25% (+12) |    12%→   22% (+9) | err 41/336→21/335
maas_deepseek          |    94%→  100% (+6) |    85%→   94% (+8) |     0%→    0% (+0) |     0%→    0% (+0) | err 0/48→0/48
maas_glm-53-flash      |   100%→   98% (-2) |    38%→   71% (+33) |    33%→   62% (+29) |    33%→   54% (+21) | err 0/48→1/48
maas_granite           |    88%→  100% (+12) |    44%→   96% (+52) |     0%→    0% (+0) |     0%→    0% (+0) | err 1/48→0/48
maas_llama31-70b       |     0%→    2% (+2) |     0%→    0% (+0) |     0%→    0% (+0) |     0%→    0% (+0) | err 7/48→1/48
maas_qwen3             |   100%→   98% (-2) |    25%→   31% (+6) |     8%→    2% (-6) |     6%→    2% (-4) | err 0/48→1/48
maas_qwen38-27b         |    98%→   79% (-19) |    15%→   38% (+24) |    15%→   32% (+17) |    15%→   30% (+15) | err 1/48→10/47
maas_qwen36-27b        |    33%→   79% (+46) |    33%→   79% (+46) |    33%→   77% (+44) |    33%→   67% (+33) | err 32/48→8/48
```

_Matrix complete: 335/336 scored (1 harness crash: C2-3 × kill_pod × qwen38-27b). Rates MLflow-verified 2026-09-28 — see EVALUATION_RESULTS_27sep26.md._

