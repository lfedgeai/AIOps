# Context Engineering Evaluation Report

*Generated 2026-07-06 15:23 UTC — 108 runs*

## Executive summary

- **Matrix**: agents × 3 faults × single CONTEXT_C corpora (C1–C4; C0 optional).
- **Corpora**: C1=source, C2=docs, C3=architecture, C4=dependencies.
- **compare_telemetry**: always-on for telemetry profiles (not a corpus).

### RCA by context corpus

| Context | RCA accuracy | MTTD median |
|---------|--------------|-------------|
| C1 | 3/24 (12%) | 170s |
| C2 | 4/24 (17%) | 88s |
| C3 | 3/24 (12%) | 80s |
| C4 | 4/24 (17%) | 157s |
| C0 | 0/12 (0%) | 153s |

### RCA by fault

| Fault | RCA accuracy |
|-------|--------------|
| config_corruption | 4/36 (11%) |
| kill_pod | 4/36 (11%) |
| scale_zero | 6/36 (17%) |

## Per-run detail

| File | C | Agent | Scenario | Fault | RCA | Remediation |
|------|---|-------|----------|-------|-----|-------------|
| context_L0_s0_config_corruption_nemotron.json | C0 | - | 0 | config_corruption | False | False |
| context_L0_sa_config_corruption_nemotron.json | C0 | - | a | config_corruption | False | False |
| context_L0_sb_config_corruption_nemotron.json | C0 | - | b | config_corruption | False | False |
| context_L0_sc_config_corruption_nemotron.json | C0 | - | c | config_corruption | False | False |
| context_L0_s0_kill_pod_nemotron.json | C0 | - | 0 | kill_pod | False | False |
| context_L0_sa_kill_pod_nemotron.json | C0 | - | a | kill_pod | False | False |
| context_L0_sb_kill_pod_nemotron.json | C0 | - | b | kill_pod | False | False |
| context_L0_sc_kill_pod_nemotron.json | C0 | - | c | kill_pod | False | False |
| context_L0_s0_scale_zero_nemotron.json | C0 | - | 0 | scale_zero | False | False |
| context_L0_sa_scale_zero_nemotron.json | C0 | - | a | scale_zero | False | False |
| context_L0_sb_scale_zero_nemotron.json | C0 | - | b | scale_zero | False | False |
| context_L0_sc_scale_zero_nemotron.json | C0 | - | c | scale_zero | False | False |
| context_C1_sa_config_corruption_maas_deepseek.json | C1 | - | a | config_corruption | False | False |
| context_C1_sa_config_corruption_maas_llama-scout.json | C1 | - | a | config_corruption | True | True |
| context_C1_sa_config_corruption_maas_qwen3.json | C1 | - | a | config_corruption | False | False |
| context_C1_sa_config_corruption_nemotron-nano-3.json | C1 | - | a | config_corruption | False | False |
| context_L1_s0_config_corruption_nemotron.json | C1 | - | 0 | config_corruption | False | False |
| context_L1_sa_config_corruption_nemotron.json | C1 | - | a | config_corruption | False | False |
| context_L1_sb_config_corruption_nemotron.json | C1 | - | b | config_corruption | False | False |
| context_L1_sc_config_corruption_nemotron.json | C1 | - | c | config_corruption | False | False |
| context_C1_sa_kill_pod_maas_deepseek.json | C1 | - | a | kill_pod | False | False |
| context_C1_sa_kill_pod_maas_llama-scout.json | C1 | - | a | kill_pod | True | True |
| context_C1_sa_kill_pod_maas_qwen3.json | C1 | - | a | kill_pod | False | False |
| context_C1_sa_kill_pod_nemotron-nano-3.json | C1 | - | a | kill_pod | False | False |
| context_L1_s0_kill_pod_nemotron.json | C1 | - | 0 | kill_pod | False | False |
| context_L1_sa_kill_pod_nemotron.json | C1 | - | a | kill_pod | False | False |
| context_L1_sb_kill_pod_nemotron.json | C1 | - | b | kill_pod | False | False |
| context_L1_sc_kill_pod_nemotron.json | C1 | - | c | kill_pod | False | False |
| context_C1_sa_scale_zero_maas_deepseek.json | C1 | - | a | scale_zero | False | True |
| context_C1_sa_scale_zero_maas_llama-scout.json | C1 | - | a | scale_zero | True | False |
| context_C1_sa_scale_zero_maas_qwen3.json | C1 | - | a | scale_zero | False | False |
| context_C1_sa_scale_zero_nemotron-nano-3.json | C1 | - | a | scale_zero | False | False |
| context_L1_s0_scale_zero_nemotron.json | C1 | - | 0 | scale_zero | False | False |
| context_L1_sa_scale_zero_nemotron.json | C1 | - | a | scale_zero | False | False |
| context_L1_sb_scale_zero_nemotron.json | C1 | - | b | scale_zero | False | False |
| context_L1_sc_scale_zero_nemotron.json | C1 | - | c | scale_zero | False | True |
| context_C2_sa_config_corruption_maas_deepseek.json | C2 | - | a | config_corruption | False | False |
| context_C2_sa_config_corruption_maas_llama-scout.json | C2 | - | a | config_corruption | False | False |
| context_C2_sa_config_corruption_maas_qwen3.json | C2 | - | a | config_corruption | False | False |
| context_C2_sa_config_corruption_nemotron-nano-3.json | C2 | - | a | config_corruption | False | False |
| context_L2_s0_config_corruption_nemotron.json | C2 | - | 0 | config_corruption | False | False |
| context_L2_sa_config_corruption_nemotron.json | C2 | - | a | config_corruption | False | False |
| context_L2_sb_config_corruption_nemotron.json | C2 | - | b | config_corruption | False | False |
| context_L2_sc_config_corruption_nemotron.json | C2 | - | c | config_corruption | False | False |
| context_C2_sa_kill_pod_maas_deepseek.json | C2 | - | a | kill_pod | False | False |
| context_C2_sa_kill_pod_maas_llama-scout.json | C2 | - | a | kill_pod | True | True |
| context_C2_sa_kill_pod_maas_qwen3.json | C2 | - | a | kill_pod | False | False |
| context_C2_sa_kill_pod_nemotron-nano-3.json | C2 | - | a | kill_pod | False | False |
| context_L2_s0_kill_pod_nemotron.json | C2 | - | 0 | kill_pod | False | False |
| context_L2_sa_kill_pod_nemotron.json | C2 | - | a | kill_pod | False | False |
| context_L2_sb_kill_pod_nemotron.json | C2 | - | b | kill_pod | False | False |
| context_L2_sc_kill_pod_nemotron.json | C2 | - | c | kill_pod | False | False |
| context_C2_sa_scale_zero_maas_deepseek.json | C2 | - | a | scale_zero | False | False |
| context_C2_sa_scale_zero_maas_llama-scout.json | C2 | - | a | scale_zero | True | True |
| context_C2_sa_scale_zero_maas_qwen3.json | C2 | - | a | scale_zero | False | False |
| context_C2_sa_scale_zero_nemotron-nano-3.json | C2 | - | a | scale_zero | True | True |
| context_L2_s0_scale_zero_nemotron.json | C2 | - | 0 | scale_zero | False | False |
| context_L2_sa_scale_zero_nemotron.json | C2 | - | a | scale_zero | False | False |
| context_L2_sb_scale_zero_nemotron.json | C2 | - | b | scale_zero | False | False |
| context_L2_sc_scale_zero_nemotron.json | C2 | - | c | scale_zero | True | True |
| context_C3_sa_config_corruption_maas_deepseek.json | C3 | - | a | config_corruption | True | False |
| context_C3_sa_config_corruption_maas_llama-scout.json | C3 | - | a | config_corruption | False | False |
| context_C3_sa_config_corruption_maas_qwen3.json | C3 | - | a | config_corruption | False | False |
| context_C3_sa_config_corruption_nemotron-nano-3.json | C3 | - | a | config_corruption | False | False |
| context_L3_s0_config_corruption_nemotron.json | C3 | - | 0 | config_corruption | False | False |
| context_L3_sa_config_corruption_nemotron.json | C3 | - | a | config_corruption | False | False |
| context_L3_sb_config_corruption_nemotron.json | C3 | - | b | config_corruption | False | False |
| context_L3_sc_config_corruption_nemotron.json | C3 | - | c | config_corruption | False | False |
| context_C3_sa_kill_pod_maas_deepseek.json | C3 | - | a | kill_pod | False | False |
| context_C3_sa_kill_pod_maas_llama-scout.json | C3 | - | a | kill_pod | True | True |
| context_C3_sa_kill_pod_maas_qwen3.json | C3 | - | a | kill_pod | False | False |
| context_C3_sa_kill_pod_nemotron-nano-3.json | C3 | - | a | kill_pod | False | False |
| context_L3_s0_kill_pod_nemotron.json | C3 | - | 0 | kill_pod | False | False |
| context_L3_sa_kill_pod_nemotron.json | C3 | - | a | kill_pod | False | False |
| context_L3_sb_kill_pod_nemotron.json | C3 | - | b | kill_pod | False | False |
| context_L3_sc_kill_pod_nemotron.json | C3 | - | c | kill_pod | False | False |
| context_C3_sa_scale_zero_maas_deepseek.json | C3 | - | a | scale_zero | True | False |
| context_C3_sa_scale_zero_maas_llama-scout.json | C3 | - | a | scale_zero | False | False |
| context_C3_sa_scale_zero_maas_qwen3.json | C3 | - | a | scale_zero | False | False |
| context_C3_sa_scale_zero_nemotron-nano-3.json | C3 | - | a | scale_zero | False | False |
| context_L3_s0_scale_zero_nemotron.json | C3 | - | 0 | scale_zero | False | False |
| context_L3_sa_scale_zero_nemotron.json | C3 | - | a | scale_zero | False | False |
| context_L3_sb_scale_zero_nemotron.json | C3 | - | b | scale_zero | False | False |
| context_L3_sc_scale_zero_nemotron.json | C3 | - | c | scale_zero | False | False |
| context_C4_sa_config_corruption_maas_deepseek.json | C4 | - | a | config_corruption | False | False |
| context_C4_sa_config_corruption_maas_llama-scout.json | C4 | - | a | config_corruption | True | True |
| context_C4_sa_config_corruption_maas_qwen3.json | C4 | - | a | config_corruption | False | False |
| context_C4_sa_config_corruption_nemotron-nano-3.json | C4 | - | a | config_corruption | True | False |
| context_L4_s0_config_corruption_nemotron.json | C4 | - | 0 | config_corruption | False | False |
| context_L4_sa_config_corruption_nemotron.json | C4 | - | a | config_corruption | False | False |
| context_L4_sb_config_corruption_nemotron.json | C4 | - | b | config_corruption | False | False |
| context_L4_sc_config_corruption_nemotron.json | C4 | - | c | config_corruption | False | False |
| context_C4_sa_kill_pod_maas_deepseek.json | C4 | - | a | kill_pod | False | False |
| context_C4_sa_kill_pod_maas_llama-scout.json | C4 | - | a | kill_pod | False | False |
| context_C4_sa_kill_pod_maas_qwen3.json | C4 | - | a | kill_pod | False | False |
| context_C4_sa_kill_pod_nemotron-nano-3.json | C4 | - | a | kill_pod | True | True |
| context_L4_s0_kill_pod_nemotron.json | C4 | - | 0 | kill_pod | False | False |
| context_L4_sa_kill_pod_nemotron.json | C4 | - | a | kill_pod | False | False |
| context_L4_sb_kill_pod_nemotron.json | C4 | - | b | kill_pod | False | False |
| context_L4_sc_kill_pod_nemotron.json | C4 | - | c | kill_pod | False | False |
| context_C4_sa_scale_zero_maas_deepseek.json | C4 | - | a | scale_zero | False | False |
| context_C4_sa_scale_zero_maas_llama-scout.json | C4 | - | a | scale_zero | True | False |
| context_C4_sa_scale_zero_maas_qwen3.json | C4 | - | a | scale_zero | False | False |
| context_C4_sa_scale_zero_nemotron-nano-3.json | C4 | - | a | scale_zero | False | False |
| context_L4_s0_scale_zero_nemotron.json | C4 | - | 0 | scale_zero | False | False |
| context_L4_sa_scale_zero_nemotron.json | C4 | - | a | scale_zero | False | False |
| context_L4_sb_scale_zero_nemotron.json | C4 | - | b | scale_zero | False | False |
| context_L4_sc_scale_zero_nemotron.json | C4 | - | c | scale_zero | False | True |

---

MLflow: `http://localhost:5050` — filter by tag `context_c`.

