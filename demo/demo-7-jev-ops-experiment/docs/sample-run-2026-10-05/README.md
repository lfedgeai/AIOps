# Sample run: 2026-10-05, Tier 1, four engines

This is a sample run. We published it on purpose so that readers can check the figures in [../results-2026-10-05-tier1-baselines.md](../results-2026-10-05-tier1-baselines.md). Normal runs write to `results/`, which is not in git.

| File | What it is |
|---|---|
| `20261005-224157-70runs.jsonl` | Jev, 70 runs |
| `20261005-224548-70runs-claude-sonnet.jsonl` | Claude Sonnet 5.5 through `claude -p`, 70 runs |
| `20261005-224835-70runs-claude-opus.jsonl` | Claude Opus 5.5 through `claude -p`, 70 runs |
| `20261005-230609-70runs-ollama-qwen3-8b.jsonl` | qwen3:8b on Ollama, 70 runs |
| `compare-report.md` | output of `report.py --compare` on the four files |
| `metrics-report.md` | output of `report.py --metrics` on the four files |

All four batches used `--seed 1000`. Each line is one run: the observations exactly as sent, the expected outcome, the engine's decision with its raw response, and the match flags. No API keys are stored.

`report.py` saves a new timestamped report next to the input files. To keep this folder unchanged, delete those copies afterwards. To rebuild the reports from these files, run:

```
python report.py --compare docs/sample-run-2026-10-05/*.jsonl
python report.py --metrics docs/sample-run-2026-10-05/*.jsonl
```
