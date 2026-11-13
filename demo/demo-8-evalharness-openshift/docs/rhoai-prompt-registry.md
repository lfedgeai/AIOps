# Agent prompts in RHOAI MLflow

Use the MLflow workspace that matches the pipeline namespace. Each run links to
a prompt manifest in its `aiops.prompt_registry_manifest` tag. The manifest
lists the system and user prompt parts in order. Read each named prompt version
and join its parts without a separator to recover the exact text. The manifest
also contains SHA-256 hashes for the parts and the complete prompt.

MLflow limits the length of each prompt. The harness splits long prompts into
parts of at most 6,000 characters. It reuses a part version when the text
matches. A version's `aiops.source_run_id` tag names the first run that created
it. Use each run's manifest to find all runs that use a version.

New runs also store snapshots under `agent_trace/`. Each agent invocation has a
separate directory. Each logging call has a separate snapshot. The root-level
files remain as the latest snapshot for compatibility. The JSON prompt fields
contain the full system and user text. Other conversation and tool fields can
contain shortened text.

The backfill script registers the prompt fields from existing run artifacts.
It cannot recover snapshots that older runs overwrote. Set the MLflow tracking
URI, token, and workspace in the environment before running it:

```bash
python3 scripts/backfill_mlflow_prompts.py RUN_ID [RUN_ID ...]
```

The pipeline service account needs the role in
`manifests/rbac/pipeline-prompt-registry.yaml`. RHOAI uses the same permission
for prompts and registered models. Its `update` permission also permits some
model-version changes, including deletion. Restrict access to the workspace
accordingly. Prompts can contain sensitive context; apply the same care as for
run artifacts.
