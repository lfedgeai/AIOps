"""Register the saved prompts of existing MLflow runs without running agents."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlflow.tracking import MlflowClient

from code.agents.mlflow_prompt_registry import register_agent_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    args = parser.parse_args()
    client = MlflowClient()
    for run_id in args.run_ids:
        prompts = json.loads(Path(client.download_artifacts(run_id, "agent_llm_prompts.json")).read_text())
        manifest = register_agent_prompts(client, run_id, prompts)
        for role, info in manifest["roles"].items():
            text = "".join(
                client.get_prompt_version(part["name"], part["version"]).template
                for part in info["parts"]
            )
            if text != prompts[f"{role}_prompt"] or hashlib.sha256(text.encode()).hexdigest() != info["sha256"]:
                raise RuntimeError(f"Prompt reconstruction failed for {run_id} ({role})")
        path = "prompt_registry/backfill.json"
        client.log_dict(run_id, manifest, path)
        client.set_tag(run_id, "aiops.prompt_registry_manifest", path)
        client.set_tag(run_id, "aiops.prompt_registry_status", "ok")
        print(f"Registered and verified {run_id}: " + ", ".join(
            f"{role}={len(info['parts'])} part(s)" for role, info in manifest["roles"].items()
        ))


if __name__ == "__main__":
    main()
