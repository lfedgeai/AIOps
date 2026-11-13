import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from code.agents import mlflow_agent_logging as logging
from code.agents import mlflow_prompt_registry as registry


class FakeClient:
    tracking_uri = "https://example.test"

    def __init__(self):
        self.versions = {}
        self.aliases = {}
        self.artifacts = {}
        self.tags = {}

    def get_prompt_version_by_alias(self, name, alias):
        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

        version = self.aliases.get((name, alias))
        if version is None:
            raise MlflowException("Prompt alias not found", error_code=RESOURCE_DOES_NOT_EXIST)
        return self.get_prompt_version(name, version)

    def set_prompt_alias(self, name, alias, version):
        self.aliases[(name, alias)] = str(version)

    def register_prompt(self, name, template, commit_message, tags):
        versions = self.versions.setdefault(name, [])
        version = SimpleNamespace(version=str(len(versions) + 1), template=template, tags=dict(tags))
        versions.append(version)
        return version

    def get_prompt_version(self, name, version):
        return self.versions[name][int(version) - 1]

    def log_dict(self, run_id, data, path):
        self.artifacts[path] = data

    def log_artifact(self, run_id, local_path, artifact_path=None):
        from pathlib import Path

        path = Path(local_path)
        self.artifacts[f"{artifact_path}/{path.name}" if artifact_path else path.name] = path.read_text()

    def set_tag(self, run_id, key, value):
        self.tags[key] = value


class PromptRegistryTests(unittest.TestCase):
    def setUp(self):
        registry._cache.clear()
        self.client = FakeClient()

    def test_reconstruct_and_reuse_versions_across_processes(self):
        prompts = {"openai_model": "test/model", "system_prompt": "あ" * 13000, "user_prompt": "hello"}
        first = registry.register_agent_prompts(self.client, "first", prompts)
        registry._cache.clear()
        second = registry.register_agent_prompts(self.client, "second", prompts)
        self.assertEqual(first, second)
        self.assertEqual(sum(len(v) for v in self.client.versions.values()), 4)
        for role, info in first["roles"].items():
            text = "".join(self.client.get_prompt_version(p["name"], p["version"]).template for p in info["parts"])
            self.assertEqual(text, prompts[f"{role}_prompt"])
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), info["sha256"])

    def test_distinct_prompt_is_a_new_version(self):
        registry.register_agent_prompts(self.client, "one", {"openai_model": "m", "user_prompt": "first"})
        registry.register_agent_prompts(self.client, "two", {"openai_model": "m", "user_prompt": "second"})
        self.assertEqual(len(next(iter(self.client.versions.values()))), 2)

    def test_alias_mismatch_is_not_reused(self):
        prompts = {"openai_model": "m", "user_prompt": "first"}
        manifest = registry.register_agent_prompts(self.client, "one", prompts)
        part = manifest["roles"]["user"]["parts"][0]
        self.client.versions[part["name"]][0].template = "different"
        registry._cache.clear()
        with self.assertRaisesRegex(ValueError, "alias collision"):
            registry.register_agent_prompts(self.client, "two", prompts)

    def test_trace_snapshots_are_unique_and_link_registry(self):
        prompts = {"openai_model": "m", "system_prompt": "system", "user_prompt": "user"}
        with patch("mlflow.tracking.MlflowClient", return_value=self.client), \
             patch.dict("os.environ", {"MLFLOW_WORKSPACE": "workspace"}), \
             patch.object(logging, "log_mlflow_dataset_inputs_for_agent_trace"):
            for _ in range(2):
                logging.log_agent_trace_to_mlflow(
                    run_id="run", tracking_uri="https://example.test", prompts=prompts,
                    tool_calls=[], thinking=None, llm_rounds=[],
                )
        snapshots = [k for k in self.client.artifacts if k.startswith("agent_trace/") and k.endswith("/prompts.json")]
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(self.client.tags["aiops.prompt_registry_status"], "ok")
        self.assertIn(self.client.tags["aiops.prompt_registry_manifest"], self.client.artifacts)
        self.assertEqual(sum(len(v) for v in self.client.versions.values()), 2)

    def test_registry_failure_preserves_prompt_snapshot(self):
        prompts = {"openai_model": "m", "system_prompt": "system", "user_prompt": "user"}
        with patch("mlflow.tracking.MlflowClient", return_value=self.client), \
             patch.dict("os.environ", {"MLFLOW_WORKSPACE": "workspace"}), \
             patch.object(logging, "log_mlflow_dataset_inputs_for_agent_trace"), \
             patch.object(registry, "register_agent_prompts", side_effect=RuntimeError("denied")):
            logging.log_agent_trace_to_mlflow(
                run_id="run", tracking_uri="https://example.test", prompts=prompts,
                tool_calls=[], thinking=None, llm_rounds=[],
            )
        self.assertEqual(self.client.tags["aiops.prompt_registry_status"], "error")
        self.assertTrue(any(k.startswith("agent_trace/") and k.endswith("/prompts.json") for k in self.client.artifacts))
        self.assertTrue(any(k.endswith("/prompt_registry_error.json") for k in self.client.artifacts))


if __name__ == "__main__":
    unittest.main()
