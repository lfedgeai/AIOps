#!/usr/bin/env python3
"""Load, validate, prompt, and apply greenfield site secrets (YAML)."""

from __future__ import annotations

import argparse
import copy
import getpass
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

DEFAULT_REL = "config/site-secrets.local.yaml"
EXAMPLE_REL = "config/site-secrets.example.yaml"
EXAMPLE_JSON = "config/site-secrets.example.json"
DEFAULT_JSON = "config/site-secrets.local.json"


@dataclass(frozen=True)
class PromptField:
    dotpath: str
    label: str
    secret: bool = False
    help: str = ""
    example: str = ""
    optional: bool = False


@dataclass(frozen=True)
class PromptSection:
    key: str
    title: str
    when: str
    blurb: str
    fields: list[PromptField]
    skippable: bool = False


def _oc_jsonpath(expr: str) -> str:
    try:
        proc = subprocess.run(
            ["oc", "get", "infrastructure", "cluster", "-o", f"jsonpath={{{expr}}}"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


def cluster_hints() -> dict[str, str]:
    infra = _oc_jsonpath(".status.infrastructureName")
    region = _oc_jsonpath(".status.platformStatus.aws.region")
    s3 = f"netobserv-loki-{infra}" if infra else ""
    return {"cluster_id": infra, "aws_region": region, "s3_bucket": s3}


PROMPT_SECTIONS: list[PromptSection] = [
    PromptSection(
        key="site",
        title="Site label",
        when="Anytime (cosmetic)",
        blurb="A short name so you can tell this install apart from other clusters.",
        fields=[
            PromptField(
                "site.name",
                "Site name",
                help="Your choice only — stored in site-secrets; install scripts do not use it.",
                example="cluster1, my-demo-site",
            ),
        ],
    ),
    PromptSection(
        key="aws",
        title="AWS (NetObserv / Loki)",
        when="Required for Phase 1 (netobserv)",
        blurb=(
            "Credentials need permission to create and manage one S3 bucket.\n"
            "You provide a bucket NAME here; Phase 1 creates the bucket if it does not exist."
        ),
        fields=[
            PromptField(
                "aws.region",
                "AWS region",
                help="Region for the Loki S3 bucket (often matches the OpenShift cluster region).",
                example="ap-southeast-1",
            ),
            PromptField(
                "aws.s3_bucket",
                "S3 bucket name for Loki",
                help=(
                    "Globally unique across all AWS accounts. Does not need to exist yet — "
                    "install-netobserv-aws.sh creates it during Phase 1."
                ),
                example="netobserv-loki-my-cluster",
            ),
            PromptField(
                "aws.access_key_id",
                "AWS access key ID",
                help="IAM user or role key with s3:CreateBucket and object read/write on that bucket.",
            ),
            PromptField(
                "aws.secret_access_key",
                "AWS secret access key",
                secret=True,
            ),
        ],
    ),
    PromptSection(
        key="llm",
        title="LLM (OpenShell / OpenClaw)",
        when="Required for Phase 2 (openshell)",
        blurb=(
            "OpenAI-compatible API used by the agent. See docs/PHASE-2-LLM-PROVIDERS.md or run:\n"
            "  ./scripts/phase2-openshell.sh providers"
        ),
        fields=[
            PromptField(
                "llm.provider",
                "LLM provider",
                help="Preset name — litemaas is the recommended default.",
                example="litemaas | vllm | maas-16k | openai",
            ),
            PromptField(
                "llm.base_url",
                "LLM base URL",
                help="OpenAI-compatible /v1 endpoint (include https://).",
                example="https://litemaas.example.com/v1",
            ),
            PromptField(
                "llm.model_id",
                "LLM model id",
                example="Qwen3.6-35B-A3B",
            ),
            PromptField(
                "llm.api_key",
                "LLM API key",
                secret=True,
            ),
            PromptField(
                "llm.policy_host",
                "LLM policy host",
                help="Hostname only (no https://) — added to OpenShell network policy.",
                example="litemaas.example.com",
            ),
            PromptField(
                "llm.context_window",
                "LLM context window",
                help="Optional — Enter keeps default from template.",
                example="131072",
                optional=True,
            ),
            PromptField(
                "llm.max_tokens",
                "LLM max tokens",
                help="Optional — Enter keeps default from template.",
                example="12288",
                optional=True,
            ),
        ],
    ),
    PromptSection(
        key="openshift",
        title="OpenShift defaults",
        when="Phase 1+ (usually leave default)",
        blurb="Storage class for Loki PVCs and operator namespaces.",
        fields=[
            PromptField(
                "openshift.storage_class",
                "Storage class for Loki",
                help="On AWS OpenShift labs this is usually gp3-csi.",
                example="gp3-csi",
            ),
        ],
    ),
    PromptSection(
        key="slack",
        title="Slack",
        when="Optional until Phase 8",
        blurb=(
            "Bot + app tokens for Socket Mode. Safe to skip now and run config prompt again before Phase 8."
        ),
        fields=[
            PromptField(
                "site.slack_channel_id",
                "Slack channel ID",
                help="Starts with C — channel where alerts and bot messages go.",
                example="C0123456789",
                optional=True,
            ),
            PromptField(
                "slack.bot_token",
                "Slack bot token",
                secret=True,
                help="xoxb-… from your Slack app.",
                optional=True,
            ),
            PromptField(
                "slack.app_token",
                "Slack app token",
                secret=True,
                help="xapp-… (Socket Mode).",
                optional=True,
            ),
        ],
        skippable=True,
    ),
    PromptSection(
        key="aap",
        title="Ansible Automation Platform",
        when="Optional until Phase 6",
        blurb="Path to an AAP subscription manifest on this bastion, if you have one ready.",
        fields=[
            PromptField(
                "aap.license_file",
                "AAP license file path",
                help="Ignored by greenfield — apply subscription in AAP Gateway UI (Phase 6).",
                example="~/licenses/aap-subscription.json",
                optional=True,
            ),
        ],
        skippable=True,
    ),
]

# Flat list for `show` and legacy references: (dotpath, label, secret)
FIELDS: list[tuple[str, str, bool]] = [
    (f.dotpath, f.label, f.secret) for sec in PROMPT_SECTIONS for f in sec.fields
]

PHASE_REQUIRED: dict[str, list[str]] = {
    "prereq": [],  # slack not required until phase 8
    "netobserv": [
        "aws.region",
        "aws.s3_bucket",
        "aws.access_key_id",
        "aws.secret_access_key",
    ],
    "openshell": [
        "llm.base_url",
        "llm.model_id",
        "llm.api_key",
        "llm.policy_host",
    ],
    "slack": ["slack.bot_token", "slack.app_token", "site.slack_channel_id"],
    "event": ["site.slack_channel_id"],
    "spiffe": ["site.slack_channel_id"],
    "aap": [],
    "all": [
        "site.slack_channel_id",
        "aws.region",
        "aws.s3_bucket",
        "aws.access_key_id",
        "aws.secret_access_key",
        "llm.base_url",
        "llm.model_id",
        "llm.api_key",
        "llm.policy_host",
        "slack.bot_token",
        "slack.app_token",
    ],
}


def gf_root() -> Path:
    return Path(__file__).resolve().parent.parent


def config_path(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("SITE_SECRETS_FILE") or os.environ.get("GREENFIELD_SECRETS_FILE")
    if env:
        return Path(env).expanduser().resolve()
    yaml_path = gf_root() / DEFAULT_REL
    json_path = gf_root() / DEFAULT_JSON
    if yaml_path.is_file():
        return yaml_path
    if json_path.is_file():
        return json_path
    return yaml_path


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    elif yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        sys.stderr.write(
            "error: PyYAML not installed for .yaml config.\n"
            "  Fix: dnf install python3-pyyaml  OR  pip3 install --user pyyaml\n"
            "  Or use JSON: cp config/site-secrets.example.json config/site-secrets.local.json\n"
            "       export SITE_SECRETS_FILE=config/site-secrets.local.json\n"
        )
        sys.exit(1)
    if not isinstance(data, dict):
        sys.stderr.write(f"error: {path} must be a mapping\n")
        sys.exit(1)
    return data


def save_config(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(data, indent=2) + "\n")
    elif yaml is not None:
        path.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False))
    else:
        sys.stderr.write("error: PyYAML required to save .yaml — use .json or install pyyaml\n")
        sys.exit(1)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def load_yaml(path: Path) -> dict[str, Any]:
    return load_config(path)


def get_nested(data: dict[str, Any], dotpath: str) -> Any:
    cur: Any = data
    for part in dotpath.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def set_nested(data: dict[str, Any], dotpath: str, value: Any) -> None:
    parts = dotpath.split(".")
    cur = data
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
        if not isinstance(cur, dict):
            raise ValueError(f"invalid path {dotpath}")
    cur[parts[-1]] = value


def is_empty(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and not val.strip():
        return True
    return False


def mask(val: Any) -> str:
    if val is None or (isinstance(val, str) and not val):
        return "(empty)"
    s = str(val)
    if len(s) <= 8:
        return "***"
    return s[:4] + "…" + s[-4:]


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def default_data() -> dict[str, Any]:
    example_json = gf_root() / EXAMPLE_JSON
    example = gf_root() / EXAMPLE_REL
    if example_json.is_file():
        return load_config(example_json)
    if example.is_file():
        return load_config(example)
    return {
        "site": {"name": "", "slack_channel_id": ""},
        "aws": {"region": "", "s3_bucket": "", "access_key_id": "", "secret_access_key": ""},
        "llm": {
            "provider": "litemaas",
            "base_url": "",
            "model_id": "",
            "api_key": "",
            "policy_host": "",
            "context_window": 131072,
            "max_tokens": 12288,
            "enable_thinking": True,
        },
        "slack": {"bot_token": "", "app_token": ""},
        "aap": {"license_file": ""},
        "openshift": {
            "storage_class": "gp3-csi",
            "openclaw_namespace": "openclaw",
            "openshell_namespace": "openshell",
        },
    }


def cmd_init(path: Path) -> int:
    if path.is_file():
        print(f"[ok] already exists: {path}")
        return 0
    data = default_data()
    if path.suffix.lower() != ".json" and yaml is None:
        path = path.with_suffix(".json")
        print(f"[warn] PyYAML not installed — using JSON: {path}")
    save_config(path, data)
    print(f"[ok] created {path}")
    print("     Next: ./scripts/greenfield-install.sh config prompt")
    return 0


def _print_rule(char: str = "─", width: int = 62) -> None:
    print(char * width)


def _print_banner(path: Path, mode: str) -> None:
    _print_rule("═")
    print("  Greenfield site secrets wizard")
    _print_rule("═")
    print()
    print("  Saves to:", path)
    print()
    print("  This wizard collects AWS, LLM, and (optionally) Slack settings.")
    print("  OpenShift cluster login is separate — NOT collected here:")
    print("    oc login …  &&  ./scripts/cluster-login.sh check")
    print("    See docs/CLUSTER-LOGIN.md")
    print()
    if mode == "minimal":
        print("  Mode: minimal (Phases 1–2 essentials — skips Slack and AAP sections)")
    else:
        print("  Mode: full (includes optional Slack and AAP — you can skip those sections)")
    print()
    print("  Tips:")
    print("    • Press Enter to keep an existing or default value")
    print("    • Set secrets show [**** set — Enter to keep]")
    print()


def _print_section_header(index: int, total: int, section: PromptSection) -> None:
    _print_rule()
    print(f"  {index}/{total}  {section.title}")
    print(f"  When needed: {section.when}")
    _print_rule()
    for line in textwrap.dedent(section.blurb).strip().splitlines():
        print(f"  {line}")
    print()


def _field_help(field: PromptField) -> None:
    if field.help:
        for line in textwrap.dedent(field.help).strip().splitlines():
            print(f"    → {line}")
    if field.example:
        print(f"    Example: {field.example}")
    if field.optional:
        print("    (optional — Enter to skip)")


def _suggest_default(dotpath: str, data: dict[str, Any], hints: dict[str, str]) -> str:
    if dotpath == "aws.region" and hints.get("aws_region"):
        return hints["aws_region"]
    if dotpath == "aws.s3_bucket" and hints.get("s3_bucket"):
        return hints["s3_bucket"]
    cur = get_nested(data, dotpath)
    if not is_empty(cur):
        return str(cur)
    if dotpath == "llm.provider":
        return "litemaas"
    if dotpath == "openshift.storage_class":
        return "gp3-csi"
    if dotpath == "llm.context_window":
        return str(get_nested(data, dotpath) or 131072)
    if dotpath == "llm.max_tokens":
        return str(get_nested(data, dotpath) or 12288)
    return ""


def _prompt_value(
    field: PromptField,
    data: dict[str, Any],
    hints: dict[str, str],
) -> str | None:
    """Return new string value, or None to leave unchanged."""
    cur = get_nested(data, dotpath := field.dotpath)
    suggested = _suggest_default(dotpath, data, hints)

    print()
    opt = " (optional)" if field.optional else ""
    print(f"  {field.label}{opt}")
    _field_help(field)

    if field.secret and not is_empty(cur):
        hint = " [**** set — Enter to keep]"
    elif suggested and (is_empty(cur) or str(cur) == suggested):
        hint = f" [{suggested}]" if not field.secret else ""
    elif not field.secret and not is_empty(cur):
        hint = f" [{cur}]"
    else:
        hint = ""

    try:
        if field.secret:
            raw = getpass.getpass(f"  Value{hint}: ")
        else:
            raw = input(f"  Value{hint}: ")
    except (EOFError, KeyboardInterrupt):
        raise

    raw = raw.strip()
    if not raw:
        if field.secret and not is_empty(cur):
            return None
        if suggested and (is_empty(cur) or (not field.secret and field.optional)):
            return suggested if is_empty(cur) or field.optional else None
        return None
    return raw


def _apply_field_value(data: dict[str, Any], dotpath: str, raw: str) -> None:
    if dotpath in ("llm.context_window", "llm.max_tokens"):
        set_nested(data, dotpath, int(raw))
    elif dotpath == "llm.enable_thinking":
        set_nested(data, dotpath, raw.lower() in ("1", "true", "yes", "y"))
    else:
        set_nested(data, dotpath, raw)


def _ask_skip_section(section: PromptSection) -> bool:
    print()
    try:
        ans = input(f"  Skip '{section.title}' for now? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        raise
    return ans in ("", "y", "yes")


def _sections_for_mode(mode: str) -> list[PromptSection]:
    if mode == "minimal":
        return [s for s in PROMPT_SECTIONS if s.key in ("site", "aws", "llm", "openshift")]
    return list(PROMPT_SECTIONS)


def _print_post_summary(path: Path) -> None:
    print()
    _print_rule("═")
    print("  Summary")
    _print_rule("═")
    for phase in ("netobserv", "openshell"):
        if cmd_validate(path, phase, quiet_ok=True) == 0:
            print(f"  [ok] Ready for Phase: {phase}")
        else:
            print(
                f"  [--] Not ready for Phase: {phase} — re-run config prompt or edit {path.name}"
            )
    print()
    print("  Next on bastion:")
    print("    ./scripts/greenfield-install.sh preflight")
    print("    ./scripts/greenfield-install.sh netobserv")
    print()


def cmd_prompt(path: Path, force: bool, mode: str) -> int:
    data = default_data()
    if path.is_file() and not force:
        data = deep_merge(data, load_yaml(path))

    hints = cluster_hints()
    if hints.get("cluster_id"):
        print(f"[hint] Detected cluster: {hints['cluster_id']}")
    elif shutil.which("oc"):
        print("[hint] oc found but cluster not reachable — defaults may be generic")

    _print_banner(path, mode)
    sections = _sections_for_mode(mode)
    total = len(sections)

    try:
        for idx, section in enumerate(sections, start=1):
            _print_section_header(idx, total, section)
            if section.skippable and _ask_skip_section(section):
                print(f"  Skipped {section.title}.")
                continue
            for field in section.fields:
                raw = _prompt_value(field, data, hints)
                if raw is not None:
                    _apply_field_value(data, field.dotpath, raw)
    except (EOFError, KeyboardInterrupt):
        print("\naborted", file=sys.stderr)
        return 1

    save_config(path, data)
    print(f"\n[ok] saved {path} (mode 600)")
    _print_post_summary(path)
    return 0


def cmd_validate(path: Path, phase: str, *, quiet_ok: bool = False) -> int:
    data = load_yaml(path)
    required = PHASE_REQUIRED.get(phase, PHASE_REQUIRED["all"])
    missing = [k for k in required if is_empty(get_nested(data, k))]
    if missing:
        print(f"[fail] {path}: missing required for phase '{phase}':", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print(
            "\n  Fix: edit the YAML or run ./scripts/greenfield-install.sh config prompt",
            file=sys.stderr,
        )
        return 1
    if not quiet_ok:
        print(f"[ok] config valid for phase '{phase}'")
    return 0


def cmd_show(path: Path) -> int:
    data = load_yaml(path)
    if not data:
        print(f"(no file: {path})")
        return 1
    secret_paths = {f[0] for f in FIELDS if f[2]}
    for dotpath, label, _ in FIELDS:
        val = get_nested(data, dotpath)
        disp = mask(val) if dotpath in secret_paths else (val if not is_empty(val) else "(empty)")
        print(f"  {dotpath:28} {disp}")
    return 0


def cmd_export_shell(path: Path) -> int:
    data = load_yaml(path)
    if not data:
        return 0
    llm = data.get("llm") or {}
    aws = data.get("aws") or {}
    site = data.get("site") or {}
    slack = data.get("slack") or {}
    ocp = data.get("openshift") or {}

    def esc(s: str) -> str:
        return s.replace("'", "'\"'\"'")

    exports = {
        "SLACK_CHANNEL_ID": site.get("slack_channel_id", ""),
        "AWS_REGION": aws.get("region", ""),
        "AWS_DEFAULT_REGION": aws.get("region", ""),
        "S3_BUCKET": aws.get("s3_bucket", ""),
        "AWS_ACCESS_KEY_ID": aws.get("access_key_id", ""),
        "AWS_SECRET_ACCESS_KEY": aws.get("secret_access_key", ""),
        "LLM_PROVIDER": llm.get("provider", ""),
        "LLM_BASE_URL": llm.get("base_url", ""),
        "LLM_MODEL": llm.get("model_id", ""),
        "LLM_API_KEY": llm.get("api_key", ""),
        "LLM_POLICY_HOST": llm.get("policy_host", ""),
        "LLM_CONTEXT_WINDOW": str(llm.get("context_window", "")),
        "LLM_MAX_TOKENS": str(llm.get("max_tokens", "")),
        "LLM_ENABLE_THINKING": "1" if llm.get("enable_thinking") else "0",
        "SLACK_BOT_TOKEN": slack.get("bot_token", ""),
        "SLACK_APP_TOKEN": slack.get("app_token", ""),
        "AAP_LICENSE_FILE": (data.get("aap") or {}).get("license_file", ""),
        "STORAGE_CLASS": ocp.get("storage_class", "gp3-csi"),
        "OPENCLAW_NS": ocp.get("openclaw_namespace", "openclaw"),
        "OPENSHELL_NS": ocp.get("openshell_namespace", "openshell"),
        "OPENCLAW_NAMESPACE": ocp.get("openclaw_namespace", "openclaw"),
        "OPENSHELL_NAMESPACE": ocp.get("openshell_namespace", "openshell"),
        "SITE_SECRETS_FILE": str(path),
    }
    for key, val in exports.items():
        if val is None or val == "":
            continue
        print(f"export {key}='{esc(str(val))}'")
    return 0


def kubectl() -> list[str]:
    for bin_name in ("oc", "kubectl"):
        if subprocess.run(["which", bin_name], capture_output=True).returncode == 0:
            return [bin_name]
    sys.stderr.write("error: oc/kubectl required for apply\n")
    sys.exit(1)


def run_kubectl(args: list[str], stdin: str | None = None) -> None:
    cmd = kubectl() + args
    subprocess.run(cmd, input=stdin, text=True, check=True)


def apply_llm_secret(ns: str, api_key: str) -> None:
    proc = subprocess.run(
        kubectl()
        + [
            "create", "secret", "generic", "my-llm-key",
            f"--namespace={ns}",
            "--from-file=api-key=/dev/stdin",
            "--dry-run=client", "-o", "yaml",
        ],
        input=api_key,
        text=True,
        capture_output=True,
        check=True,
    )
    subprocess.run(kubectl() + ["apply", "-f", "-"], input=proc.stdout, text=True, check=True)
    print(f"[ok] secret/my-llm-key in {ns}")


def patch_openclaw_config(cfg_path: Path, llm: dict[str, Any]) -> None:
    if not cfg_path.is_file():
        sys.stderr.write(f"warn: config not found: {cfg_path}\n")
        return
    d = json.loads(cfg_path.read_text())
    base_url = llm.get("base_url", "")
    model_id = llm.get("model_id", "")
    primary = f"openai/{model_id}"
    ctx = int(llm.get("context_window") or 131072)
    max_tok = int(llm.get("max_tokens") or 12288)
    thinking = bool(llm.get("enable_thinking"))

    agents = d.setdefault("agents", {}).setdefault("defaults", {})
    agents.setdefault("model", {})["primary"] = primary
    models_map = agents.setdefault("models", {})
    entry = models_map.setdefault(primary, {"alias": model_id})
    if thinking:
        entry.setdefault("params", {})["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": True}
        }
    elif "params" in entry and "extra_body" in entry.get("params", {}):
        entry["params"]["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    prov = d.setdefault("models", {}).setdefault("providers", {}).setdefault("openai", {})
    prov["baseUrl"] = base_url
    prov.setdefault("apiKey", "env:OPENAI_API_KEY")
    prov.setdefault("api", "openai-completions")
    prov["models"] = [
        {
            "id": model_id,
            "name": model_id,
            "contextWindow": ctx,
            "maxTokens": max_tok,
        }
    ]
    cfg_path.write_text(json.dumps(d, indent=2) + "\n")
    print(f"[ok] patched {cfg_path}")


def patch_managed_policy(policy_path: Path, host: str) -> None:
    if not policy_path.is_file() or not host:
        return
    text = policy_path.read_text()
    if host in text:
        print(f"[ok] policy already allows {host}")
        return
    block = f"""      - host: {host}
        port: 443
        protocol: rest
        access: full
        enforcement: enforce
"""
    marker = "network_policies:"
    if marker not in text:
        sys.stderr.write(f"warn: could not patch {policy_path}\n")
        return
    # Insert after openai_api endpoints header if present
    if "openai_api:" in text and "endpoints:" in text:
        text = re.sub(
            r"(openai_api:\s*\n\s*endpoints:\s*\n)",
            r"\1" + block,
            text,
            count=1,
        )
    else:
        text = text.replace(marker, marker + "\n  openai_api:\n    endpoints:\n" + block)
    policy_path.write_text(text)
    print(f"[ok] patched policy host {host} in {policy_path}")


def apply_slack(ns: str, bot: str, app: str) -> None:
    proc = subprocess.run(
        kubectl()
        + [
            "create", "secret", "generic", "openclaw-slack-tokens",
            f"--namespace={ns}",
            f"--from-literal=SLACK_BOT_TOKEN={bot}",
            f"--from-literal=SLACK_APP_TOKEN={app}",
            "--dry-run=client", "-o", "yaml",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(kubectl() + ["apply", "-f", "-"], input=proc.stdout, text=True, check=True)
    subprocess.run(
        kubectl()
        + [
            "set", "env", f"deployment/openclaw", "-n", ns,
            "--from=secret/openclaw-slack-tokens",
        ],
        check=True,
    )
    print(f"[ok] secret/openclaw-slack-tokens + deployment env in {ns}")


def cmd_apply(path: Path, phase: str, lab_dir: Path) -> int:
    data = load_yaml(path)
    ns = (data.get("openshift") or {}).get("openclaw_namespace", "openclaw")
    llm = data.get("llm") or {}
    slack = data.get("slack") or {}
    aap = data.get("aap") or {}

    if phase in ("llm", "openshell", "all"):
        if is_empty(llm.get("api_key")):
            sys.stderr.write("error: llm.api_key required for apply\n")
            return 1
        ns_proc = subprocess.run(
            kubectl() + ["create", "namespace", ns, "--dry-run=client", "-o", "yaml"],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            kubectl() + ["apply", "-f", "-"],
            input=ns_proc.stdout,
            text=True,
            check=True,
        )
        apply_llm_secret(ns, llm["api_key"])
        lab = lab_dir / "manifests" / "openclaw"
        patch_openclaw_config(lab / "config.yaml", llm)
        patch_managed_policy(lab / "policies" / "managed-policy.yaml", llm.get("policy_host", ""))

    if phase in ("slack", "all"):
        if is_empty(slack.get("bot_token")) or is_empty(slack.get("app_token")):
            sys.stderr.write("error: slack tokens required for apply\n")
            return 1
        apply_slack(ns, slack["bot_token"], slack["app_token"])

    if phase in ("aap", "all"):
        print("[info] AAP license: apply manually in Gateway UI (greenfield does not automate license upload)")

    print(f"[ok] apply complete (phase={phase})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Greenfield site secrets")
    parser.add_argument("--file", "-f", help="path to site-secrets.local.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create site-secrets.local.yaml from example")
    p_prompt = sub.add_parser("prompt", help="interactive wizard")
    p_prompt.add_argument("--force", action="store_true", help="re-prompt all fields")
    mode_grp = p_prompt.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--minimal",
        action="store_const",
        const="minimal",
        dest="mode",
        help="Phases 1–2 only (site, AWS, LLM, storage class)",
    )
    mode_grp.add_argument(
        "--full",
        action="store_const",
        const="full",
        dest="mode",
        help="include Slack and AAP sections (default; optional sections can be skipped)",
    )
    p_prompt.set_defaults(mode="full")
    p_val = sub.add_parser("validate", help="validate required fields")
    p_val.add_argument("--phase", default="all")
    sub.add_parser("show", help="masked summary")
    sub.add_parser("export-shell", help="print export statements for bash")
    p_apply = sub.add_parser("apply", help="apply secrets to cluster / lab files")
    p_apply.add_argument("--phase", default="all", choices=["all", "llm", "openshell", "slack", "aap"])
    p_apply.add_argument(
        "--lab-dir",
        default=os.environ.get("OPENCLAW_LAB_DIR", str(Path.home() / "labs/openshell-on-openshift-lab")),
    )

    args = parser.parse_args()
    path = config_path(args.file)

    if args.cmd == "init":
        return cmd_init(path)
    if args.cmd == "prompt":
        return cmd_prompt(path, args.force, args.mode)
    if args.cmd == "validate":
        if not path.is_file():
            sys.stderr.write(f"error: missing {path} — run: config init\n")
            return 1
        return cmd_validate(path, args.phase)
    if args.cmd == "show":
        return cmd_show(path)
    if args.cmd == "export-shell":
        return cmd_export_shell(path)
    if args.cmd == "apply":
        if not path.is_file():
            sys.stderr.write(f"error: missing {path}\n")
            return 1
        return cmd_apply(path, args.phase, Path(args.lab_dir).expanduser())
    return 0


if __name__ == "__main__":
    sys.exit(main())
