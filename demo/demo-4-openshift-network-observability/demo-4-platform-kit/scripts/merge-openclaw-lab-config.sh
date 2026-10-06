#!/usr/bin/env bash
# merge-openclaw-lab-config.sh — merge kit-owned fields into lab openclaw.json (single source of truth).
#
# Problem this solves: seed/hooks/slack/trustyai each patch openclaw-config differently;
# `kubectl apply -k` on the lab resets fields the last writer did not own (e.g. guard baseUrl).
#
# Usage (internal — prefer reconcile-openclaw-demo-wiring.sh):
#   SLACK_CHANNEL_ID=C0123456789 ./scripts/merge-openclaw-lab-config.sh apply
#   ./scripts/merge-openclaw-lab-config.sh print   # merged JSON to stdout (dry-run)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
GUARDRAILS_NS="${GUARDRAILS_NS:-netobserv-guardrails}"
GATEWAY_PRESET="${GATEWAY_PRESET:-netobserv-sre}"
SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}"
LAB_CFG="${LAB_CFG:-$HOME/labs/openshell-on-openshift-lab/manifests/openclaw/config.yaml}"
KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-apply}"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }

[[ -n "$KUBECTL" ]] || { echo "oc/kubectl required" >&2; exit 1; }
[[ -f "$LAB_CFG" ]] || { warn "Lab config missing: $LAB_CFG"; exit 1; }

"$KUBECTL" apply -k "$ROOT/manifests/openclaw-demo-wiring" >/dev/null 2>&1 || true

merge_python() {
  local mode="${1:-print}"
  LAB_CFG="$LAB_CFG" OPENCLAW_NS="$OPENCLAW_NS" GUARDRAILS_NS="$GUARDRAILS_NS" \
  GATEWAY_PRESET="$GATEWAY_PRESET" SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" \
  KUBECTL="$KUBECTL" MERGE_MODE="$mode" python3 - <<'PY'
import json, os, subprocess, sys
from pathlib import Path

lab = Path(os.environ["LAB_CFG"])
ns = os.environ["OPENCLAW_NS"]
gns = os.environ["GUARDRAILS_NS"]
preset = os.environ["GATEWAY_PRESET"]
slack_id = os.environ.get("SLACK_CHANNEL_ID", "").strip()
kubectl = os.environ["KUBECTL"]
merge_mode = os.environ.get("MERGE_MODE", "print")

def run(*args):
    return subprocess.run([kubectl, *args], capture_output=True, text=True)

def get_cm(key):
    r = run("-n", ns, "get", "cm", "netobserv-demo-desired-state",
            "-o", f"jsonpath={{.data.{key}}}")
    return (r.stdout or "").strip() if r.returncode == 0 else ""

def secret_exists(name):
    return run("-n", ns, "get", "secret", name).returncode == 0

def crd_exists():
    return run("-n", gns, "get", "guardrailsorchestrator", "guardrails-orchestrator").returncode == 0

def guard_proxy_url():
    return f"http://netobserv-llm-guard-proxy.{gns}.svc.cluster.local:8080/{preset}/v1"

features_raw = get_cm("features.yaml") or "guardrails: auto\nevent_aiops: auto\nslack: auto"
features = {}
for line in features_raw.splitlines():
    if ":" in line:
        k, v = line.split(":", 1)
        features[k.strip()] = v.strip()

desired_slack = slack_id or get_cm("slack_channel_id") or ""
direct_llm = get_cm("direct_llm_base_url") or "https://litemaas.example.com/v1"

def feature_on(name):
    mode = features.get(name, "auto")
    if mode == "off":
        return False
    if mode == "on":
        return True
    if name == "guardrails":
        return crd_exists()
    if name == "event_aiops":
        return secret_exists("openclaw-hooks-token") and secret_exists("openclaw-slack-tokens")
    if name == "slack":
        return secret_exists("openclaw-slack-tokens") and bool(desired_slack)
    return False

d = json.loads(lab.read_text())
changed = []

# --- Layer 0: LLM baseUrl (kit-owned when TrustyAI installed) ---
prov = d.setdefault("models", {}).setdefault("providers", {}).setdefault("openai", {})
want_url = guard_proxy_url() if feature_on("guardrails") else direct_llm
if prov.get("baseUrl") != want_url:
    prov["baseUrl"] = want_url
    changed.append(f"baseUrl→{'guard-proxy' if feature_on('guardrails') else 'direct-llm'}")

# Disable thinking on primary model (TrustyAI / proxy safe)
primary = (d.get("agents", {}).get("defaults", {}).get("model") or {}).get("primary") or "openai/Qwen3.6-35B-A3B"
for key in (primary, primary.split("/", 1)[-1] if "/" in primary else primary):
    m = d.setdefault("agents", {}).setdefault("defaults", {}).setdefault("models", {}).setdefault(key, {})
    params = m.setdefault("params", {})
    extra = params.setdefault("extra_body", {})
    kwargs = extra.setdefault("chat_template_kwargs", {})
    if kwargs.get("enable_thinking") is not False:
        kwargs["enable_thinking"] = False
        changed.append("enable_thinking=false")

# --- Hooks (event path) ---
if feature_on("event_aiops"):
    r = run("-n", ns, "get", "secret", "openclaw-hooks-token",
            "-o", "jsonpath={.data.OPENCLAW_HOOKS_TOKEN}")
    token = ""
    if r.returncode == 0 and r.stdout.strip():
        import base64
        token = base64.b64decode(r.stdout.strip()).decode()
    wanted_hooks = {"enabled": True, "path": "/hooks", "token": token}
    if token and d.get("hooks") != wanted_hooks:
        d["hooks"] = wanted_hooks
        changed.append("hooks")

# --- Slack ---
if feature_on("slack") and desired_slack:
    plugins = d.setdefault("plugins", {})
    allow = plugins.setdefault("allow", [])
    if isinstance(allow, list) and "slack" not in allow:
        allow.append("slack")
        changed.append("plugins.allow+slack")
    entries = plugins.setdefault("entries", {})
    if entries.get("slack") != {"enabled": True}:
        entries["slack"] = {"enabled": True}
        changed.append("slack plugin")
    slack_ch = d.setdefault("channels", {}).setdefault("slack", {})
    if slack_ch.get("enabled") is not True:
        slack_ch["enabled"] = True
        changed.append("slack.enabled")
    ch_map = slack_ch.setdefault("channels", {})
    want_ch = {"enabled": True, "requireMention": True}
    cur = ch_map.get(desired_slack)
    if isinstance(cur, dict) and "allow" in cur:
        cur = {k: v for k, v in cur.items() if k != "allow"}
        if cur.get("enabled") is not True:
            cur["enabled"] = True
        ch_map[desired_slack] = cur
        changed.append(f"slack.channel.migrate-allow→enabled={desired_slack}")
        cur = ch_map.get(desired_slack)
    if cur != want_ch:
        ch_map[desired_slack] = want_ch
        changed.append(f"slack.channel={desired_slack}")

# --- Remove deprecated input-guard plugin ---
plugins = d.setdefault("plugins", {})
allow = plugins.get("allow")
if isinstance(allow, list) and "netobserv-input-guard" in allow:
    plugins["allow"] = [x for x in allow if x != "netobserv-input-guard"]
    changed.append("drop input-guard allow")
entries = plugins.get("entries") or {}
if isinstance(entries, dict) and "netobserv-input-guard" in entries:
    entries.pop("netobserv-input-guard", None)
    changed.append("drop input-guard entry")
load = plugins.setdefault("load", {})
paths = load.get("paths")
if isinstance(paths, list):
    new_paths = [p for p in paths if "netobserv-input-guard" not in p]
    if new_paths != paths:
        load["paths"] = new_paths
        changed.append("drop input-guard load.paths")

d.pop("meta", None)  # OpenClaw schema rejects unknown meta keys

out = json.dumps(d, indent=2) + "\n"
if merge_mode == "write":
    if changed:
        lab.write_text(out)
    print("changed:", ",".join(changed) if changed else "none")
    sys.exit(0)

print(out)
if changed:
    print("changed:", ",".join(changed), file=sys.stderr)
PY
}

case "$CMD" in
  print|dry-run)
    merge_python
    ;;
  apply|merge)
    step "Merge kit-owned fields into lab openclaw.json"
    merge_python write
    ok "Lab config merged ($(basename "$LAB_CFG"))"
    ;;
  *)
    echo "usage: $0 [apply|print]" >&2
    exit 1
    ;;
esac
