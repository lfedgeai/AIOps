#!/usr/bin/env bash
# Harden OpenClaw's seed-openclaw init in the openshell-on-openshift lab deployment
# so cluster power-cycles / node blips do not leave the pod in Init:CrashLoopBackOff.
#
# Guards:
#   1) Wait for ConfigMap mount (both openclaw.json + managed policy)
#   2) Retry OpenShell CLI download (GitHub may be slow right after power-up)
#   3) Reuse openshell binary / plugin from EmptyDir on init restarts
#   4) Idempotent `plugins install --force` with retries
#   5) Control UI brand logo patch (v3): root-absolute /apple-touch-icon.png + SW cache bust
#   (v4) mlflow-openclaw npm plugin is NOT installed — audit uses netobserv-mcp Runs instead.
#   (v6) OpenShell CLI download uses node fetch when curl is absent (2026.6.11 image).
#   (v7) OpenShell tarball extract uses node ustar parser when tar is absent (July+ images).
#
# Logo: OpenClaw 2026.6.11 builds sidebar/login logo URLs with a /chat basePath, so
# apple-touch-icon.png was requested as /chat/apple-touch-icon.png (404). Init copies
# dist/control-ui to /opt/openclaw/control-ui and patches index-*.js so so()/co() return
# Marker: NETOBSERV_SEED_HARDENED_v7 (+ node fetch + node tar.gz extract fallbacks).
#
# Always prefer: oc -n openclaw apply -k <lab>/manifests/openclaw
# Never: oc create configmap openclaw-config --from-file=openclaw.json=... alone
# (that drops openclaw-managed-policy.yaml and CrashLoops init).
set -euo pipefail

LAB_OPENCLAW_DIR="${LAB_OPENCLAW_DIR:-$HOME/labs/openshell-on-openshift-lab/manifests/openclaw}"
DEP="${LAB_OPENCLAW_DEPLOYMENT:-$LAB_OPENCLAW_DIR/deployment.yaml}"
APPLY="${APPLY_LAB_OPENCLAW:-1}"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
KUBECTL="$(command -v oc || command -v kubectl)"
MARKER="NETOBSERV_SEED_HARDENED_v7"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }

[[ -n "$KUBECTL" ]] || { echo "oc/kubectl required" >&2; exit 1; }
[[ -f "$DEP" ]] || { warn "lab deployment not found at $DEP — skip"; exit 0; }

python3 - "$DEP" "$MARKER" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
marker = sys.argv[2]
lines = p.read_text().splitlines(True)

# Find seed-openclaw init script block via a simple line scan (avoid regex backtracking).
in_seed = False
pipe_idx = None
sec_idx = None
for i, line in enumerate(lines):
    if "name: seed-openclaw" in line:
        in_seed = True
        continue
    if not in_seed:
        continue
    if pipe_idx is None and line.lstrip().startswith("- |"):
        # The command list item that starts the script body
        pipe_idx = i
        continue
    if pipe_idx is not None and line.lstrip().startswith("securityContext:"):
        sec_idx = i
        break
    # Left the init container without finding securityContext
    if pipe_idx is None and line.lstrip().startswith("- name:") and "seed-openclaw" not in line:
        break

if pipe_idx is None or sec_idx is None:
    print("needle-missing: could not locate seed-openclaw |- script block", file=sys.stderr)
    raise SystemExit(2)

body = "".join(lines[pipe_idx + 1 : sec_idx])
if marker in body:
    print("already-patched")
    raise SystemExit(0)

# Indent = leading whitespace of the first non-empty body line, else 14 spaces.
ind = "              "
for bl in lines[pipe_idx + 1 : sec_idx]:
    if bl.strip():
        ind = bl[: len(bl) - len(bl.lstrip())]
        break

hardened = [
    "set -euo pipefail",
    f"# {marker} — power-cycle / init-restart safe",
    "mkdir -p /opt/openclaw/config /opt/openclaw/config/policies /opt/openclaw/workspace /openshell-bin",
    "",
    "# ConfigMap mount can lag after node power-up; both keys required.",
    "for i in $(seq 1 60); do",
    "  if [ -f /bootstrap-config/openclaw.json ] && [ -f /bootstrap-config/openclaw-managed-policy.yaml ]; then",
    "    break",
    "  fi",
    '  echo "waiting for bootstrap-config mount ($i/60)..."',
    "  sleep 2",
    "done",
    "if [ ! -f /bootstrap-config/openclaw.json ] || [ ! -f /bootstrap-config/openclaw-managed-policy.yaml ]; then",
    '  echo "FATAL: ConfigMap openclaw-config incomplete under /bootstrap-config" >&2',
    '  echo "Expected openclaw.json + openclaw-managed-policy.yaml" >&2',
    '  echo "Fix: oc -n openclaw apply -k <lab>/manifests/openclaw  (never create CM with only openclaw.json)" >&2',
    "  ls -la /bootstrap-config >&2 || true",
    "  exit 1",
    "fi",
    "cp /bootstrap-config/openclaw.json /opt/openclaw/config/openclaw.json",
    "cp /bootstrap-config/openclaw-managed-policy.yaml /opt/openclaw/config/policies/openclaw-managed-policy.yaml",
    "",
    "# Writable Control UI copy so we can fix brand logo + bust SW cache",
    "rm -rf /opt/openclaw/control-ui",
    "cp -a /app/dist/control-ui /opt/openclaw/control-ui",
    "# Cosmetic only — never fail init. Brand logos must be root-absolute: on /chat the UI",
    "# passes a basePath so Ja('apple-touch-icon.png', base) becomes /chat/apple-touch-icon.png (404).",
    "UI_JS=\"$(ls /opt/openclaw/control-ui/assets/index-*.js 2>/dev/null | head -1)\"",
    "if [ -n \"${UI_JS:-}\" ]; then",
    "  sed -i 's/function so(e){return Ja(`favicon.svg`,e)}/function so(e){return `\\/apple-touch-icon.png`}/' \"$UI_JS\" 2>/dev/null || true",
    "  sed -i 's/function so(e){return Ja(`apple-touch-icon.png`,e)}/function so(e){return `\\/apple-touch-icon.png`}/' \"$UI_JS\" 2>/dev/null || true",
    "  sed -i 's/function co(e){return Ja(`favicon.svg`,e)}/function co(e){return `\\/apple-touch-icon.png`}/' \"$UI_JS\" 2>/dev/null || true",
    "  sed -i 's/function co(e){return Ja(`apple-touch-icon.png`,e)}/function co(e){return `\\/apple-touch-icon.png`}/' \"$UI_JS\" 2>/dev/null || true",
    "fi",
    "sed -i 's/2026.6.11-e085fa1a3ffd-logo2/2026.6.11-e085fa1a3ffd-logo3/' /opt/openclaw/control-ui/sw.js 2>/dev/null || true",
    "sed -i 's/2026.6.11-e085fa1a3ffd/2026.6.11-e085fa1a3ffd-logo3/' /opt/openclaw/control-ui/sw.js 2>/dev/null || true",
    "",
    "# OpenShell CLI: reuse EmptyDir on init restart; retry download after power-up",
    "if [ -x /openshell-bin/openshell ]; then",
    '  echo "openshell binary already present; skipping download"',
    "else",
    "  dl_ok=0",
    "  OPENSHELL_URL='https://github.com/NVIDIA/OpenShell/releases/download/v0.0.80/openshell-x86_64-unknown-linux-musl.tar.gz'",
    "  OPENSHELL_SHA='e06ac01e7527b4aadeed549265850a197f3d7ed9347f8ba476a062f10d274611'",
    "  export OPENSHELL_URL OPENSHELL_SHA",
    "  for i in $(seq 1 12); do",
    "    if command -v curl >/dev/null 2>&1; then",
    "      if curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 20 \\",
    "          \"${OPENSHELL_URL}\" -o /tmp/openshell.tgz \\",
    '        && echo "${OPENSHELL_SHA}  /tmp/openshell.tgz" | sha256sum -c -; then',
    "        dl_ok=1",
    "        break",
    "      fi",
    "    elif OPENSHELL_URL=\"${OPENSHELL_URL}\" OPENSHELL_SHA=\"${OPENSHELL_SHA}\" node -e 'const fs=require(\"fs\");const crypto=require(\"crypto\");const url=process.env.OPENSHELL_URL;const out=\"/tmp/openshell.tgz\";const want=process.env.OPENSHELL_SHA;(async()=>{const r=await fetch(url);if(!r.ok)throw new Error(\"HTTP \"+r.status);const buf=Buffer.from(await r.arrayBuffer());if(crypto.createHash(\"sha256\").update(buf).digest(\"hex\")!==want)throw new Error(\"checksum\");fs.writeFileSync(out,buf);})().catch(e=>{console.error(e);process.exit(1)});'; then",
    "      dl_ok=1",
    "      break",
    "    fi",
    '    echo "openshell download failed (attempt $i/12); retrying..."',
    "    sleep $((i * 5))",
    "  done",
    '  if [ "$dl_ok" != 1 ]; then',
    '    echo "FATAL: could not download OpenShell CLI after retries" >&2',
    "    exit 1",
    "  fi",
    "  if command -v tar >/dev/null 2>&1; then",
    "    tar -xzf /tmp/openshell.tgz -C /openshell-bin",
    "  else",
    "    node -e 'const fs=require(\"fs\"),zlib=require(\"zlib\"),path=require(\"path\");const dst=\"/openshell-bin\";const data=zlib.gunzipSync(fs.readFileSync(\"/tmp/openshell.tgz\"));let off=0;while(off<data.length){const hdr=data.subarray(off,off+512);if(hdr.every(b=>b===0))break;const name=hdr.subarray(0,100).toString(\"utf8\").replace(/\\0.*/,\"\");const size=parseInt(hdr.subarray(124,136).toString(\"utf8\").trim(),8)||0;off+=512;if(name&&size>0){const content=data.subarray(off,off+size);const out=path.join(dst,path.basename(name));fs.mkdirSync(dst,{recursive:true});fs.writeFileSync(out,content);fs.chmodSync(out,0o755);}off+=Math.ceil(size/512)*512;}'",
    "  fi",
    "  chmod 0755 /openshell-bin/openshell",
    "fi",
    "",
    "# Plugin: EmptyDir may already have it after NetworkNotReady init restart",
    "if [ -d /opt/openclaw/config/npm/projects ] && ls /opt/openclaw/config/npm/projects/*/node_modules/@openclaw/openshell-sandbox/dist/index.js >/dev/null 2>&1; then",
    '  echo "openshell-sandbox plugin already installed; skipping"',
    "else",
    "  plug_ok=0",
    "  for i in $(seq 1 8); do",
    "    if OPENCLAW_CONFIG_PATH=/opt/openclaw/config/openclaw.json \\",
    "        node /app/openclaw.mjs plugins install @openclaw/openshell-sandbox@2026.6.11 --force; then",
    "      plug_ok=1",
    "      break",
    "    fi",
    '    echo "plugin install failed (attempt $i/8); retrying..."',
    "    sleep $((i * 5))",
    "  done",
    '  if [ "$plug_ok" != 1 ]; then',
    '    echo "FATAL: openshell-sandbox plugin install failed after retries" >&2',
    "    exit 1",
    "  fi",
    "fi",
    "",
    "# Slack channel plugin (optional — warn only if install fails)",
    "if [ -d /opt/openclaw/config/npm/projects ] && ls /opt/openclaw/config/npm/projects/*/node_modules/@openclaw/slack/dist/index.js >/dev/null 2>&1; then",
    '  echo "slack plugin already installed; skipping"',
    "else",
    "  slack_ok=0",
    "  for i in $(seq 1 8); do",
    "    if OPENCLAW_CONFIG_PATH=/opt/openclaw/config/openclaw.json \\",
    "        node /app/openclaw.mjs plugins install @openclaw/slack@2026.6.11 --force; then",
    "      slack_ok=1",
    "      break",
    "    fi",
    '    echo "slack plugin install failed (attempt $i/8); retrying..."',
    "    sleep $((i * 5))",
    "  done",
    '  if [ "$slack_ok" != 1 ]; then',
    '    echo "WARN: @openclaw/slack plugin install failed — channels.slack stays disabled" >&2',
    "  fi",
    "fi",
]

new_body_lines = [(ind + line + "\n") if line else "\n" for line in hardened]
new_lines = lines[: pipe_idx + 1] + new_body_lines + lines[sec_idx:]
p.write_text("".join(new_lines))
print("patched")
PY

ensure_configmap_both_keys() {
  local json policy
  json="$("$KUBECTL" -n "$OPENCLAW_NS" get cm openclaw-config -o jsonpath='{.data.openclaw\.json}' 2>/dev/null || true)"
  policy="$("$KUBECTL" -n "$OPENCLAW_NS" get cm openclaw-config -o jsonpath='{.data.openclaw-managed-policy\.yaml}' 2>/dev/null || true)"
  if [[ -n "$json" && -n "$policy" ]]; then
    ok "ConfigMap openclaw-config has both bootstrap keys"
    return 0
  fi
  warn "ConfigMap openclaw-config incomplete (json=${#json} policy=${#policy}) — re-applying kustomize"
  if [[ -d "$LAB_OPENCLAW_DIR" ]]; then
    "$KUBECTL" -n "$OPENCLAW_NS" apply -k "$LAB_OPENCLAW_DIR"
  fi
  json="$("$KUBECTL" -n "$OPENCLAW_NS" get cm openclaw-config -o jsonpath='{.data.openclaw\.json}' 2>/dev/null || true)"
  policy="$("$KUBECTL" -n "$OPENCLAW_NS" get cm openclaw-config -o jsonpath='{.data.openclaw-managed-policy\.yaml}' 2>/dev/null || true)"
  if [[ -n "$json" && -n "$policy" ]]; then
    ok "ConfigMap openclaw-config repaired"
    return 0
  fi
  warn "ConfigMap still incomplete after apply"
  return 1
}

if [[ "$APPLY" == "1" ]]; then
  if [[ -d "$LAB_OPENCLAW_DIR" ]]; then
    "$KUBECTL" -n "$OPENCLAW_NS" apply -k "$LAB_OPENCLAW_DIR"
    ok "Applied hardened seed-openclaw via $LAB_OPENCLAW_DIR"
  else
    "$KUBECTL" apply -f "$DEP"
    ok "Applied hardened seed-openclaw via $DEP"
  fi
  ensure_configmap_both_keys || true
else
  ok "Patched $DEP (APPLY_LAB_OPENCLAW=0 — not applied)"
fi
