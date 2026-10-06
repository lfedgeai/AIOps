# Cluster login (bastion)

**Not in site-secrets.** `config prompt` collects AWS, LLM, and Slack — **not** the OpenShift API URL or login credentials. You authenticate with **`oc login` on the bastion** before preflight and every install phase.

---

## When to log in

| When | Action |
|------|--------|
| First visit to a new cluster | `oc login` (once per bastion session) |
| Before `./scripts/greenfield-install.sh preflight` | Must show `system:admin` or another cluster-admin user |
| Before Phase 1 `netobserv` | Same — `install-netobserv-aws.sh` reuses the session or prompts again |
| After cluster reboot / session timeout | Re-run `oc login` |
| After `oc logout` | Re-run `oc login` |

---

## Step 1 — SSH to bastion

```bash
ssh lab-user@bastion.<base-domain>
export PATH="$HOME/.local/bin:$PATH"
cd ~/AIOps/demo/demo-4-greenfield-install
export DEMO_KIT_ROOT=~/AIOps/demo/demo-4-platform-kit
```

Set `<base-domain>` to your environment (e.g. `cluster-name.example.com` on AWS IPI, or your lab provider’s bastion hostname).

---

## Step 2 — Find the API server URL

From the **OpenShift console** → copy cluster API URL, or on a host that already has kubeconfig:

```bash
oc whoami --show-server
```

Common OpenShift API URL shapes:

```text
https://api.<cluster-name>.<base-domain>:6443
https://api.cluster-<name>.<base-domain>:6443
```

Copy the exact URL from the OpenShift console (**Copy login command** / cluster details).

---

## Step 3 — Log in

### Option A — kubeadmin (typical sandbox / IPI lab)

Use the **kubeadmin password** from your cluster install output or lab credentials page.

```bash
export OCP_API="https://api.<cluster-name>.<base-domain>:6443"

oc login "$OCP_API" -u kubeadmin -p '<kubeadmin-password>'
```

### Option B — Bearer token

```bash
oc login "$OCP_API" --token='<token>'
```

### Option C — Username + password (interactive password prompt)

```bash
oc login "$OCP_API" -u <username>
# oc prompts for password (not echoed)
```

`install-netobserv-aws.sh` supports the same three methods if you are not logged in when Phase 1 starts.

---

## Step 4 — Verify (required)

```bash
./scripts/cluster-login.sh check
```

Or manually:

```bash
oc whoami
oc whoami --show-server
oc auth can-i '*' '*' --all-namespaces   # must print: yes
```

Expect **cluster-admin** (often `system:admin` after kubeadmin login on lab clusters).

---

## Step 5 — Site secrets (separate from login)

```bash
./scripts/greenfield-install.sh config prompt   # AWS + LLM — not cluster login
./scripts/greenfield-install.sh preflight
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `error: You must be logged in` | Run `oc login` again |
| `oc auth can-i` → `no` | Use kubeadmin or a user with cluster-admin |
| Wrong cluster | `oc whoami --show-server` — re-login to correct API |
| TLS / certificate errors | Lab only: `oc login --insecure-skip-tls-verify=true …` (avoid in production) |
| Login works but preflight fails CNI | Cluster must use **OVN-Kubernetes** |

---

## Quick reference

```bash
./scripts/cluster-login.sh help
./scripts/cluster-login.sh check
```
