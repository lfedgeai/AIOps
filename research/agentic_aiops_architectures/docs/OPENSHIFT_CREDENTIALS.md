# OpenShift Access Credentials

To deploy and run the experiment tracking stack on OpenShift, provide the following:

---

## 1. Cluster Access

**Option A — Interactive login (recommended for first run)**

```bash
oc login <cluster-api-url>
```

You'll be prompted for username and password (or token). No config file needed; `oc` stores the session in `~/.kube/config`.

**Option B — Service account token**

If using a service account or CI/CD:

- **Token**: Bearer token with `cluster-admin` or namespace-scoped `edit` role
- **API URL**: `https://api.<cluster-domain>:6443`

**Option C — Kubeconfig file**

- Path to a `kubeconfig` file (e.g. `~/.kube/config` or a downloaded project kubeconfig)
- Ensure the current context targets the correct cluster

---

## 2. What I Need From You

| Item | Purpose | Example |
|------|---------|---------|
| **Cluster API URL** | Deploy manifests, create resources | `https://api.cluster-abc123.example.com:6443` |
| **Login method** | Authenticate | `oc login` command OR path to kubeconfig |
| **Namespace** | Where to deploy | `agentic-aiops` (or your choice) |
| **Permissions** | Create Project, Deployment, Service, Route, PVC | `edit` or `admin` role in namespace |

---

## 3. Verifying Access

After logging in, run:

```bash
# Check you're logged in
oc whoami

# Check current context
oc config current-context

# Create the namespace (or use existing)
oc new-project agentic-aiops

# Verify you can create resources
oc auth can-i create deployment --all-namespaces
oc auth can-i create route --all-namespaces
oc auth can-i create pvc --all-namespaces
```

---

## 4. No Secrets in Repo

**Do not** commit tokens, passwords, or kubeconfig files to the repository. Credentials are used via:

- `oc login` → session in `~/.kube/config`
- Environment variable `KUBECONFIG` pointing to a local file
- CI/CD secrets (e.g. sealed secrets, Vault) for automation

---

## 5. Config File Override

You can override `config/openshift.yaml` values via environment variables or a local override file (excluded from git):

```bash
# Override namespace
export OPENSHIFT_NAMESPACE=my-project
```
