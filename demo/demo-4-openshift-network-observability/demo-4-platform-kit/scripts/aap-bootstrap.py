#!/usr/bin/env python3
"""
Bootstrap AAP Controller objects for NetObserv heal demo.
Requires admin credentials (first-time setup after operator install).

Env:
  AAP_CONTROLLER_URL   https://<gateway-host>  (no path suffix required)
  AAP_ADMIN_USER       admin (default)
  AAP_ADMIN_PASSWORD   from install-aap.sh status secret
  AAP_ORG              netobserv-demo (default)
  OPENCLAW_NS          openclaw
"""

from __future__ import annotations

import base64
import json
import os
import ssl
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLAYBOOKS = ROOT / "ansible" / "playbooks"
ORG = os.environ.get("AAP_ORG", "netobserv-demo")
OPENCLAW_NS = os.environ.get("OPENCLAW_NS", "openclaw")
SA_NAME = os.environ.get("AAP_HEAL_SA", "aap-netobserv-heal")
GITEA_GIT_URL = os.environ.get(
    "GITEA_REPO_URL",
    "http://gitea.gitea.svc.cluster.local:3000/netobserv/netobserv-heal.git",
)
GITEA_GIT_USER = os.environ.get("GITEA_GIT_USER", "netobserv")
GITEA_GIT_PASSWORD = os.environ.get("GITEA_GIT_PASSWORD", "")
GITEA_GIT_BRANCH = os.environ.get("GITEA_GIT_BRANCH", "main")


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if os.environ.get("AAP_TLS_SKIP_VERIFY", "").lower() in ("1", "true", "yes"):
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _gateway_root() -> str:
    raw = os.environ.get("AAP_CONTROLLER_URL", "").rstrip("/")
    if not raw:
        die("AAP_CONTROLLER_URL required")
    for suffix in ("/api/controller/v2", "/api/gateway/v1", "/api/controller"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
    return raw.rstrip("/")


def _controller_base() -> str:
    return f"{_gateway_root()}/api/controller/v2"


def _gateway_base() -> str:
    return f"{_gateway_root()}/api/gateway/v1"


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _basic_auth_header() -> str:
    user = os.environ.get("AAP_ADMIN_USER", "admin")
    password = os.environ.get("AAP_ADMIN_PASSWORD", "")
    if not password:
        die("AAP_ADMIN_PASSWORD required for bootstrap")
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {token}"


def _request(
    base: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    auth: str | None = None,
) -> Any:
    url = f"{base}{path}"
    headers = {"Accept": "application/json"}
    if auth:
        headers["Authorization"] = auth
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120, context=_ssl_ctx()) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        die(f"HTTP {e.code} {path}: {detail[:800]}")


def controller_api(
    method: str, path: str, body: dict[str, Any] | None = None, *, auth: str
) -> Any:
    return _request(_controller_base(), method, path, body, auth=auth)


def gateway_api(
    method: str, path: str, body: dict[str, Any] | None = None, *, auth: str
) -> Any:
    return _request(_gateway_base(), method, path, body, auth=auth)


def get_or_create_token() -> str:
    existing = os.environ.get("AAP_TOKEN", "").strip()
    if existing:
        return existing
    auth = _basic_auth_header()
    data = gateway_api(
        "POST",
        "/tokens/",
        {"description": "netobserv-bootstrap"},
        auth=auth,
    )
    tok = data.get("token") or data.get("access_token")
    if not tok:
        die(f"token create failed: {data}")
    return str(tok)


def get_or_create_org(token: str) -> int:
    auth = f"Bearer {token}"
    data = controller_api(
        "GET", f"/organizations/?name={urllib.parse.quote(ORG)}", auth=auth
    )
    results = data.get("results") or []
    if results:
        return int(results[0]["id"])
    created = controller_api("POST", "/organizations/", {"name": ORG}, auth=auth)
    return int(created["id"])


def get_or_create_project(token: str, org_id: int, scm_cred_id: int | None) -> int:
    auth = f"Bearer {token}"
    name = "netobserv-heal"
    data = controller_api(
        "GET", f"/projects/?name={name}&organization={org_id}", auth=auth
    )
    results = data.get("results") or []
    body: dict[str, Any] = {
        "name": name,
        "organization": org_id,
        "scm_type": "git",
        "scm_url": GITEA_GIT_URL,
        "scm_branch": GITEA_GIT_BRANCH,
        "scm_clean": True,
        "scm_delete_on_update": False,
        "scm_update_on_launch": True,
    }
    if scm_cred_id:
        body["credential"] = scm_cred_id
    if results:
        project_id = int(results[0]["id"])
        controller_api("PATCH", f"/projects/{project_id}/", body, auth=auth)
    else:
        created = controller_api("POST", "/projects/", body, auth=auth)
        project_id = int(created["id"])
    sync_project(token, project_id)
    return project_id


def sync_project(token: str, project_id: int) -> None:
    auth = f"Bearer {token}"
    url = f"{_controller_base()}/projects/{project_id}/update/"
    req = urllib.request.Request(
        url,
        data=b"{}",
        headers={
            "Accept": "application/json",
            "Authorization": auth,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120, context=_ssl_ctx()) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError:
        return
    job_id = data.get("id") or (data.get("project_update") or {}).get("id")
    if job_id:
        wait_project_update(token, int(job_id))


def wait_project_update(token: str, job_id: int, timeout: int = 180) -> None:
    import time

    auth = f"Bearer {token}"
    end = time.time() + timeout
    while time.time() < end:
        job = controller_api("GET", f"/project_updates/{job_id}/", auth=auth)
        status = job.get("status")
        if status in ("successful", "failed", "error", "canceled"):
            if status != "successful":
                die(f"project sync {job_id} ended with status={status}")
            return
        time.sleep(3)
    die(f"project sync {job_id} timed out after {timeout}s")


def sa_token() -> str:
    kubectl = os.environ.get("KUBECTL", "oc")
    out = subprocess.check_output(
        [kubectl, "create", "token", SA_NAME, "-n", OPENCLAW_NS, "--duration=8760h"],
        text=True,
    )
    return out.strip()


def get_or_create_inventory(token: str, org_id: int) -> int:
    auth = f"Bearer {token}"
    name = "netobserv-localhost"
    data = controller_api(
        "GET", f"/inventories/?name={urllib.parse.quote(name)}&organization={org_id}", auth=auth
    )
    results = data.get("results") or []
    if results:
        return int(results[0]["id"])
    created = controller_api(
        "POST",
        "/inventories/",
        {"name": name, "organization": org_id, "description": "Localhost for in-cluster K8s playbooks"},
        auth=auth,
    )
    inv_id = int(created["id"])
    # Add localhost host for ansible playbooks targeting localhost
    hosts = controller_api("GET", f"/hosts/?inventory={inv_id}&name=localhost", auth=auth)
    if not (hosts.get("results") or []):
        controller_api(
            "POST",
            "/hosts/",
            {"name": "localhost", "inventory": inv_id, "variables": "ansible_connection: local\n"},
            auth=auth,
        )
    return inv_id


def get_or_create_gitea_credential(token: str, org_id: int) -> int | None:
    if not GITEA_GIT_PASSWORD:
        return None
    auth = f"Bearer {token}"
    name = "netobserv-gitea"
    data = controller_api(
        "GET", f"/credentials/?name={name}&organization={org_id}", auth=auth
    )
    results = data.get("results") or []
    inputs = {"username": GITEA_GIT_USER, "password": GITEA_GIT_PASSWORD}
    if results:
        cred_id = int(results[0]["id"])
        controller_api(
            "PATCH",
            f"/credentials/{cred_id}/",
            {"credential_type": results[0]["credential_type"], "inputs": inputs},
            auth=auth,
        )
        return cred_id
    ctypes = controller_api(
        "GET",
        f"/credential_types/?search={urllib.parse.quote('Source Control')}",
        auth=auth,
    )
    ct_results = ctypes.get("results") or []
    if not ct_results:
        die("Source Control credential type not found in AAP")
    ct_id = int(ct_results[0]["id"])
    created = controller_api(
        "POST",
        "/credentials/",
        {
            "name": name,
            "organization": org_id,
            "credential_type": ct_id,
            "inputs": inputs,
        },
        auth=auth,
    )
    return int(created["id"])


def get_or_create_credential(token: str, org_id: int) -> int:
    auth = f"Bearer {token}"
    name = "netobserv-openshift"
    data = controller_api(
        "GET", f"/credentials/?name={name}&organization={org_id}", auth=auth
    )
    results = data.get("results") or []
    inputs = {
        "host": "https://kubernetes.default.svc",
        "bearer_token": sa_token(),
        "verify_ssl": False,
    }
    if results:
        cred_id = int(results[0]["id"])
        controller_api(
            "PATCH",
            f"/credentials/{cred_id}/",
            {
                "credential_type": results[0]["credential_type"],
                "inputs": inputs,
            },
            auth=auth,
        )
        return cred_id

    ctypes = controller_api(
        "GET",
        f"/credential_types/?name={urllib.parse.quote('OpenShift or Kubernetes API Bearer Token')}",
        auth=auth,
    )
    ct_results = ctypes.get("results") or []
    if not ct_results:
        ctypes = controller_api("GET", "/credential_types/?search=kubernetes", auth=auth)
        ct_results = ctypes.get("results") or []
    if not ct_results:
        die("OpenShift/Kubernetes credential type not found in AAP")
    ct_id = int(ct_results[0]["id"])

    created = controller_api(
        "POST",
        "/credentials/",
        {
            "name": name,
            "organization": org_id,
            "credential_type": ct_id,
            "inputs": inputs,
        },
        auth=auth,
    )
    return int(created["id"])


def attach_job_template_credential(token: str, jt_id: int, cred_id: int) -> None:
    """AAP 2.x ignores credentials[] on PATCH — attach via sub-resource."""
    auth = f"Bearer {token}"
    attached = controller_api("GET", f"/job_templates/{jt_id}/credentials/", auth=auth)
    for row in attached.get("results") or []:
        if int(row.get("id") or 0) == cred_id:
            return
    controller_api(
        "POST",
        f"/job_templates/{jt_id}/credentials/",
        {"id": cred_id},
        auth=auth,
    )


def get_or_create_job_template(
    token: str,
    org_id: int,
    project_id: int,
    cred_id: int,
    inv_id: int,
    *,
    name: str,
    playbook: str,
) -> int:
    auth = f"Bearer {token}"
    data = controller_api(
        "GET", f"/job_templates/?name={name}&organization={org_id}", auth=auth
    )
    results = data.get("results") or []
    body = {
        "name": name,
        "organization": org_id,
        "project": project_id,
        "inventory": inv_id,
        "playbook": playbook,
        "credentials": [cred_id],
        # MCP passes confirmed=true at launch; template must not pin confirmed=false.
        "ask_variables_on_launch": True,
        "extra_vars": "",
    }
    if results:
        jt_id = int(results[0]["id"])
        controller_api("PATCH", f"/job_templates/{jt_id}/", body, auth=auth)
        attach_job_template_credential(token, jt_id, cred_id)
        return jt_id
    created = controller_api("POST", "/job_templates/", body, auth=auth)
    jt_id = int(created["id"])
    attach_job_template_credential(token, jt_id, cred_id)
    return jt_id


def create_launcher_token(admin_bearer: str) -> str:
    data = gateway_api(
        "POST",
        "/tokens/",
        {"description": "openclaw-mcp-launcher"},
        auth=f"Bearer {admin_bearer}",
    )
    launcher = data.get("token") or data.get("access_token")
    if not launcher:
        die(f"launcher token create failed: {data}")
    return str(launcher)


def main() -> int:
    admin_token = get_or_create_token()
    org_id = get_or_create_org(admin_token)
    scm_cred_id = get_or_create_gitea_credential(admin_token, org_id)
    project_id = get_or_create_project(admin_token, org_id, scm_cred_id)
    cred_id = get_or_create_credential(admin_token, org_id)
    inv_id = get_or_create_inventory(admin_token, org_id)
    heal_jt = get_or_create_job_template(
        admin_token,
        org_id,
        project_id,
        cred_id,
        inv_id,
        name="netobserv-heal-db-path",
        playbook="playbooks/netobserv-heal-db-path.yml",
    )
    policy_jt = get_or_create_job_template(
        admin_token,
        org_id,
        project_id,
        cred_id,
        inv_id,
        name="netobserv-restore-policy",
        playbook="playbooks/netobserv-restore-policy.yml",
    )
    launcher = create_launcher_token(admin_token)
    print(
        json.dumps(
            {
                "organization": ORG,
                "project_id": project_id,
                "credential_id": cred_id,
                "heal_job_template_id": heal_jt,
                "policy_job_template_id": policy_jt,
                "launcher_token": launcher,
                "gateway_url": _gateway_root(),
                "controller_api": _controller_base(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
