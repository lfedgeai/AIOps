#!/usr/bin/env python3
"""AAP Controller API client (stdlib only)."""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def _base_url() -> str:
    url = os.environ.get("AAP_CONTROLLER_URL", "").rstrip("/")
    if not url:
        raise RuntimeError("AAP_CONTROLLER_URL not set")
    for suffix in ("/api/controller/v2", "/api/gateway/v1", "/api/controller"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url.rstrip("/")


def _controller_url(path: str) -> str:
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{_base_url()}/api/controller/v2{path}"


def _token() -> str:
    tok = os.environ.get("AAP_TOKEN", "").strip()
    if not tok:
        raise RuntimeError("AAP_TOKEN not set (openclaw-aap-launcher secret)")
    return tok


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if os.environ.get("AAP_TLS_SKIP_VERIFY", "").lower() in ("1", "true", "yes"):
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def request(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    timeout: int = 120,
) -> Any:
    url = _controller_url(path)
    headers = {
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/json",
    }
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"AAP HTTP {e.code} {path}: {detail[:800]}") from e


def list_job_templates(organization: str | None = None) -> list[dict[str, Any]]:
    path = "/job_templates/"
    if organization:
        path += f"?organization__name={urllib.parse.quote(organization)}"
    data = request("GET", path)
    return list(data.get("results") or [])


def find_job_template_id(name: str) -> int:
    data = request("GET", f"/job_templates/?name={urllib.parse.quote(name)}")
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"AAP job template not found: {name}")
    return int(results[0]["id"])


def launch_job_template(
    name: str,
    extra_vars: dict[str, Any] | None = None,
    *,
    poll: bool = True,
    poll_timeout: int = 180,
    poll_interval: int = 5,
) -> dict[str, Any]:
    jt_id = find_job_template_id(name)
    payload: dict[str, Any] = {}
    if extra_vars:
        payload["extra_vars"] = extra_vars
    launched = request("POST", f"/job_templates/{jt_id}/launch/", payload)
    job_id = launched.get("id")
    if not job_id:
        raise RuntimeError(f"AAP launch returned no job id: {launched}")
    result: dict[str, Any] = {
        "job_id": int(job_id),
        "job_template": name,
        "job_url": f"{_base_url()}/#/jobs/playbook/{job_id}/output",
        "status": "running",
    }
    if not poll:
        return result
    end = time.time() + poll_timeout
    status = "running"
    while time.time() < end:
        job = request("GET", f"/jobs/{job_id}/")
        status = str(job.get("status") or "unknown")
        if status in ("successful", "failed", "error", "canceled"):
            break
        time.sleep(poll_interval)
    result["status"] = status
    result["highlights"] = _stdout_highlights(int(job_id))
    return result


def job_status(job_id: int, *, include_stdout: bool = False) -> dict[str, Any]:
    job = request("GET", f"/jobs/{job_id}/")
    out: dict[str, Any] = {
        "job_id": job_id,
        "status": job.get("status"),
        "job_template": (job.get("summary_fields") or {}).get("job_template", {}).get("name"),
        "started": job.get("started"),
        "finished": job.get("finished"),
        "job_url": f"{_base_url()}/#/jobs/playbook/{job_id}/output",
    }
    if include_stdout:
        out["highlights"] = _stdout_highlights(job_id)
    return out


def _stdout_highlights(job_id: int) -> list[str]:
    stdout = request("GET", f"/jobs/{job_id}/stdout/?format=json")
    content = ""
    if isinstance(stdout, dict):
        content = str(stdout.get("content") or "")
    elif isinstance(stdout, str):
        content = stdout
    highlights: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(
            ("VERDICT:", "HEAL_RESULT:", "POLICY_VERDICT:", "sample=", "AAP_", "TASK [")
        ):
            highlights.append(stripped)
    return highlights[:40]


def format_launch_result(result: dict[str, Any]) -> str:
    lines = [
        f"AAP_JOB_ID={result.get('job_id')}",
        f"AAP_JOB_TEMPLATE={result.get('job_template')}",
        f"AAP_JOB_URL={result.get('job_url')}",
        f"AAP_JOB_STATUS={result.get('status')}",
    ]
    for h in result.get("highlights") or []:
        lines.append(h)
    if result.get("status") not in ("successful", "running", None):
        lines.append(f"AAP_JOB_FAILED: status={result.get('status')}")
    return "\n".join(lines)
