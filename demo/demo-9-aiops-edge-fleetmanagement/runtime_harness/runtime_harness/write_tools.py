"""Guarded Kubernetes write tools — single source of truth.

Purpose
-------
The four remediation writes with the Oct-26 validated guardrails, shared by
the MCP server and the Direct remediation mode. The harness (never the
model) owns execution.

Guardrails:
  G1: set_env may only OVERRIDE env vars that already exist on the deployment
      (hallucinated names must not mutate live state)
  G2: no-op patches return Error (a write that changes nothing is not a
      remediation and earns no exec credit)
  G3: 409-Conflict is retried with backoff (rollout bumps resourceVersion)

Namespace discipline: omit ``namespace`` to target the workload namespace;
404s return a self-healing hint naming the correct namespace.

How it connects
---------------
- ``mcp/server.py`` exposes these as MCP tools (gated by
  ``AIOPS_RT_ENABLE_WRITE_TOOLS``).
- ``remediation.py`` Direct mode executes validated tool selections through
  ``execute_write``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

GUARDED_WRITE_TOOLS = ("scale_deployment", "restart_deployment", "delete_pod", "patch_deployment")

WORKLOAD_NAMESPACE = os.environ.get("AIOPS_RT_NAMESPACE", "otel-demo")


def writes_armed() -> bool:
    """Direct/MCP writes require an explicit arming key."""
    return os.environ.get("AIOPS_RT_ENABLE_WRITE_TOOLS", "") == "1"


def _k8s():
    from kubernetes import client as k8s_client, config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except Exception:  # noqa: BLE001
        k8s_config.load_kube_config()
    return k8s_client


def _not_found(exc: Exception) -> bool:
    body = str(exc)
    return "404" in body or "not found" in body.lower()


def execute_write(name: str, args: dict[str, Any], namespace: Optional[str] = None) -> str:
    """Execute one guarded write tool. Returns a result string (Error: … on failure)."""
    ns = namespace or args.get("namespace") or WORKLOAD_NAMESPACE
    args = args or {}
    try:
        if name == "scale_deployment":
            apps = _k8s().AppsV1Api()
            apps.patch_namespaced_deployment_scale(
                str(args["name"]), ns, {"spec": {"replicas": int(args.get("replicas", 1))}}
            )
            return f"scaled {args['name']} to {args.get('replicas', 1)} replicas in {ns}"

        if name == "restart_deployment":
            apps = _k8s().AppsV1Api()
            apps.patch_namespaced_deployment(
                str(args["name"]),
                ns,
                {
                    "spec": {
                        "template": {
                            "metadata": {
                                "annotations": {
                                    "kubectl.kubernetes.io/restartedAt": time.strftime(
                                        "%Y-%m-%dT%H:%M:%SZ"
                                    )
                                }
                            }
                        }
                    }
                },
            )
            return f"restarted {args['name']} in {ns}"

        if name == "delete_pod":
            core = _k8s().CoreV1Api()
            core.delete_namespaced_pod(str(args["name"]), ns)
            return f"deleted pod {args['name']} in {ns}"

        if name == "patch_deployment":
            return _patch_deployment(
                str(args["name"]),
                ns,
                set_env=args.get("set_env") or {},
                remove_env=args.get("remove_env") or [],
            )

        return f"Error: unknown write tool '{name}' (allowed: {GUARDED_WRITE_TOOLS})"
    except Exception as exc:  # noqa: BLE001
        if _not_found(exc):
            return (
                f"Error: resource not found in namespace '{ns}' — the workload "
                f"namespace is '{WORKLOAD_NAMESPACE}'."
            )
        return f"Error: {exc}"


def _patch_deployment(
    name: str, ns: str, set_env: dict[str, str], remove_env: list[str]
) -> str:
    if not set_env and not remove_env:
        return "Error: no-op patch — provide set_env and/or remove_env"
    apps = _k8s().AppsV1Api()
    for attempt in range(4):
        try:
            dep = apps.read_namespaced_deployment(name, ns)
            changed = False
            for c in dep.spec.template.spec.containers:
                env = {(e.name or ""): e for e in (c.env or [])}
                for k in remove_env:
                    if k in env:
                        c.env = [e for e in c.env if e.name != k]
                        changed = True
                        env = {(e.name or ""): e for e in (c.env or [])}
                if set_env:
                    from kubernetes.client import V1EnvVar

                    for k, v in set_env.items():
                        if k in env:  # G1: override existing only
                            env[k].value = v
                            changed = True
            if not changed:
                return (
                    "Error: patch changed nothing (G2: no-op patches do not count as "
                    "remediation). set_env keys must already exist."
                )
            apps.replace_namespaced_deployment(name, ns, dep)
            return f"patched {name} in {ns} (set={list(set_env)}, removed={remove_env})"
        except Exception as exc:  # noqa: BLE001
            if "409" in str(exc) and attempt < 3:
                time.sleep(2 ** attempt)  # G3
                continue
            raise
    return "Error: patch retries exhausted (409 conflicts)"


__all__ = ["GUARDED_WRITE_TOOLS", "WORKLOAD_NAMESPACE", "writes_armed", "execute_write"]
