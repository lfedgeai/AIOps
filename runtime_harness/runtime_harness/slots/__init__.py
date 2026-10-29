"""Pluggable AIOps pipeline slots.

Each submodule is one stage factory used by ``assembly.Assembly``:
telemetry → enrichment → detector (anomaly) → rca → remediator → verifier.
LLM stages read live model ids from ``stage_models`` and prompts from ``prompts``.
"""
