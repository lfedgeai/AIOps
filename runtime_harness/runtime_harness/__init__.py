"""LF Edge AIOps Runtime Harness — pluggable end-to-end assembly.

Package map (how pieces connect)
--------------------------------
``assembly``     YAML → wired slots + seed stage model selection
``orchestrator`` runs enrich → telemetry → detect → RCA → remediate → verify
``stage_models`` which LLM for Anomaly Detection / RCA / Remediation
``prompts``      what we ask those LLMs (UI-editable)
``telemetry``    ClickHouse OTEL client (app self-adoption + signals)
``mcp``          external MCP tools over the same telemetry client
``slots/*``      pluggable stage implementations
``api`` / ``cli`` status UI, config APIs, and operator commands
"""

__version__ = "0.1.0"
