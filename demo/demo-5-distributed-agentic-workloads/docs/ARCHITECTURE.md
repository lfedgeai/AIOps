# Architecture (demo-5)

See [AGENTS.md](../AGENTS.md) for the full contract.

```text
Sites (MicroShift/K3s)
  todo + postgres + loadgen + site OTEL collector
        | OTLP/HTTPS (OpenShift Route only)
        v
Hub demo5-observability
  otel-intake Route → hub collector → ClickHouse   ← AIOps OBSERVE handover
Hub demo5-aiops (Phase 4+)
  agents + MCP / gated tools → site APIs / OpenShell   ← AIOps ACT
```
