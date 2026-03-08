# OTEL Demo — Working URLs

## Application Access

| Service | URL | Status |
|---------|-----|--------|
| **Astronomy Shop (main app)** | https://otel-demo-frontend-default.apps.sno1gpu.localdomain/ | ✅ |
| **Flagd UI (chaos/fault injection)** | https://otel-demo-frontend-default.apps.sno1gpu.localdomain/feature | ✅ |
| **Flagd API (harness read/write)** | https://flagd-ui-api-default.apps.sno1gpu.localdomain/api/read, /api/write | ✅ |
| **Grafana** | https://otel-demo-frontend-default.apps.sno1gpu.localdomain/grafana/ | ✅ |
| **Grafana (direct)** | https://grafana-otel-demo.apps.sno1gpu.localdomain | ✅ |
| **Jaeger** | https://otel-demo-frontend-default.apps.sno1gpu.localdomain/jaeger/ | ✅ |
| **Load Generator** | https://otel-demo-frontend-default.apps.sno1gpu.localdomain/loadgen/ | ✅ |

## Notes

- Use **HTTPS** (not HTTP). The Route uses edge TLS termination.
- Replace `sno1gpu.localdomain` with your cluster's app domain if different.
- To get your cluster's routes: `oc get route -n otel-demo`
