# OTEL Demo — Working URLs

## Application Access

| Service | URL | Status |
|---------|-----|--------|
| **Astronomy Shop (main app)** | https://otel-demo-frontend-default.apps.your-cluster.example.com/ | ✅ |
| **Flagd UI (chaos/fault injection)** | https://otel-demo-frontend-default.apps.your-cluster.example.com/feature | ✅ |
| **Flagd API (harness read/write)** | https://flagd-ui-api-default.apps.your-cluster.example.com/api/read, /api/write | ✅ |
| **Grafana** | https://otel-demo-frontend-default.apps.your-cluster.example.com/grafana/ | ✅ |
| **Grafana (direct)** | https://grafana-otel-demo.apps.your-cluster.example.com | ✅ |
| **Jaeger** | https://otel-demo-frontend-default.apps.your-cluster.example.com/jaeger/ | ✅ |
| **Load Generator** | https://otel-demo-frontend-default.apps.your-cluster.example.com/loadgen/ | ✅ |

## Notes

- Use **HTTPS** (not HTTP). The Route uses edge TLS termination.
- Replace `your-cluster.example.com` with your cluster's app domain if different.
- To get your cluster's routes: `oc get route -n otel-demo`
