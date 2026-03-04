-- Full scan: count all rows in combined data (batch + OTLP)
SELECT 'logs' AS tbl, COUNT(*) AS cnt FROM "telemetry_logs"
UNION ALL
SELECT 'spans', COUNT(*) FROM "telemetry_spans"
UNION ALL
SELECT 'metrics', COUNT(*) FROM "telemetry_metrics"
