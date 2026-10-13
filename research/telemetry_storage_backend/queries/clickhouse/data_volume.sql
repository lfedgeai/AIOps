-- Full scan: count all rows in combined data (batch + OTLP)
SELECT 'logs' AS tbl, count() AS cnt FROM telemetry.logs
UNION ALL
SELECT 'spans', count() FROM telemetry.spans
UNION ALL
SELECT 'metrics', count() FROM telemetry.metrics
