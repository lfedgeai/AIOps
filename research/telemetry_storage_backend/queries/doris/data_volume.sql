-- Full scan: count all rows in combined data (batch + OTLP)
SELECT 'logs' AS tbl, COUNT(*) AS cnt FROM logs
UNION ALL
SELECT 'spans', COUNT(*) FROM spans
UNION ALL
SELECT 'metrics', COUNT(*) FROM metrics;
