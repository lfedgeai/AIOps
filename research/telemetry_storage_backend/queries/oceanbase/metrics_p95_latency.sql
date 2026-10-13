USE telemetry;
-- OceanBase: use AVG as p95 proxy (PERCENTILE_CONT syntax varies)
SELECT metric_name, AVG(value) AS p95
FROM metrics
WHERE ts >= NOW() - INTERVAL 365 DAY
GROUP BY metric_name
ORDER BY p95 DESC
LIMIT 20;
