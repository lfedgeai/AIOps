USE telemetry;
-- p95 latency by metric_name (labels omitted to avoid JSON func dependency)
SELECT metric_name,
       PERCENTILE_APPROX(value, 0.95) AS p95
FROM metrics
WHERE ts >= NOW() - INTERVAL 365 DAY
GROUP BY metric_name
ORDER BY p95 DESC
LIMIT 20;

