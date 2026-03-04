SELECT metric_name, TIME_FLOOR(__time, 'PT1H') AS hour_bucket, COUNT(*) AS cnt, AVG("value") AS avg_val
FROM "telemetry_metrics"
WHERE __time >= CURRENT_TIMESTAMP - INTERVAL '7' DAY
GROUP BY metric_name, TIME_FLOOR(__time, 'PT1H')
ORDER BY TIME_FLOOR(__time, 'PT1H') DESC, metric_name
LIMIT 100
