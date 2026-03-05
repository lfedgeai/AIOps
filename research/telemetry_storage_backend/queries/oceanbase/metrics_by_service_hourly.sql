USE telemetry;
SELECT metric_name, DATE_FORMAT(ts, '%Y-%m-%d %H:00:00') AS hour, COUNT(*) AS cnt, AVG(value) AS avg_val
FROM metrics
WHERE ts >= NOW() - INTERVAL 7 DAY
GROUP BY metric_name, DATE_FORMAT(ts, '%Y-%m-%d %H:00:00')
ORDER BY hour DESC, metric_name
LIMIT 100;
