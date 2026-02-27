-- Time-series: metrics aggregated by metric_name and hour
SELECT metric_name, date_trunc(ts, 'hour') AS hour, COUNT(*) AS cnt, AVG(value) AS avg_val
FROM metrics
WHERE ts >= NOW() - INTERVAL 7 DAY
GROUP BY metric_name, date_trunc(ts, 'hour')
ORDER BY hour DESC, metric_name
LIMIT 100;
