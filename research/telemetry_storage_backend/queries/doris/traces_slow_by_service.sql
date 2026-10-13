USE telemetry;
SELECT service, COUNT(*) AS slow_cnt
FROM spans
WHERE ts_start >= NOW() - INTERVAL 1 DAY
  AND duration_ms > 500
GROUP BY service
ORDER BY slow_cnt DESC
LIMIT 20;

