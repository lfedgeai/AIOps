USE telemetry;
SELECT service, COUNT(*) AS err_count
FROM logs
WHERE ts >= NOW() - INTERVAL 1 DAY
  AND (level = 'error' OR message LIKE '%error%')
GROUP BY service
ORDER BY err_count DESC
LIMIT 20;

