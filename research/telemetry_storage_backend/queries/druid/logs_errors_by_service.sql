SELECT service, COUNT(*) AS err_count
FROM "telemetry_logs"
WHERE __time >= CURRENT_TIMESTAMP - INTERVAL '1' DAY
  AND (level = 'error' OR LOWER(message) LIKE '%error%')
GROUP BY service
ORDER BY err_count DESC
LIMIT 20
