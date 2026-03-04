SELECT __time, service, level, SUBSTRING(message, 1, 200) AS message_preview
FROM "telemetry_logs"
WHERE __time >= CURRENT_TIMESTAMP - INTERVAL '30' DAY
  AND (LOWER(message) LIKE '%error%')
ORDER BY __time DESC
LIMIT 100
