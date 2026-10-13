-- Log search: find logs containing 'error' in message
SELECT ts, service, level, LEFT(message, 200) AS message_preview
FROM logs
WHERE ts >= NOW() - INTERVAL 30 DAY
  AND (message LIKE '%error%' OR message LIKE '%Error%' OR message LIKE '%ERROR%')
ORDER BY ts DESC
LIMIT 100;
