-- Recent activity: last 100 logs
SELECT ts, service, level, LEFT(message, 150) AS message_preview
FROM logs
ORDER BY ts DESC
LIMIT 100;
