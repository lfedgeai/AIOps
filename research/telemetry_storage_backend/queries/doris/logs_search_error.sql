-- Log search: find logs containing the token 'error' (any case) via the idx_message full-text index.
-- The index is lower-cased, so one MATCH_ANY replaces the three case-variant LIKE scans.
SELECT ts, service, level, LEFT(message, 200) AS message_preview
FROM logs
WHERE ts >= NOW() - INTERVAL 30 DAY
  AND message MATCH_ANY 'error'
ORDER BY ts DESC
LIMIT 100;
