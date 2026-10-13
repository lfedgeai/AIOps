SELECT ts, service, level, left(message,200) AS message_preview FROM telemetry.logs WHERE ts>=now()-INTERVAL 30 DAY AND positionCaseInsensitive(message,'error')>0 ORDER BY ts DESC LIMIT 100
