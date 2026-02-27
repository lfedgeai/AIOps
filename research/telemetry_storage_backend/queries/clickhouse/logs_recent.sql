SELECT ts, service, level, left(message,150) AS message_preview FROM telemetry.logs ORDER BY ts DESC LIMIT 100
