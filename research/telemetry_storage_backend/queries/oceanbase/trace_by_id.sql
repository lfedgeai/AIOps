USE telemetry;
SELECT trace_id, span_id, service, name, ts_start, duration_ms
FROM spans
WHERE trace_id = (SELECT trace_id FROM spans LIMIT 1)
ORDER BY ts_start;
