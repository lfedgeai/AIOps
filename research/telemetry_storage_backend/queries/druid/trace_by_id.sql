SELECT trace_id, span_id, service, name, __time AS ts_start, duration_ms
FROM "telemetry_spans"
WHERE trace_id = (SELECT trace_id FROM "telemetry_spans" LIMIT 1)
ORDER BY __time
