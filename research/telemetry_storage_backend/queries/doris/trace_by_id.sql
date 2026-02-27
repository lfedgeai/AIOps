-- Single-trace lookup: fetch all spans for one trace (uses subquery to pick a real trace_id)
SELECT trace_id, span_id, service, name, ts_start, duration_ms
FROM spans
WHERE trace_id = (SELECT trace_id FROM spans LIMIT 1)
ORDER BY ts_start;
