"""
Minimal Prometheus remote-write encoder (protobuf + snappy).
Implements just enough of the WriteRequest proto to push samples
to VictoriaMetrics /api/v1/write.

Proto schema (from prometheus/prometheus):
  message WriteRequest { repeated TimeSeries timeseries = 1; }
  message TimeSeries   { repeated Label labels = 1; repeated Sample samples = 2; }
  message Label        { string name = 1; string value = 2; }
  message Sample       { double value = 1; int64 timestamp = 2; }
"""
from __future__ import annotations
import struct

import snappy


def _encode_varint(value: int) -> bytes:
    bits = value & 0x7F
    value >>= 7
    out = b""
    while value:
        out += bytes([0x80 | bits])
        bits = value & 0x7F
        value >>= 7
    out += bytes([bits])
    return out


def _encode_bytes(field_number: int, data: bytes) -> bytes:
    tag = _encode_varint((field_number << 3) | 2)
    return tag + _encode_varint(len(data)) + data


def _encode_double(field_number: int, value: float) -> bytes:
    tag = _encode_varint((field_number << 3) | 1)
    return tag + struct.pack("<d", value)


def _encode_sint64(field_number: int, value: int) -> bytes:
    tag = _encode_varint((field_number << 3) | 0)
    return tag + _encode_varint(value)


def _encode_label(name: str, value: str) -> bytes:
    inner = _encode_bytes(1, name.encode()) + _encode_bytes(2, value.encode())
    return _encode_bytes(1, inner)


def _encode_sample(value: float, timestamp_ms: int) -> bytes:
    inner = _encode_double(1, value) + _encode_sint64(2, timestamp_ms)
    return _encode_bytes(2, inner)


def encode_write_request(timeseries: list[dict]) -> bytes:
    """
    Encode a list of time series into a snappy-compressed WriteRequest.

    Each entry in timeseries: {"labels": {"__name__": "x", ...}, "value": float, "timestamp_ms": int}
    """
    body = b""
    for ts in timeseries:
        labels_bytes = b""
        for k, v in sorted(ts["labels"].items()):
            labels_bytes += _encode_label(k, v)
        sample_bytes = _encode_sample(ts["value"], ts["timestamp_ms"])
        ts_msg = labels_bytes + sample_bytes
        body += _encode_bytes(1, ts_msg)
    return snappy.compress(body)