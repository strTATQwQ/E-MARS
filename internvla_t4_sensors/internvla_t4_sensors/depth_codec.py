"""Bounded lossless transport decoder for T4 metric-depth frames."""

from __future__ import annotations

import array
import base64
import binascii
import sys
import zlib
from typing import Any


DEPTH_ENCODING = "uint16_mm_zlib_b64_v1"
MAX_DEPTH_PIXELS = 640 * 480
MAX_DEPTH_BASE64_BYTES = 480 * 1024


def _dimensions(request: dict[str, Any]) -> tuple[int, int, int]:
    height = int(request.get("depth_height", 0))
    width = int(request.get("depth_width", 0))
    count = height * width
    if height <= 0 or width <= 0 or count > MAX_DEPTH_PIXELS:
        raise ValueError("invalid bounded metric depth dimensions")
    return height, width, count


def decode_depth_request(request: dict[str, Any]) -> list[float] | None:
    """Decode one depth payload while rejecting ambiguity and expansion bombs."""

    legacy = request.get("depth_values")
    encoded = request.get("depth_zlib_b64")
    if legacy is not None and encoded is not None:
        raise ValueError("ambiguous metric depth payload")
    if encoded is None:
        if legacy is None:
            return None
        if not isinstance(legacy, list):
            raise ValueError("legacy metric depth payload must be a list")
        if not legacy:
            return []
        _, _, count = _dimensions(request)
        if len(legacy) != count:
            raise ValueError("legacy metric depth sample count mismatch")
        return [float(value) for value in legacy]

    if request.get("depth_encoding") != DEPTH_ENCODING:
        raise ValueError("unsupported metric depth encoding")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("compressed metric depth payload is empty")
    if len(encoded.encode("ascii", errors="strict")) > MAX_DEPTH_BASE64_BYTES:
        raise ValueError("compressed metric depth payload exceeds bound")
    _, _, count = _dimensions(request)
    expected_bytes = count * 2
    if int(request.get("depth_uncompressed_bytes", -1)) != expected_bytes:
        raise ValueError("metric depth byte-count claim mismatch")
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid metric depth base64") from exc
    if int(request.get("depth_compressed_bytes", -1)) != len(compressed):
        raise ValueError("metric depth compressed-byte claim mismatch")

    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(compressed, expected_bytes + 1)
    except zlib.error as exc:
        raise ValueError("invalid compressed metric depth stream") from exc
    if (
        len(raw) != expected_bytes
        or not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
    ):
        raise ValueError("metric depth expansion is incomplete or exceeds bound")

    millimeters = array.array("H")
    millimeters.frombytes(raw)
    if sys.byteorder != "little":
        millimeters.byteswap()
    if len(millimeters) != count:
        raise ValueError("metric depth decoded sample count mismatch")
    return [value / 1000.0 for value in millimeters]
