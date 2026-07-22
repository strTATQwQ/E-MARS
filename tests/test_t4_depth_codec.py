from __future__ import annotations

import array
import base64
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "internvla_t4_sensors"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from internvla_t4_sensors.depth_codec import DEPTH_ENCODING, decode_depth_request


def _compressed_request(values_mm: list[int], *, width: int) -> dict[str, object]:
    samples = array.array("H", values_mm)
    if sys.byteorder != "little":
        samples.byteswap()
    raw = samples.tobytes()
    compressed = zlib.compress(raw, level=1)
    return {
        "depth_height": len(values_mm) // width,
        "depth_width": width,
        "depth_encoding": DEPTH_ENCODING,
        "depth_zlib_b64": base64.b64encode(compressed).decode("ascii"),
        "depth_uncompressed_bytes": len(raw),
        "depth_compressed_bytes": len(compressed),
    }


def test_uint16_depth_codec_round_trips_millimeter_precision() -> None:
    request = _compressed_request([0, 281, 1234, 6000], width=2)
    assert decode_depth_request(request) == [0.0, 0.281, 1.234, 6.0]


def test_depth_codec_rejects_ambiguous_and_mismatched_payloads() -> None:
    request = _compressed_request([1000, 2000, 3000, 4000], width=2)
    request["depth_values"] = [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError, match="ambiguous"):
        decode_depth_request(request)

    request.pop("depth_values")
    request["depth_uncompressed_bytes"] = 7
    with pytest.raises(ValueError, match="byte-count"):
        decode_depth_request(request)


def test_depth_codec_rejects_trailing_stream_and_expansion_overflow() -> None:
    request = _compressed_request([1000, 2000, 3000, 4000], width=2)
    encoded = str(request["depth_zlib_b64"])
    compressed = base64.b64decode(encoded) + b"trailing"
    request["depth_zlib_b64"] = base64.b64encode(compressed).decode("ascii")
    request["depth_compressed_bytes"] = len(compressed)
    with pytest.raises(ValueError, match="expansion"):
        decode_depth_request(request)

    overflow = _compressed_request([1000, 2000, 3000, 4000, 5000], width=5)
    overflow["depth_height"] = 1
    overflow["depth_width"] = 4
    overflow["depth_uncompressed_bytes"] = 8
    with pytest.raises(ValueError, match="expansion"):
        decode_depth_request(overflow)


def test_depth_codec_keeps_legacy_fixture_compatibility() -> None:
    request = {
        "depth_height": 1,
        "depth_width": 2,
        "depth_values": [0.5, 1.25],
    }
    assert decode_depth_request(request) == [0.5, 1.25]
    assert decode_depth_request({}) is None
