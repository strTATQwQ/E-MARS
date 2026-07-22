from __future__ import annotations

import json
import statistics
import struct
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from sensor_runtime.contract import REQUIRED_STREAMS
from sensor_runtime.core import SensorBatch
import sensor_runtime.wire as wire
from sensor_runtime.wire import MAGIC, LatestBatchEmitter, decode_batch, encode_batch


def _batch() -> SensorBatch:
    rng = np.random.default_rng(1234)
    stamp, render = 123_000_000, 7
    payloads = {
        "d435i_rgb": {
            "rgb8": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
            "front_rgb8": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
            "d435i_fx": 500.0, "d435i_fy": 510.0, "front_fx": 46.0, "front_fy": 78.0,
            "render_id": render, "render_generation": 0,
        },
        "d435i_depth": {
            "depth_m": rng.uniform(0.28, 6.0, (480, 640)).astype(np.float32),
            "fx": 400.0, "fy": 410.0, "cx": 319.5, "cy": 239.5,
            "pitch_down_deg": 20.0, "translation_from_base_m": [0.2, 0.0, 0.2],
            "support_plane_world_z": 0.0,
            "link_centers_base": [
                {"name": "base" if index == 0 else f"link_{index}", "center_base": [float(index), 0.0, 0.0]}
                for index in range(13)
            ],
            "render_id": render, "render_generation": 0,
        },
        "lidar": {"points_lidar": rng.uniform(-2, 2, (8, 180, 3)).astype(np.float32)},
        "pose": {
            "position": np.asarray([0.0, 0.0, 0.42], dtype=np.float64),
            "rotation_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "linear_velocity": np.zeros(3, dtype=np.float64),
            "angular_velocity": np.zeros(3, dtype=np.float64),
        },
        "tf": {
            "base_translation": [0.0, 0.0, 0.42],
            "base_rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "fixed": [],
        },
    }
    return SensorBatch(
        stamp_ns=stamp,
        generation=0,
        sequence=0,
        payloads=payloads,
        stream_stamps_ns={name: stamp for name in REQUIRED_STREAMS},
        safe_stop={
            "identity": "sensor-soak:0:0", "linear_x": 0.0, "angular_z": 0.0,
            "emergency_stop": True, "physics_step": 10, "applied_step_count": 10,
            "steps_since_previous_capture": 10, "render_id": render, "render_generation": 0,
            "reset_kind": "continuous_world_articulation_state",
        },
        reset_reason="initial",
    )


def _parts(packet: bytes) -> tuple[dict, bytes]:
    header_size = struct.unpack("!I", packet[len(MAGIC) : len(MAGIC) + 4])[0]
    start = len(MAGIC) + 4
    return json.loads(packet[start : start + header_size]), packet[start + header_size :]


def _rebuild(header: dict, raw: bytes) -> bytes:
    value = json.dumps(header, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
    return MAGIC + struct.pack("!I", len(value)) + value + raw


def _first_array(value):
    if isinstance(value, dict):
        if set(value) == {"$array"}:
            return value["$array"]
        for item in value.values():
            found = _first_array(item)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = _first_array(item)
            if found is not None:
                return found
    return None


def test_binary_wire_roundtrip_preserves_full_resolution_arrays() -> None:
    original = _batch()
    packet = encode_batch(original)
    assert packet.startswith(MAGIC)
    restored = decode_batch(packet)
    assert restored.stamp_ns == original.stamp_ns
    for stream, key in (("d435i_rgb", "rgb8"), ("d435i_rgb", "front_rgb8"), ("d435i_depth", "depth_m"), ("lidar", "points_lidar")):
        np.testing.assert_array_equal(restored.payloads[stream][key], original.payloads[stream][key])


@pytest.mark.parametrize("corruption", ["offset", "shape", "extra"])
def test_binary_wire_rejects_corrupt_segments(corruption: str) -> None:
    header, raw = _parts(encode_batch(_batch()))
    descriptor = _first_array(header)
    assert descriptor is not None
    if corruption == "offset":
        descriptor["offset"] += 1
    elif corruption == "shape":
        descriptor["shape"][0] += 1
    else:
        raw += b"x"
    with pytest.raises(ValueError):
        decode_batch(_rebuild(header, raw))


def test_binary_wire_rejects_nan_json_and_boolean_identity_metadata() -> None:
    header, raw = _parts(encode_batch(_batch()))
    header["stamp_ns"] = True
    with pytest.raises(ValueError, match="non-boolean integer"):
        decode_batch(_rebuild(header, raw))
    header, raw = _parts(encode_batch(_batch()))
    valid = _rebuild(header, raw)
    header_size = struct.unpack("!I", valid[len(MAGIC) : len(MAGIC) + 4])[0]
    start = len(MAGIC) + 4
    header_bytes = valid[start : start + header_size].replace(b'"stamp_ns":123000000', b'"stamp_ns":NaN')
    tampered = MAGIC + struct.pack("!I", len(header_bytes)) + header_bytes + valid[start + header_size :]
    with pytest.raises(ValueError, match="non-finite JSON"):
        decode_batch(tampered)


def test_binary_wire_representative_roundtrip_is_faster_than_10hz_budget() -> None:
    batch = _batch()
    elapsed: list[float] = []
    for _ in range(9):
        started = time.perf_counter()
        restored = decode_batch(encode_batch(batch))
        assert restored.sequence == 0
        elapsed.append(time.perf_counter() - started)
    assert statistics.median(elapsed) < 0.05, elapsed
    assert max(elapsed) < 0.10, elapsed


@pytest.mark.parametrize("value", [True, 0.0, -1.0, 5.1, float("inf"), "5"])
def test_emitter_rejects_unbounded_or_implicit_send_timeout(value: object) -> None:
    with pytest.raises(ValueError, match="within"):
        LatestBatchEmitter(Path("/tmp/not-used.sock"), send_timeout_sec=value)  # type: ignore[arg-type]


def test_emitter_keeps_connect_strict_and_applies_bounded_send_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSocket:
        def __init__(self) -> None:
            self.timeouts: list[float] = []
            self.connected = ""
            self.closed = False

        def settimeout(self, value: float) -> None:
            self.timeouts.append(value)

        def connect(self, path: str) -> None:
            self.connected = path

        def close(self) -> None:
            self.closed = True

    fake = FakeSocket()
    monkeypatch.setattr(wire.socket, "socket", lambda *_args: fake)
    emitter = LatestBatchEmitter(Path("/tmp/profile.sock"), send_timeout_sec=5.0)
    assert emitter._connect() is fake
    assert fake.connected == "/tmp/profile.sock"
    assert fake.timeouts == [0.2, 5.0]
    assert fake.closed is False
