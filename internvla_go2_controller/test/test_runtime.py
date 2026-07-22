import math
import base64
import json
import socket
import zlib

import pytest

from internvla_go2_controller import runtime
from internvla_go2_controller.runtime import ControllerIPCClient
from internvla_go2_controller.runtime import JerkLimitedTwist


def test_limits_velocity_acceleration_and_jerk() -> None:
    limiter = JerkLimitedTwist(
        max_linear=0.25,
        max_angular=0.8,
        max_linear_acceleration=0.6,
        max_angular_acceleration=1.8,
        max_linear_jerk=2.5,
        max_angular_jerk=7.0,
    )
    previous = limiter.value()
    for _ in range(200):
        current = limiter.step(10.0, -10.0, 0.025)
        assert abs(current.linear_x) <= 0.25 + 1e-12
        assert abs(current.angular_z) <= 0.8 + 1e-12
        assert abs(current.linear_acceleration) <= 0.6 + 1e-12
        assert abs(current.angular_acceleration) <= 1.8 + 1e-12
        assert abs(current.linear_acceleration - previous.linear_acceleration) <= 2.5 * 0.025 + 1e-12
        assert abs(current.angular_acceleration - previous.angular_acceleration) <= 7.0 * 0.025 + 1e-12
        previous = current


def test_emergency_and_nonfinite_are_immediate_zero() -> None:
    limiter = JerkLimitedTwist()
    for _ in range(20):
        limiter.step(0.2, 0.4, 0.025)
    assert limiter.step(0.2, 0.4, 0.025, emergency_stop=True).linear_x == 0.0
    assert limiter.step(math.nan, 0.0, 0.025).angular_z == 0.0


def test_rejects_invalid_dt() -> None:
    with pytest.raises(ValueError):
        JerkLimitedTwist().step(0.0, 0.0, 0.0)


def test_obstacle_scenario_preserves_optional_route_yaw() -> None:
    runtime.set_obstacle_scenario("doorway", 7, route_yaw_world=1.25)
    assert runtime.get_obstacle_scenario() == runtime.ObstacleScenario(
        "doorway", 7, 1.25
    )
    runtime.reset_obstacle_scenario(8)
    assert runtime.get_obstacle_scenario() == runtime.ObstacleScenario("", 8, None)


def test_ipc_reconnects_once_after_idle_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    class Connection:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    first = Connection()
    second = Connection()
    client = ControllerIPCClient("unused")
    client.connection = first  # type: ignore[assignment]
    sent: list[Connection] = []

    def reconnect() -> Connection:
        client.connection = second  # type: ignore[assignment]
        return second

    def send(connection: Connection, _request: dict[str, object]) -> None:
        sent.append(connection)
        if connection is first:
            raise BrokenPipeError("server closed idle connection")

    monkeypatch.setattr(client, "_connect", reconnect)
    monkeypatch.setattr(runtime, "_send_json", send)
    monkeypatch.setattr(
        runtime,
        "_recv_json",
        lambda _connection: {
            "status": "ok",
            "linear_x": 0.0,
            "angular_z": 0.0,
            "emergency_stop": True,
        },
    )

    response = client.exchange({"operation": "update"})

    assert response["status"] == "ok"
    assert first.closed
    assert sent == [first, second]


def test_ipc_endpoint_defaults_to_unix_and_accepts_bounded_ipv4_tcp() -> None:
    unix = ControllerIPCClient("/tmp/controller.sock")
    assert unix.transport == "unix"
    assert unix.address == "/tmp/controller.sock"

    tcp = ControllerIPCClient("tcp://10.100.100.128:24137", timeout_sec=5.0)
    assert tcp.transport == "tcp"
    assert tcp.address == ("10.100.100.128", 24137)


@pytest.mark.parametrize(
    "endpoint",
    [
        "tcp://dgx-spark:24137",
        "tcp://10.100.100.128",
        "tcp://10.100.100.128:80",
        "tcp://user@10.100.100.128:24137",
        "tcp://10.100.100.128:24137/path",
        "tcp://0.0.0.0:24137?unsafe=1",
        "tcp://0.0.0.0:24137",
        "tcp://224.0.0.1:24137",
        "tcp://[::1]:24137",
    ],
)
def test_ipc_endpoint_rejects_ambiguous_or_unbounded_tcp(endpoint: str) -> None:
    with pytest.raises(ValueError):
        ControllerIPCClient(endpoint)


def test_tcp_ipc_client_connects_to_configured_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, tuple[str, int]]] = []

    class Connection:
        def settimeout(self, value: float) -> None:
            assert value == 3.0

        def connect(self, address: tuple[str, int]) -> None:
            calls.append((socket.AF_INET, address))

        def close(self) -> None:
            pass

    monkeypatch.setattr(runtime.socket, "socket", lambda family, _kind: Connection())
    client = ControllerIPCClient("tcp://10.100.100.128:24137", timeout_sec=3.0)
    assert client._connect() is client.connection
    assert calls == [(socket.AF_INET, ("10.100.100.128", 24137))]


def test_first_real_sensor_frame_is_recorded_only_after_dgx_ack(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    target = tmp_path / "sensor_frames.jsonl"
    monkeypatch.setenv("INTERNVLA_T5_SENSOR_FRAME_READY_RECORD", str(target))
    client = ControllerIPCClient("tcp://10.100.100.128:24137")
    monkeypatch.setattr(client, "_connect", lambda: object())
    monkeypatch.setattr(runtime, "_send_json", lambda *_args: None)
    monkeypatch.setattr(runtime, "_recv_json", lambda *_args: {"status": "ok"})
    request = {
        "schema_version": 1,
        "operation": "update",
        "episode_id": "a::433_121",
        "reset_generation": 0,
        "sequence_id": 7,
        "depth_height": 1,
        "depth_width": 2,
        "depth_values": [1.0, 2.0],
    }

    assert client.exchange(request)["status"] == "ok"
    value = json.loads(target.read_text(encoding="utf-8"))
    assert value["status"] == "PASS"
    assert value["episode_id"].startswith("a::")
    assert value["depth_sample_count"] == 2
    assert "depth_values" not in value
    assert value["depth_transport_encoding"] == "legacy_float_list"

    client.exchange({**request, "sequence_id": 8})
    assert len(target.read_text(encoding="utf-8").splitlines()) == 1


def test_disabled_t5_sensor_readiness_never_scans_t4_depth_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DepthPayloadThatMustNotBeInspected(list[float]):
        def __iter__(self):
            raise AssertionError("disabled T5 readiness inspected the T4 depth frame")

    monkeypatch.delenv("INTERNVLA_T5_SENSOR_FRAME_READY_RECORD", raising=False)
    client = ControllerIPCClient("tcp://10.100.100.128:24137")
    monkeypatch.setattr(client, "_connect", lambda: object())
    monkeypatch.setattr(runtime, "_send_json", lambda *_args: None)
    monkeypatch.setattr(runtime, "_recv_json", lambda *_args: {"status": "ok"})

    response = client.exchange(
        {
            "operation": "update",
            "depth_height": 480,
            "depth_width": 640,
            "depth_values": DepthPayloadThatMustNotBeInspected(),
        }
    )

    assert response["status"] == "ok"


def test_state_only_or_failed_sensor_exchange_never_claims_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    target = tmp_path / "sensor_frames.jsonl"
    monkeypatch.setenv("INTERNVLA_T5_SENSOR_FRAME_READY_RECORD", str(target))
    client = ControllerIPCClient("tcp://10.100.100.128:24137")
    monkeypatch.setattr(client, "_connect", lambda: object())
    monkeypatch.setattr(runtime, "_send_json", lambda *_args: None)
    monkeypatch.setattr(runtime, "_recv_json", lambda *_args: {"status": "ok"})
    request = {
        "operation": "update",
        "state_only": True,
        "depth_height": 1,
        "depth_width": 1,
        "depth_values": [1.0],
    }
    client.exchange(request)
    assert not target.exists()

    monkeypatch.setattr(
        runtime, "_recv_json", lambda *_args: {"status": "error", "message": "no"}
    )
    with pytest.raises(RuntimeError):
        client.exchange({**request, "state_only": False})
    assert not target.exists()


@pytest.mark.parametrize(
    "depth_values",
    ([1.0], [1.0, float("nan")], [1.0, float("inf")], [1.0, True]),
)
def test_incomplete_or_nonfinite_depth_never_claims_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path, depth_values,
) -> None:
    target = tmp_path / "sensor_frames.jsonl"
    monkeypatch.setenv("INTERNVLA_T5_SENSOR_FRAME_READY_RECORD", str(target))
    client = ControllerIPCClient("tcp://10.100.100.128:24137")
    monkeypatch.setattr(client, "_connect", lambda: object())
    monkeypatch.setattr(runtime, "_send_json", lambda *_args: None)
    monkeypatch.setattr(runtime, "_recv_json", lambda *_args: {"status": "ok"})

    client.exchange(
        {
            "operation": "update",
            "depth_height": 1,
            "depth_width": 2,
            "depth_values": depth_values,
        }
    )
    assert not target.exists()


def test_compressed_r3_depth_frame_is_validated_before_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    target = tmp_path / "sensor_frames.jsonl"
    monkeypatch.setenv("INTERNVLA_T5_SENSOR_FRAME_READY_RECORD", str(target))
    client = ControllerIPCClient("tcp://10.100.100.128:24137")
    monkeypatch.setattr(client, "_connect", lambda: object())
    monkeypatch.setattr(runtime, "_send_json", lambda *_args: None)
    monkeypatch.setattr(runtime, "_recv_json", lambda *_args: {"status": "ok"})
    raw = b"\x00\x00\xe8\x03\xd0\x07\x00\x00"
    compressed = zlib.compress(raw, level=1)

    client.exchange(
        {
            "operation": "update",
            "depth_height": 2,
            "depth_width": 2,
            "depth_encoding": "uint16_mm_zlib_b64_v1",
            "depth_zlib_b64": base64.b64encode(compressed).decode("ascii"),
            "depth_uncompressed_bytes": len(raw),
            "depth_compressed_bytes": len(compressed),
        }
    )
    value = json.loads(target.read_text(encoding="utf-8"))
    assert value["depth_sample_count"] == 4
    assert value["depth_transport_encoding"] == "uint16_mm_zlib_b64_v1"
    assert value["nonempty_depth"] is True


@pytest.mark.parametrize("mutation", ("wrong_bytes", "trailing_stream", "all_zero"))
def test_invalid_compressed_depth_never_claims_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path, mutation,
) -> None:
    target = tmp_path / "sensor_frames.jsonl"
    monkeypatch.setenv("INTERNVLA_T5_SENSOR_FRAME_READY_RECORD", str(target))
    client = ControllerIPCClient("tcp://10.100.100.128:24137")
    monkeypatch.setattr(client, "_connect", lambda: object())
    monkeypatch.setattr(runtime, "_send_json", lambda *_args: None)
    monkeypatch.setattr(runtime, "_recv_json", lambda *_args: {"status": "ok"})
    raw = b"\x00\x00\x00\x00" if mutation == "all_zero" else b"\xe8\x03\xd0\x07"
    compressed = zlib.compress(raw, level=1)
    encoded = base64.b64encode(
        compressed + (b"trailing" if mutation == "trailing_stream" else b"")
    ).decode("ascii")
    request = {
        "operation": "update",
        "depth_height": 1,
        "depth_width": 2,
        "depth_encoding": "uint16_mm_zlib_b64_v1",
        "depth_zlib_b64": encoded,
        "depth_uncompressed_bytes": len(raw) + (2 if mutation == "wrong_bytes" else 0),
        "depth_compressed_bytes": len(base64.b64decode(encoded)),
    }

    client.exchange(request)
    assert not target.exists()
