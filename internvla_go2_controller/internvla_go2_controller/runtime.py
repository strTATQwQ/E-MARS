"""Isaac-safe controller state, IPC, and jerk-limited twist primitives.

This module deliberately has no ROS imports so Isaac Sim's Python can use it.
"""

from __future__ import annotations

import base64
import binascii
import json
import ipaddress
import math
import os
import socket
import struct
import threading
import time
import zlib
from pathlib import Path
from urllib.parse import urlsplit
from dataclasses import dataclass
from typing import Any


MAX_MESSAGE_BYTES = 512 * 1024


@dataclass(frozen=True)
class ExecutionIdentity:
    episode_id: str = "uninitialized"
    reset_generation: int = 0
    sequence_id: int = 0
    stop: bool = True


_identity_lock = threading.Lock()
_identity = ExecutionIdentity()


@dataclass(frozen=True)
class ObstacleScenario:
    name: str = ""
    reset_generation: int = -1
    route_yaw_world: float | None = None


_obstacle_lock = threading.Lock()
_obstacle_scenario = ObstacleScenario()


def set_execution_identity(
    episode_id: str,
    reset_generation: int,
    sequence_id: int,
    *,
    stop: bool,
) -> None:
    global _identity
    value = ExecutionIdentity(
        episode_id=str(episode_id),
        reset_generation=int(reset_generation),
        sequence_id=int(sequence_id),
        stop=bool(stop),
    )
    with _identity_lock:
        _identity = value


def reset_execution_identity(episode_id: str, reset_generation: int) -> None:
    set_execution_identity(episode_id, reset_generation, 0, stop=True)


def get_execution_identity() -> ExecutionIdentity:
    with _identity_lock:
        return _identity


def set_obstacle_scenario(
    name: str,
    reset_generation: int,
    route_yaw_world: float | None = None,
) -> None:
    global _obstacle_scenario
    value = ObstacleScenario(
        str(name),
        int(reset_generation),
        None if route_yaw_world is None else float(route_yaw_world),
    )
    with _obstacle_lock:
        _obstacle_scenario = value


def reset_obstacle_scenario(reset_generation: int) -> None:
    set_obstacle_scenario("", reset_generation)


def get_obstacle_scenario() -> ObstacleScenario:
    with _obstacle_lock:
        return _obstacle_scenario


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    output = bytearray()
    while len(output) < count:
        block = connection.recv(count - len(output))
        if not block:
            raise ConnectionError("Go2 controller IPC disconnected")
        output.extend(block)
    return bytes(output)


def _send_json(connection: socket.socket, value: dict[str, Any]) -> None:
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Go2 controller IPC request exceeds bound")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def _recv_json(connection: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", _recv_exact(connection, 4))[0]
    if size < 2 or size > MAX_MESSAGE_BYTES:
        raise ValueError("invalid Go2 controller IPC response size")
    value = json.loads(_recv_exact(connection, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Go2 controller IPC response is not an object")
    return value


class ControllerIPCClient:
    def __init__(self, socket_path: str, timeout_sec: float = 0.2):
        self.endpoint = str(socket_path)
        self.timeout_sec = float(timeout_sec)
        if self.timeout_sec <= 0.0:
            raise ValueError("controller IPC timeout must be positive")
        self.transport, self.address = self._parse_endpoint(self.endpoint)
        self.connection: socket.socket | None = None
        self._sensor_readiness_recorded = False

    @staticmethod
    def _parse_endpoint(endpoint: str) -> tuple[str, str | tuple[str, int]]:
        if not endpoint.startswith("tcp://"):
            if not endpoint or "\x00" in endpoint:
                raise ValueError("invalid controller Unix socket path")
            return "unix", endpoint
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "tcp"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("invalid controller TCP endpoint")
        try:
            host = str(ipaddress.ip_address(parsed.hostname or ""))
            port = parsed.port
        except ValueError as exc:
            raise ValueError("controller TCP endpoint must use an IP literal") from exc
        address = ipaddress.ip_address(host)
        if address.version != 4 or address.is_unspecified or address.is_multicast:
            raise ValueError("controller TCP endpoint must use bounded unicast IPv4")
        if not isinstance(port, int) or not 1024 <= port <= 65535:
            raise ValueError("controller TCP port is outside the allowed range")
        return "tcp", (host, port)

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            finally:
                self.connection = None

    def _connect(self) -> socket.socket:
        self.close()
        family = socket.AF_UNIX if self.transport == "unix" else socket.AF_INET
        connection = socket.socket(family, socket.SOCK_STREAM)
        connection.settimeout(self.timeout_sec)
        connection.connect(self.address)
        self.connection = connection
        return connection

    def exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        response: dict[str, Any] | None = None
        for attempt in range(2):
            connection = self.connection or self._connect()
            try:
                _send_json(connection, request)
                response = _recv_json(connection)
                break
            except OSError:
                self.close()
                if attempt:
                    raise
        if response is None:
            raise ConnectionError("Go2 controller IPC produced no response")
        if response.get("status") != "ok":
            raise RuntimeError(str(response.get("message", "controller bridge error")))
        self._record_first_real_sensor_frame(request, response)
        return response

    def query_active_identity(
        self, *, expected_episode_prefix: str = ""
    ) -> ExecutionIdentity:
        """Read the persistent DGX navigation identity without changing it.

        A T5 soak may keep the DGX client and controller bridge alive while a
        bounded evaluator process is replaced.  The replacement must bind its
        state-only bootstrap samples to the bridge's real reset generation;
        a process-local counter restarts at zero and is therefore not an
        identity source.
        """

        response = self.exchange(
            {
                "schema_version": 1,
                "operation": "query_active_identity",
            }
        )
        if response.get("operation") != "query_active_identity":
            raise RuntimeError("controller bridge returned the wrong IPC operation")
        episode_id = response.get("active_episode_id")
        reset_generation = response.get("active_reset_generation")
        sequence_id = response.get("active_sequence_id")
        if (
            not isinstance(episode_id, str)
            or not episode_id
            or isinstance(reset_generation, bool)
            or not isinstance(reset_generation, int)
            or reset_generation < 0
            or isinstance(sequence_id, bool)
            or not isinstance(sequence_id, int)
            or sequence_id < -1
        ):
            raise RuntimeError("controller bridge active identity is unavailable")
        if expected_episode_prefix and not episode_id.startswith(
            expected_episode_prefix
        ):
            raise RuntimeError("controller bridge active identity has the wrong lane")
        return ExecutionIdentity(
            episode_id=episode_id,
            reset_generation=reset_generation,
            sequence_id=sequence_id,
            stop=True,
        )

    def _record_first_real_sensor_frame(
        self, request: dict[str, Any], response: dict[str, Any]
    ) -> None:
        """Record the first simulator depth frame acknowledged by the DGX.

        The optional record is deliberately produced only after the controller
        round trip succeeds.  It therefore proves more than local Kit startup:
        a real simulator sensor payload crossed the lane's bounded TCP channel
        and was accepted by the DGX controller.  The payload itself is never
        copied into the readiness artifact.
        """

        if self._sensor_readiness_recorded:
            return
        output = os.environ.get("INTERNVLA_T5_SENSOR_FRAME_READY_RECORD", "")
        # This hook is T5-only and disabled by default.  Return before touching
        # the (potentially 640x480) depth payload so the frozen T4 controller
        # path pays no per-frame readiness cost.
        if not output:
            return
        height = int(request.get("depth_height", 0))
        width = int(request.get("depth_width", 0))
        sample_count = height * width
        values = request.get("depth_values")
        encoded = request.get("depth_zlib_b64")
        encoding = "legacy_float_list"
        payload_valid = False
        nonempty_depth = False
        if values is not None and encoded is None:
            payload_valid = (
                isinstance(values, list)
                and len(values) == sample_count
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) >= 0.0
                    for value in values
                )
            )
            nonempty_depth = bool(
                payload_valid and any(float(value) > 0.0 for value in values)
            )
        elif values is None and isinstance(encoded, str):
            encoding = "uint16_mm_zlib_b64_v1"
            try:
                encoded_bytes = encoded.encode("ascii")
                if len(encoded_bytes) > 480 * 1024:
                    raise ValueError("compressed depth readiness payload exceeds bound")
                compressed = base64.b64decode(encoded, validate=True)
                expected_bytes = sample_count * 2
                decoder = zlib.decompressobj()
                raw = decoder.decompress(compressed, expected_bytes + 1)
                payload_valid = (
                    request.get("depth_encoding") == encoding
                    and len(encoded_bytes) <= 480 * 1024
                    and int(request.get("depth_uncompressed_bytes", -1))
                    == expected_bytes
                    and int(request.get("depth_compressed_bytes", -1))
                    == len(compressed)
                    and len(raw) == expected_bytes
                    and decoder.eof
                    and not decoder.unused_data
                    and not decoder.unconsumed_tail
                )
                nonempty_depth = bool(
                    payload_valid
                    and any(raw[index] or raw[index + 1] for index in range(0, len(raw), 2))
                )
            except (UnicodeEncodeError, binascii.Error, ValueError, zlib.error):
                payload_valid = False
                nonempty_depth = False
        if (
            request.get("operation") != "update"
            or bool(request.get("state_only", False))
            or height <= 0
            or width <= 0
            or sample_count > 640 * 480
            or not payload_valid
            or not nonempty_depth
        ):
            return
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "status": "PASS",
            "evidence": "real_sim_depth_frame_acknowledged_by_dgx_controller",
            "transport": self.transport,
            "controller_endpoint": self.endpoint,
            "episode_id": str(request.get("episode_id", "")),
            "reset_generation": int(request.get("reset_generation", -1)),
            "sequence_id": int(request.get("sequence_id", -1)),
            "depth_height": height,
            "depth_width": width,
            "depth_sample_count": sample_count,
            "depth_transport_encoding": encoding,
            "nonempty_depth": True,
            "controller_response_status": str(response.get("status", "")),
            "wall_unix": time.time(),
        }
        try:
            with target.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            # One record per lane run is sufficient.  A second client in the
            # same process must not replace the first-frame evidence.
            pass
        self._sensor_readiness_recorded = True


@dataclass
class LimitedTwist:
    linear_x: float
    angular_z: float
    linear_acceleration: float
    angular_acceleration: float


class JerkLimitedTwist:
    def __init__(
        self,
        *,
        max_linear: float = 0.25,
        max_angular: float = 0.8,
        max_linear_acceleration: float = 0.6,
        max_angular_acceleration: float = 1.8,
        max_linear_jerk: float = 2.5,
        max_angular_jerk: float = 7.0,
    ) -> None:
        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)
        self.max_linear_acceleration = float(max_linear_acceleration)
        self.max_angular_acceleration = float(max_angular_acceleration)
        self.max_linear_jerk = float(max_linear_jerk)
        self.max_angular_jerk = float(max_angular_jerk)
        if min(
            self.max_linear,
            self.max_angular,
            self.max_linear_acceleration,
            self.max_angular_acceleration,
            self.max_linear_jerk,
            self.max_angular_jerk,
        ) <= 0.0:
            raise ValueError("all motion limits must be positive")
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.linear_acceleration = 0.0
        self.angular_acceleration = 0.0

    @staticmethod
    def _clip(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def reset(self) -> LimitedTwist:
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.linear_acceleration = 0.0
        self.angular_acceleration = 0.0
        return self.value()

    def value(self) -> LimitedTwist:
        return LimitedTwist(
            self.linear_x,
            self.angular_z,
            self.linear_acceleration,
            self.angular_acceleration,
        )

    def step(
        self,
        desired_linear: float,
        desired_angular: float,
        dt: float,
        *,
        emergency_stop: bool = False,
    ) -> LimitedTwist:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("controller dt must be finite and positive")
        if not math.isfinite(desired_linear) or not math.isfinite(desired_angular):
            emergency_stop = True
        if emergency_stop:
            return self.reset()
        desired_linear = self._clip(float(desired_linear), self.max_linear)
        desired_angular = self._clip(float(desired_angular), self.max_angular)
        target_linear_acceleration = self._clip(
            (desired_linear - self.linear_x) / dt, self.max_linear_acceleration
        )
        target_angular_acceleration = self._clip(
            (desired_angular - self.angular_z) / dt, self.max_angular_acceleration
        )
        self.linear_acceleration += self._clip(
            target_linear_acceleration - self.linear_acceleration,
            self.max_linear_jerk * dt,
        )
        self.angular_acceleration += self._clip(
            target_angular_acceleration - self.angular_acceleration,
            self.max_angular_jerk * dt,
        )
        previous_linear = self.linear_x
        previous_angular = self.angular_z
        self.linear_x = self._clip(
            self.linear_x + self.linear_acceleration * dt, self.max_linear
        )
        self.angular_z = self._clip(
            self.angular_z + self.angular_acceleration * dt, self.max_angular
        )
        if (desired_linear - previous_linear) * (desired_linear - self.linear_x) <= 0.0:
            self.linear_x = desired_linear
        if (desired_angular - previous_angular) * (desired_angular - self.angular_z) <= 0.0:
            self.angular_z = desired_angular
        return self.value()
