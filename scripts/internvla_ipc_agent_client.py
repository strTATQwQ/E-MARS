#!/usr/bin/env python3
"""Isaac-Python AgentClient facade for the local ROS-native client node.

Only a bounded versioned JSON envelope crosses the Unix socket. RGB and depth
are copied through named POSIX shared-memory segments and are unlinked by this
producer immediately after each response. No pickle is accepted or emitted.
"""

from __future__ import annotations

import gzip
import base64
import hashlib
import json
import os
import socket
import struct
import sys
import time
import zlib
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np


MAX_IPC_MESSAGE_BYTES = 4 * 1024 * 1024
SAFE_STOP_ACTION = [{"action": [0], "ideal_flag": True}]


def _fault_control_reader() -> Any | None:
    if os.environ.get("INTERNNAV_T5_FAULT_INJECTION_PROFILE", "off") == "off":
        return None
    repository_root = Path(__file__).resolve().parents[1]
    local_package = repository_root / "internvla_ros2"
    if local_package.is_dir():
        sys.path.insert(0, str(local_package))
    from internvla_ros2.fault_injection import (
        FaultControlReader,
        fault_profile_enabled,
    )

    return FaultControlReader("model_agent") if fault_profile_enabled() else None


def _lane_identity_prefix() -> str:
    prefix = os.environ.get("INTERNNAV_T5_ID_PREFIX", "")
    if prefix not in {"", "a::", "b::"}:
        raise RuntimeError("INTERNNAV_T5_ID_PREFIX must be empty, a::, or b::")
    return prefix


def _dataset_episode_ids(path_text: str) -> tuple[str, ...]:
    """Load the evaluator's frozen episode order when completion_sim provides it."""

    if not path_text:
        return ()
    path = Path(path_text).resolve()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    episodes = value.get("episodes") if isinstance(value, dict) else None
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError("model dataset has no episodes")
    identifiers: list[str] = []
    for item in episodes:
        episode_id = str(item.get("episode_id", "")) if isinstance(item, dict) else ""
        if not episode_id or len(episode_id.encode("utf-8")) > 256:
            raise RuntimeError("model dataset contains an invalid episode_id")
        if any(ord(character) < 32 or ord(character) == 127 for character in episode_id):
            raise RuntimeError("model dataset episode_id contains control characters")
        identifiers.append(episode_id)
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("model dataset contains duplicate episode_id values")
    return tuple(identifiers)


def _ordered_dataset_episode_ids(
    path_text: str, order_manifest_text: str
) -> tuple[str, ...]:
    identifiers = _dataset_episode_ids(path_text)
    if not order_manifest_text:
        return identifiers
    dataset_path = Path(path_text).resolve()
    manifest_path = Path(order_manifest_text)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("model episode order manifest must be a regular file")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "PASS":
        raise RuntimeError("model episode order manifest did not pass")
    expected_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    with gzip.open(dataset_path, "rt", encoding="utf-8") as stream:
        dataset_value = json.load(stream)
    episodes = dataset_value.get("episodes") if isinstance(dataset_value, dict) else None
    if not isinstance(episodes, list) or len(episodes) != len(identifiers):
        raise RuntimeError("model dataset changed while binding its order manifest")
    dataset_keys = []
    for item in episodes:
        trajectory_id = str(item.get("trajectory_id", "")).strip()
        episode_id = str(item.get("episode_id", "")).strip()
        if not trajectory_id or not episode_id:
            raise RuntimeError("model dataset has an invalid path-key identity")
        dataset_keys.append(f"{trajectory_id}_{episode_id}")
    ordered_ids = value.get("ordered_episode_ids")
    ordered_keys = value.get("ordered_episode_keys")
    raw_keys = value.get("raw_episode_keys")
    if (
        value.get("schema_version") != 1
        or value.get("dataset_sha256") != expected_sha
        or value.get("dataset_episode_count") != len(identifiers)
        or not isinstance(ordered_ids, list)
        or not isinstance(ordered_keys, list)
        or not isinstance(raw_keys, list)
        or len(ordered_ids) != len(identifiers)
        or len(ordered_keys) != len(identifiers)
        or len(raw_keys) != len(identifiers)
        or list(map(str, raw_keys)) != dataset_keys
        or len(set(map(str, ordered_keys))) != len(ordered_keys)
        or set(map(str, ordered_keys)) != set(map(str, raw_keys))
        or sorted(map(str, ordered_ids)) != sorted(identifiers)
    ):
        raise RuntimeError("model episode order manifest does not bind the dataset")
    normalized = tuple(str(value) for value in ordered_ids)
    if any(
        not value
        or len(value.encode("utf-8")) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        for value in normalized
    ):
        raise RuntimeError("model episode order manifest has an invalid episode_id")
    return normalized


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("local InternVLA IPC closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_json(connection: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", _recv_exact(connection, 4))[0]
    if size < 2 or size > MAX_IPC_MESSAGE_BYTES:
        raise RuntimeError("invalid local InternVLA IPC response length")
    value = json.loads(_recv_exact(connection, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("local InternVLA IPC response must be an object")
    return value


def _send_json(connection: socket.socket, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_IPC_MESSAGE_BYTES:
        raise RuntimeError("local InternVLA IPC request exceeds bound")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


class ROS2IPCAgentClient:
    """Drop-in replacement for upstream ``AgentClient`` on the Isaac side."""

    def __init__(self, config: Any):
        self.agent_name = str(config.model_name)
        self.socket_path = Path(
            os.environ.get("INTERNVLA_CLIENT_SOCKET", "/tmp/internvla_client.sock")
        )
        self.tcp_endpoint = os.environ.get("INTERNVLA_CLIENT_ENDPOINT", "")
        self.timeout_sec = float(os.environ.get("INTERNVLA_LOCAL_IPC_TIMEOUT_SEC", "45"))
        self.allow_episode_reset_reconnect = (
            os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
        )
        self.identity_prefix = _lane_identity_prefix()
        self.connection: socket.socket | None = None
        self.episode_ordinal = 0
        self.dataset_episode_ids = _ordered_dataset_episode_ids(
            os.environ.get("INTERNVLA_MODEL_DATASET_FILE", ""),
            os.environ.get("INTERNVLA_MODEL_EPISODE_ORDER_MANIFEST", ""),
        )
        self.last_result: dict[str, Any] | None = None
        self.last_error = ""
        self.last_step_safe_stop_kind: str | None = None
        self._t5_last_camera_sensor_sequence = 0
        self._t5_last_camera_sensor_stamp_ns = 0
        self._t5_camera_sensor_identity = bool(
            os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
            and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
            and os.environ.get("INTERNNAV_T5_LANE", "") in {"a", "b"}
        )
        self._fault_control = _fault_control_reader()
        self._fault_reset_event_id: str | None = None
        self._connect()
        response = self._exchange(
            {
                "schema_version": 1,
                "operation": "initialize",
                "episode_id": self._episode_id(),
            }
        )
        if self._requires_evaluator_continuation_reset(response):
            # The DGX TCP IPC server closes a connection after returning a
            # ClientFailure.  Reconnect exactly once before the explicit reset;
            # otherwise the reset is written to a peer-closed socket.
            self.close()
            self._connect()
            response = self._exchange(
                {
                    "schema_version": 1,
                    "operation": "reset",
                    "next_episode_id": self._episode_id(),
                }
            )
            self._require_ok(response, "continuation reset")
            print(
                "INTERNVLA_LOCAL_IPC_CONTINUATION_RESET "
                + json.dumps(
                    {
                        "schema_version": 1,
                        "episode_id": self._episode_id(),
                        "bounded_attempt_count": 1,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        else:
            self._require_ok(response, "initialize")
        self.handshake_response = dict(response)

    def _requires_evaluator_continuation_reset(
        self, response: dict[str, Any]
    ) -> bool:
        return bool(
            self.allow_episode_reset_reconnect
            and os.environ.get("INTERNVLA_T5_EVALUATOR_CONTINUATION_RESET", "") == "1"
            and response.get("status") == "error"
            and response.get("status_code") == 2
            and response.get("message")
            == "already initialized for a different episode; use reset"
        )

    def _episode_id(self) -> str:
        dataset_episode_ids = getattr(self, "dataset_episode_ids", ())
        if dataset_episode_ids:
            if self.episode_ordinal >= len(dataset_episode_ids):
                raise RuntimeError("evaluator advanced beyond frozen model dataset")
            episode_id = dataset_episode_ids[self.episode_ordinal]
        else:
            episode_id = f"isaac-evaluator-episode-{self.episode_ordinal}"
        # Legacy T4 fixtures construct the facade with ``__new__`` and do not
        # run the T5-only identity initialization.  Empty keeps the frozen T4
        # wire identity unchanged; real T5 construction always sets a::/b::.
        return f"{getattr(self, 'identity_prefix', '')}{episode_id}"

    def _connect(self) -> None:
        if self.tcp_endpoint:
            if not self.tcp_endpoint.startswith("tcp://"):
                raise RuntimeError("INTERNVLA_CLIENT_ENDPOINT must use tcp://")
            host_port = self.tcp_endpoint[6:]
            host, separator, port_text = host_port.rpartition(":")
            if not separator or not host or not port_text.isdigit():
                raise RuntimeError("invalid InternVLA client TCP endpoint")
            port = int(port_text)
            if not 1024 <= port <= 65535:
                raise RuntimeError("InternVLA client TCP port is outside bounds")
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.settimeout(self.timeout_sec)
            connection.connect((host, port))
        else:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(self.timeout_sec)
            connection.connect(str(self.socket_path))
        connection.settimeout(self.timeout_sec)
        self.connection = connection

    def _exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.connection is None:
            raise ConnectionError("local InternVLA IPC is not connected")
        _send_json(self.connection, request)
        return _recv_json(self.connection)

    @staticmethod
    def _require_ok(response: dict[str, Any], operation: str) -> None:
        if response.get("status") != "ok" or int(response.get("status_code", 0)) != 0:
            raise RuntimeError(
                f"local InternVLA {operation} failed: "
                f"{response.get('status_code')} {response.get('message', response.get('status_message'))}"
            )

    def _wait_for_fault_restart_maintenance(self, snapshot: Any) -> None:
        fault_control = self._fault_control
        active = [
            (kind, snapshot.event_for(kind))
            for kind in ("model_service_restart", "dgx_ros_node_restart")
            if snapshot.event_for(kind) is not None
        ]
        if not active:
            return
        if len(active) != 1:
            raise RuntimeError("multiple restart maintenance events are active")
        kind, event_id = active[0]
        assert event_id is not None
        self.last_result = None
        self.last_error = f"waiting for injected {kind} maintenance"
        # The DGX evaluator server has a bounded idle timeout.  Deliberately
        # close the quiesced channel before a potentially long model reload,
        # then create one fresh connection only after the director clears the
        # bound event.
        self.close()
        fault_control.record(
            event_id, kind, "maintenance_wait_started", snapshot.observed_sim_ns
        )
        timeout_sec = float(
            os.environ.get("INTERNNAV_T5_FAULT_WALL_LIVENESS_TIMEOUT_SEC", "7200")
        )
        if not 300.0 <= timeout_sec <= 8400.0:
            raise RuntimeError("fault maintenance wall liveness timeout is invalid")
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            time.sleep(0.1)
            current = fault_control.read()
            current_event = current.event_for(kind)
            if current_event == event_id:
                continue
            if current_event is not None:
                raise RuntimeError("restart maintenance event identity changed")
            self._connect()
            health = self._exchange(
                {"schema_version": 1, "operation": "health"}
            )
            self._require_ok(health, "post-restart health")
            if (
                health.get("initialized") is not True
                or health.get("episode_id") != self._episode_id()
            ):
                raise RuntimeError("post-restart model session identity mismatch")
            fault_control.record(
                event_id,
                kind,
                "maintenance_wait_finished",
                current.observed_sim_ns,
            )
            self.last_error = ""
            return
        raise TimeoutError("restart maintenance exceeded wall liveness timeout")

    @staticmethod
    def _expected_model_timeout_event(exc: BaseException, snapshot: Any) -> str | None:
        event_id = snapshot.event_for("model_request_timeout")
        if (
            event_id is not None
            and type(exc) is RuntimeError
            and str(exc)
            == "local InternVLA step failed: 4 injected T5 completion_sim model request timeout"
        ):
            return str(event_id)
        return None

    def step(self, obs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rgb_memory: SharedMemory | None = None
        depth_memory: SharedMemory | None = None
        self.last_step_safe_stop_kind = None
        try:
            fault_control = getattr(self, "_fault_control", None)
            if fault_control is not None:
                snapshot = fault_control.read()
                self._wait_for_fault_restart_maintenance(snapshot)
                reset_event = fault_control.consume_once("episode_reset")
                if reset_event is not None:
                    self._fault_reset_event_id = reset_event
                    self.last_result = None
                    self.last_error = "injected episode reset safe-stop"
                    self.last_step_safe_stop_kind = "expected_episode_reset"
                    fault_control.record(
                        reset_event,
                        "episode_reset",
                        "safe_stop_requested",
                        fault_control.read().observed_sim_ns,
                    )
                    return SAFE_STOP_ACTION
                snapshot = fault_control.read()
                network_event = snapshot.event_for(
                    "network_short_outage_and_recovery"
                )
                if network_event is not None:
                    # The injected outage affects only the lane data plane.
                    # Keep this control connection available so reset and
                    # recovery cannot be masked by a reconnect side effect.
                    self.last_result = None
                    self.last_error = "injected lane network data-plane outage"
                    self.last_step_safe_stop_kind = "expected_network_outage"
                    return SAFE_STOP_ACTION
            if len(obs) != 1 or not isinstance(obs[0], dict):
                raise ValueError("InternVLA IPC expects one observation dictionary")
            item = obs[0]
            rgb = np.ascontiguousarray(item["rgb"], dtype=np.uint8)
            depth = np.ascontiguousarray(item["depth"], dtype=np.float32)
            if rgb.shape != (480, 640, 3) or depth.shape != (480, 640, 1):
                raise ValueError("observation violates frozen RGB-D shape contract")
            if self.tcp_endpoint:
                rgb_descriptor = {
                    "encoding": "zlib+base64",
                    "shape": list(rgb.shape),
                    "dtype": "uint8",
                    "data": base64.b64encode(zlib.compress(rgb.tobytes(), 1)).decode("ascii"),
                }
                depth_descriptor = {
                    "encoding": "zlib+base64",
                    "shape": list(depth.shape),
                    "dtype": "float32",
                    "data": base64.b64encode(zlib.compress(depth.tobytes(), 1)).decode("ascii"),
                }
                array_transport = "inline_zlib"
            else:
                rgb_memory = SharedMemory(create=True, size=rgb.nbytes)
                depth_memory = SharedMemory(create=True, size=depth.nbytes)
                np.ndarray(rgb.shape, dtype=rgb.dtype, buffer=rgb_memory.buf)[:] = rgb
                np.ndarray(depth.shape, dtype=depth.dtype, buffer=depth_memory.buf)[:] = depth
                rgb_descriptor = {
                    "name": rgb_memory.name,
                    "shape": list(rgb.shape),
                    "dtype": "uint8",
                }
                depth_descriptor = {
                    "name": depth_memory.name,
                    "shape": list(depth.shape),
                    "dtype": "float32",
                }
                array_transport = "shared_memory"
            request = {
                    "schema_version": 1,
                    "operation": "step",
                    "array_transport": array_transport,
                    "rgb": rgb_descriptor,
                    "depth": depth_descriptor,
                    "instruction": str(item["instruction"]),
                    "instruction_tokens": np.asarray(
                        item.get("instruction_tokens", []), dtype=np.int64
                    ).tolist(),
                    "global_gps": np.asarray(
                        item.get("globalgps", [0.0, 0.0, 0.0]), dtype=np.float64
                    ).tolist(),
                    "global_rotation": np.asarray(
                        item.get("globalrotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64
                    ).tolist(),
                }
            if getattr(self, "_t5_camera_sensor_identity", False):
                metadata = item.get("camera_sensor_metadata")
                if not isinstance(metadata, dict):
                    raise ValueError("T5 camera source metadata is missing")
                if metadata.get("schema_version") != 1:
                    raise ValueError("T5 camera source metadata schema is invalid")
                if metadata.get("source") != "x86_isaac_pano_camera_0":
                    raise ValueError("T5 camera source identity is invalid")
                source_sequence = metadata.get("sequence")
                source_stamp_ns = metadata.get("sim_stamp_ns")
                if (
                    isinstance(source_sequence, bool)
                    or not isinstance(source_sequence, int)
                    or source_sequence <= self._t5_last_camera_sensor_sequence
                ):
                    raise ValueError("T5 camera source sequence did not strictly advance")
                if (
                    isinstance(source_stamp_ns, bool)
                    or not isinstance(source_stamp_ns, int)
                    or source_stamp_ns <= self._t5_last_camera_sensor_stamp_ns
                ):
                    raise ValueError("T5 camera source sim stamp did not strictly advance")
                self._t5_last_camera_sensor_sequence = source_sequence
                self._t5_last_camera_sensor_stamp_ns = source_stamp_ns
                request["camera_sensor_metadata"] = dict(metadata)
            response = self._exchange(request)
            self._require_ok(response, "step")
            discrete_action = int(response["discrete_action"])
            result = [
                {
                    "action": [discrete_action],
                    "ideal_flag": True,
                }
            ]
            self.last_result = response
            self.last_error = ""
            print(
                "INTERNVLA_MODEL_ACTION_OK "
                + json.dumps(
                    {
                        "schema_version": 1,
                        "episode_ordinal": self.episode_ordinal,
                        "discrete_action": discrete_action,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            return result
        except BaseException as exc:
            # Returning official STOP ends the episode without allowing a
            # delayed or disconnected model response to move the robot.
            self.last_error = repr(exc)[:1024]
            expected_timeout_event = None
            fault_control = getattr(self, "_fault_control", None)
            if fault_control is not None:
                try:
                    snapshot = fault_control.read()
                    expected_timeout_event = self._expected_model_timeout_event(
                        exc, snapshot
                    )
                except BaseException:
                    expected_timeout_event = None
            marker = "INTERNVLA_LOCAL_IPC_STEP_ERROR"
            if expected_timeout_event is not None:
                marker = "INTERNVLA_EXPECTED_FAULT_SAFE_STOP"
                self.last_step_safe_stop_kind = "expected_model_timeout"
                fault_control.record(
                    expected_timeout_event,
                    "model_request_timeout",
                    "expected_timeout_safe_stop",
                    snapshot.observed_sim_ns,
                )
            else:
                self.last_step_safe_stop_kind = "unexpected_ipc_error"
            print(
                marker + " "
                + json.dumps(
                    {
                        "schema_version": 1,
                        "episode_ordinal": self.episode_ordinal,
                        "error": self.last_error,
                        "fault_event_id": expected_timeout_event,
                        "safe_stop": True,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            self.last_result = None
            self.close()
            return SAFE_STOP_ACTION
        finally:
            for memory in (rgb_memory, depth_memory):
                if memory is not None:
                    memory.close()
                    try:
                        memory.unlink()
                    except FileNotFoundError:
                        pass

    def reset(self, reset_index: Any = None) -> dict[str, Any]:
        del reset_index
        self.episode_ordinal += 1
        if self.connection is None:
            if not self.allow_episode_reset_reconnect:
                raise ConnectionError("local InternVLA IPC is not connected")
            self._connect()
            print(
                "INTERNVLA_LOCAL_IPC_RESET_RECONNECTED "
                + json.dumps(
                    {
                        "schema_version": 1,
                        "episode_ordinal": self.episode_ordinal,
                        "bounded_attempt_count": 1,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        try:
            response = self._exchange(
                {
                    "schema_version": 1,
                    "operation": "reset",
                    "next_episode_id": self._episode_id(),
                }
            )
            self._require_ok(response, "reset")
        except BaseException:
            self.close()
            raise
        self.last_result = None
        self.last_error = ""
        fault_event = getattr(self, "_fault_reset_event_id", None)
        fault_control = getattr(self, "_fault_control", None)
        if fault_event is not None and fault_control is not None:
            fault_control.record(
                fault_event,
                "episode_reset",
                "reset_completed",
                fault_control.read().observed_sim_ns,
            )
            self._fault_reset_event_id = None
        return response

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            finally:
                self.connection = None

    def __del__(self) -> None:
        self.close()
