"""Isaac-side typed bridge for a fixed reference-path Nav2 oracle gate."""

from __future__ import annotations

import json
import math
import os
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from internvla_ros2_msgs.msg import NavigationCommand
from internvla_ros2_msgs.srv import ResolveCommand
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


PROTOCOL_VERSION = 1
ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
SOURCE_SYSTEM2 = 1
SOURCE_SYSTEM1_NEW = 2
MAX_MESSAGE_BYTES = 1024 * 1024
MAX_EPISODE_ID_BYTES = 256


class OraclePeerClosed(EOFError):
    """The evaluator closed cleanly between complete protocol messages."""


def _validated_episode_id(value: Any) -> str:
    episode_id = str(value)
    if not episode_id:
        raise ValueError("oracle episode_id is empty")
    if len(episode_id.encode("utf-8")) > MAX_EPISODE_ID_BYTES:
        raise ValueError("oracle episode_id is too long")
    if any(ord(character) < 32 or ord(character) == 127 for character in episode_id):
        raise ValueError("oracle episode_id contains control characters")
    return episode_id


def _assign_time(message: Any, value_ns: int) -> None:
    message.sec = int(value_ns // 1_000_000_000)
    message.nanosec = int(value_ns % 1_000_000_000)


def _future_result(future: Any, timeout_sec: float) -> Any:
    event = threading.Event()
    future.add_done_callback(lambda _: event.set())
    if not event.wait(timeout_sec):
        raise TimeoutError("Nav2 oracle resolution timed out")
    exception = future.exception()
    if exception is not None:
        raise exception
    return future.result()


def _yaw(rotation_wxyz: list[float]) -> float:
    w, x, y, z = rotation_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class OracleBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("internvla_nav2_oracle_bridge")
        self.declare_parameter("goal_distance_m", 1.5)
        self.declare_parameter("success_stop_distance_m", 2.5)
        self.declare_parameter("resolution_timeout_sec", 10.0)
        self.declare_parameter("publish_observation_pose", True)
        self.goal_distance = float(self.get_parameter("goal_distance_m").value)
        self.stop_distance = float(self.get_parameter("success_stop_distance_m").value)
        self.resolution_timeout = float(self.get_parameter("resolution_timeout_sec").value)
        self.publish_observation_pose = bool(
            self.get_parameter("publish_observation_pose").value
        )
        self.navigation_publisher = self.create_publisher(
            NavigationCommand, "/internvla/navigation_command", 20
        )
        self.odom_publisher = self.create_publisher(Odometry, "/odom", 20)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = self.get_clock().now().to_msg()
        map_to_odom.header.frame_id = "map"
        map_to_odom.child_frame_id = "odom"
        map_to_odom.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(map_to_odom)
        self.resolve_client = self.create_client(ResolveCommand, "/internvla/nav2_resolve")
        self.lock = threading.RLock()
        self.initialized = False
        self.episode_id = ""
        self.generation = 0
        self.sequence = 0
        self.episode_bound = False
        self.identity_prefix = os.environ.get("INTERNNAV_T5_ID_PREFIX", "")
        if self.identity_prefix not in {"", "a::", "b::"}:
            raise RuntimeError("INTERNNAV_T5_ID_PREFIX must be empty, a::, or b::")

    def initialize(self) -> dict[str, Any]:
        with self.lock:
            if not self.resolve_client.wait_for_service(timeout_sec=15.0):
                raise TimeoutError("Nav2 resolver unavailable")
            self.initialized = True
            self.episode_id = "oracle-episode-0"
            self.generation = 0
            self.sequence = 0
            self.episode_bound = False
            return {"episode_id": self.episode_id, "reset_generation": self.generation}

    def reset(self) -> dict[str, Any]:
        with self.lock:
            if not self.initialized:
                raise RuntimeError("oracle bridge is not initialized")
            self.generation += 1
            self.episode_id = f"oracle-episode-{self.generation}"
            self.sequence = 0
            self.episode_bound = False
            return {"episode_id": self.episode_id, "reset_generation": self.generation}

    def step(
        self,
        global_gps: list[float],
        global_rotation: list[float],
        reference_path: list[list[float]],
        episode_id: str,
    ) -> dict[str, Any]:
        with self.lock:
            if not self.initialized:
                raise RuntimeError("oracle bridge is not initialized")
            requested_episode_id = _validated_episode_id(episode_id)
            if not self.episode_bound and self.sequence == 0:
                self.episode_id = requested_episode_id
                self.episode_bound = True
            elif not self.episode_bound or requested_episode_id != self.episode_id:
                raise RuntimeError("oracle dataset episode_id changed within a generation")
            if len(global_gps) != 3 or len(global_rotation) != 4:
                raise ValueError("invalid oracle pose")
            reference = np.asarray(reference_path, dtype=np.float64)
            if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) < 2:
                raise ValueError("reference_path must be finite [N,3]")
            if not np.isfinite(reference).all():
                raise ValueError("reference_path contains NaN/Inf")
            stamp = self.get_clock().now().to_msg()
            if self.publish_observation_pose:
                self._publish_pose(stamp, global_gps, global_rotation)
            map_reference = np.column_stack((reference[:, 0], -reference[:, 2]))
            current = np.asarray(global_gps[:2], dtype=np.float64)
            distance_to_goal = float(np.linalg.norm(current - map_reference[-1]))
            stop = distance_to_goal <= self.stop_distance
            local_path = NavPath()
            local_path.header.stamp = stamp
            local_path.header.frame_id = "base_link"
            model_action = ACTION_STOP
            if not stop:
                local_points = self._local_oracle_path(
                    current, _yaw(global_rotation), map_reference
                )
                model_action = self._first_action(local_points)
                for x, y in local_points:
                    pose = PoseStamped()
                    pose.header = local_path.header
                    pose.pose.position.x = float(x)
                    pose.pose.position.y = float(y)
                    pose.pose.orientation.w = 1.0
                    local_path.poses.append(pose)

            now_ns = self.get_clock().now().nanoseconds
            command = NavigationCommand()
            command.header.stamp = stamp
            command.header.frame_id = "base_link"
            command.protocol_version = PROTOCOL_VERSION
            command.episode_id = self.episode_id
            command.reset_generation = self.generation
            command.sequence_id = self.sequence
            command.request_id = (
                f"{self.identity_prefix}oracle:{self.generation}:{self.sequence}"
            )
            command.observation_digest = "oracle-reference-path"
            _assign_time(command.valid_until, now_ns + int(10e9))
            command.discrete_action = model_action
            command.stop = stop
            command.action_source = SOURCE_SYSTEM2 if stop else SOURCE_SYSTEM1_NEW
            command.trajectory_source = 0 if stop else 1
            command.trajectory_valid = not stop
            command.local_path = local_path
            command.global_gps = [float(value) for value in global_gps]
            command.global_rotation_wxyz = [float(value) for value in global_rotation]
            self.navigation_publisher.publish(command)
            request = ResolveCommand.Request()
            request.command = command
            response = _future_result(
                self.resolve_client.call_async(request), self.resolution_timeout
            )
            if int(response.status_code) != 0:
                raise RuntimeError(response.status_message)
            if (
                response.episode_id != self.episode_id
                or int(response.reset_generation) != self.generation
                or int(response.sequence_id) != self.sequence
                or response.request_id != command.request_id
            ):
                raise RuntimeError("stale oracle Nav2 resolution")
            self.sequence += 1
            return {
                "discrete_action": int(response.discrete_action),
                "stop": bool(response.stop),
                "distance_to_goal_m": distance_to_goal,
                "nav2_goal_sent": bool(response.nav2_goal_sent),
                "nav2_plan_valid": bool(response.nav2_plan_valid),
                "episode_id": self.episode_id,
                "reset_generation": self.generation,
                "reset_id": f"{self.identity_prefix}{self.generation}",
                "sequence_id": self.sequence - 1,
            }

    def _publish_pose(
        self, stamp: Any, gps: list[float], rotation: list[float]
    ) -> None:
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = float(gps[0])
        transform.transform.translation.y = float(gps[1])
        transform.transform.translation.z = float(gps[2])
        transform.transform.rotation.x = float(rotation[1])
        transform.transform.rotation.y = float(rotation[2])
        transform.transform.rotation.z = float(rotation[3])
        transform.transform.rotation.w = float(rotation[0])
        self.tf_broadcaster.sendTransform(transform)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(gps[0])
        odom.pose.pose.position.y = float(gps[1])
        odom.pose.pose.position.z = float(gps[2])
        odom.pose.pose.orientation = transform.transform.rotation
        self.odom_publisher.publish(odom)

    def _local_oracle_path(
        self, current: np.ndarray, base_yaw: float, reference: np.ndarray
    ) -> np.ndarray:
        # Track the reference polyline itself.  A straight chord to a lookahead
        # endpoint cuts corners and can drive the robot through doorway walls.
        # Project onto the closest segment first so lateral tracking error is
        # corrected without jumping to an unrelated reference vertex.
        best_index = 0
        best_projection = reference[0]
        best_distance = float("inf")
        for index, (first, second) in enumerate(zip(reference, reference[1:])):
            segment = second - first
            norm_squared = float(np.dot(segment, segment))
            if norm_squared <= 1e-12:
                projection = first
            else:
                ratio = float(np.dot(current - first, segment) / norm_squared)
                projection = first + min(1.0, max(0.0, ratio)) * segment
            distance = float(np.linalg.norm(current - projection))
            if distance < best_distance:
                best_index = index
                best_projection = projection
                best_distance = distance

        route = [current]
        if float(np.linalg.norm(best_projection - current)) > 1e-6:
            route.append(best_projection)
        route.extend(reference[best_index + 1 :])
        deduplicated = [np.asarray(route[0], dtype=np.float64)]
        for point in route[1:]:
            point_array = np.asarray(point, dtype=np.float64)
            if float(np.linalg.norm(point_array - deduplicated[-1])) > 1e-6:
                deduplicated.append(point_array)
        if len(deduplicated) == 1:
            deduplicated.append(np.asarray(reference[-1], dtype=np.float64))

        lengths = [
            float(np.linalg.norm(second - first))
            for first, second in zip(deduplicated, deduplicated[1:])
        ]
        total = min(self.goal_distance, sum(lengths))
        if total <= 1e-9:
            map_points = np.repeat(current[None, :], 33, axis=0)
        else:
            distances = np.linspace(0.0, total, 33)
            map_points_list: list[np.ndarray] = []
            segment_index = 0
            segment_start_distance = 0.0
            for distance in distances:
                while (
                    segment_index + 1 < len(lengths)
                    and distance > segment_start_distance + lengths[segment_index]
                ):
                    segment_start_distance += lengths[segment_index]
                    segment_index += 1
                length = lengths[segment_index]
                ratio = (
                    (distance - segment_start_distance) / length
                    if length > 1e-12
                    else 0.0
                )
                map_points_list.append(
                    deduplicated[segment_index]
                    + min(1.0, max(0.0, ratio))
                    * (deduplicated[segment_index + 1] - deduplicated[segment_index])
                )
            map_points = np.asarray(map_points_list, dtype=np.float64)
        delta = map_points - current[None, :]
        cosine, sine = math.cos(base_yaw), math.sin(base_yaw)
        local_x = cosine * delta[:, 0] + sine * delta[:, 1]
        local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
        return np.column_stack((local_x, local_y))

    @staticmethod
    def _first_action(local_points: np.ndarray) -> int:
        target = local_points[min(4, len(local_points) - 1)]
        angle = math.atan2(float(target[1]), float(target[0]))
        if angle > math.radians(15.0):
            return ACTION_LEFT
        if angle < -math.radians(15.0):
            return ACTION_RIGHT
        return ACTION_FORWARD


def _recv_exact(
    connection: socket.socket, count: int, *, clean_eof_ok: bool = False
) -> bytes:
    value = bytearray()
    while len(value) < count:
        block = connection.recv(count - len(value))
        if not block:
            if clean_eof_ok and not value:
                raise OraclePeerClosed("oracle evaluator completed")
            raise ConnectionError("oracle IPC disconnected")
        value.extend(block)
    return bytes(value)


def _recv_json(connection: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", _recv_exact(connection, 4, clean_eof_ok=True))[0]
    if size < 2 or size > MAX_MESSAGE_BYTES:
        raise ValueError("invalid oracle IPC message size")
    value = json.loads(_recv_exact(connection, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("oracle IPC request must be an object")
    return value


def _send_json(connection: socket.socket, value: dict[str, Any]) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = OracleBridgeNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    bind_host = os.environ.get("INTERNVLA_ORACLE_TCP_BIND_HOST", "")
    socket_path: Path | None = None
    if bind_host:
        bind_port = int(os.environ.get("INTERNVLA_ORACLE_TCP_PORT", "25140"))
        expected_peer = os.environ.get(
            "INTERNVLA_ORACLE_TCP_EXPECTED_PEER", "10.100.120.111"
        )
        if not 1024 <= bind_port <= 65535:
            raise RuntimeError("oracle TCP port is outside [1024,65535]")
        socket.inet_pton(socket.AF_INET, bind_host)
        socket.inet_pton(socket.AF_INET, expected_peer)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((bind_host, bind_port))
    else:
        socket_path = Path(
            os.environ.get("INTERNVLA_ORACLE_SOCKET", "/tmp/internvla_oracle.sock")
        )
        if socket_path.exists():
            socket_path.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
    listener.listen(1)
    try:
        connection, address = listener.accept()
        if bind_host and str(address[0]) != expected_peer:
            connection.close()
            raise RuntimeError(f"rejected oracle peer {address[0]}")
        with connection:
            connection.settimeout(600.0)
            while True:
                try:
                    request = _recv_json(connection)
                    operation = str(request.get("operation", ""))
                    if int(request.get("schema_version", 0)) != 1:
                        raise ValueError("unsupported oracle IPC schema")
                    if operation == "initialize":
                        result = node.initialize()
                    elif operation == "reset":
                        result = node.reset()
                    elif operation == "step":
                        result = node.step(
                            [float(v) for v in request["global_gps"]],
                            [float(v) for v in request["global_rotation"]],
                            request["reference_path"],
                            str(request["episode_id"]),
                        )
                    elif operation == "shutdown":
                        _send_json(connection, {"schema_version": 1, "status": "ok"})
                        break
                    else:
                        raise ValueError("unsupported oracle IPC operation")
                    _send_json(connection, {"schema_version": 1, "status": "ok", **result})
                except OraclePeerClosed:
                    # A fixed-dataset evaluator naturally closes the transport
                    # after its final complete episode.  EOF between frames is
                    # normal completion; EOF within a frame still fails above.
                    break
                except BaseException as exc:
                    _send_json(
                        connection,
                        {
                            "schema_version": 1,
                            "status": "error",
                            "message": repr(exc)[:1024],
                            "discrete_action": ACTION_STOP,
                        },
                    )
    finally:
        listener.close()
        if socket_path is not None and socket_path.exists():
            socket_path.unlink()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
