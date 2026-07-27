"""ROS 2 safety bridge between Nav2 and the Isaac Go2 physics controller."""

from __future__ import annotations

import json
import hashlib
import ipaddress
import math
import os
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import TransformStamped, Twist, TwistStamped
from internvla_ros2_msgs.msg import NavigationCommand
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, Float64MultiArray
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


MAX_MESSAGE_BYTES = 512 * 1024
CAMERA_TRANSLATION = (0.2, 0.0, 0.2)
# Rotation from the audited USD camera local frame (+X right, +Y up, -Z view)
# into base_link (+X forward, +Y left, +Z up).
CAMERA_ROTATION = (
    (0.0, 0.5, -0.8660254037844386),
    (-1.0, 0.0, 0.0),
    (0.0, 0.8660254037844386, 0.5),
)


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    output = bytearray()
    while len(output) < count:
        block = connection.recv(count - len(output))
        if not block:
            raise ConnectionError("Isaac controller disconnected")
        output.extend(block)
    return bytes(output)


def _recv_json(connection: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", _recv_exact(connection, 4))[0]
    if size < 2 or size > MAX_MESSAGE_BYTES:
        raise ValueError("invalid controller request size")
    value = json.loads(_recv_exact(connection, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("controller request must be an object")
    return value


def _send_json(connection: socket.socket, value: dict[str, Any]) -> None:
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("controller response exceeds bound")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def _yaw(rotation: list[float]) -> float:
    w, x, y, z = rotation
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _semantic_now(node: Any) -> float:
    """Use ROS simulation time only for the exact T5 completion lane."""

    if not getattr(node, "_t5_sim_time_semantics", False):
        return time.monotonic()
    now_ns = int(node.get_clock().now().nanoseconds)
    if now_ns <= 0:
        return 0.0
    clock_lock = getattr(node, "_semantic_clock_lock", None)
    if clock_lock is None:
        last_ns = int(getattr(node, "_last_semantic_clock_ns", 0))
        if now_ns < last_ns:
            return 0.0
        node._last_semantic_clock_ns = now_ns
    else:
        with clock_lock:
            last_ns = int(getattr(node, "_last_semantic_clock_ns", 0))
            if now_ns < last_ns:
                return 0.0
            node._last_semantic_clock_ns = now_ns
    return now_ns / 1_000_000_000


def _semantic_age(node: Any, stamp: float) -> float | None:
    now = _semantic_now(node)
    if getattr(node, "_t5_sim_time_semantics", False) and (
        now <= 0.0 or stamp <= 0.0 or now < stamp
    ):
        return None
    return now - stamp


def _command_is_fresh(
    *,
    safe_cmd_serial: int,
    barrier_cmd_serial: int,
    safe_cmd_stamp: float,
    barrier_stamp: float,
    command_age: float | None,
    timeout_sec: float,
) -> bool:
    return bool(
        safe_cmd_serial > barrier_cmd_serial
        and safe_cmd_stamp >= barrier_stamp
        and command_age is not None
        and command_age <= timeout_sec
    )


def _runtime_state_hazard(
    *,
    t5_completion_sim: bool,
    nan_detected: bool,
    fallen: bool,
    physical_collision: bool,
) -> bool:
    """Keep collisions metric-only only in the isolated navigation simulator."""

    return bool(
        nan_detected
        or fallen
        or (physical_collision and not t5_completion_sim)
    )


def _state_update_is_current(
    *,
    t5_sim_time_semantics: bool,
    update_identity: tuple[str, int, int],
    active_identity: tuple[str, int, int],
    update_epoch: int,
    current_epoch: int,
    state_only: bool = False,
) -> bool:
    bootstrap_state = bool(
        state_only
        and update_identity[0] == f"bootstrap-episode-{update_identity[1]}"
        and update_identity[1] == active_identity[1]
        and update_identity[2] == 0
    )
    return bool(
        not t5_sim_time_semantics
        or (
            (update_identity[:2] == active_identity[:2] or bootstrap_state)
            and update_epoch == current_epoch
        )
    )


def _active_identity_query_response(
    active_identity: tuple[str, int, int], replay_epoch: int
) -> dict[str, object]:
    episode_id, reset_generation, sequence_id = active_identity
    if not episode_id or reset_generation < 0 or sequence_id < -1:
        return {
            "schema_version": 1,
            "status": "error",
            "operation": "query_active_identity",
            "message": "active navigation identity is unavailable",
            "linear_x": 0.0,
            "angular_z": 0.0,
            "emergency_stop": True,
        }
    return {
        "schema_version": 1,
        "status": "ok",
        "operation": "query_active_identity",
        "active_episode_id": episode_id,
        "active_reset_generation": reset_generation,
        "active_sequence_id": sequence_id,
        "state_replay_epoch": replay_epoch,
        "linear_x": 0.0,
        "angular_z": 0.0,
        "emergency_stop": True,
    }


def _load_onboard_restart_identity() -> tuple[str, int, int] | None:
    session_value = os.environ.get(
        "INTERNVLA_T5_ONBOARD_RESTART_SESSION_FILE", ""
    )
    if not session_value:
        return None
    from internvla_ros2.fault_injection import (
        fault_profile_enabled,
        load_fault_restart_session,
    )

    if not fault_profile_enabled():
        raise RuntimeError(
            "controller session restore requires the exact T5 fault profile"
        )
    lane = os.environ.get("INTERNNAV_T5_LANE", "")
    session = load_fault_restart_session(
        Path(session_value),
        expected_lane=lane,
        expected_action="dgx_ros_node_restart",
    )
    return (
        str(session["episode_id"]),
        int(session["reset_generation"]),
        int(session["last_sequence_id"]),
    )


class Go2ControllerBridge(Node):
    def __init__(self) -> None:
        super().__init__("internvla_go2_controller_bridge")
        self.declare_parameter("result_dir", "")
        self.declare_parameter("socket_path", "/tmp/internvla_go2_controller.sock")
        self.declare_parameter("ipc_transport", "unix")
        self.declare_parameter("tcp_bind_host", "127.0.0.1")
        self.declare_parameter("tcp_port", 0)
        self.declare_parameter("tcp_expected_peer_ip", "127.0.0.1")
        self.declare_parameter("command_timeout_sec", 0.30)
        self.declare_parameter("expected_control_hz", 40.0)
        self.declare_parameter("state_republish_timeout_sec", 5.0)
        self.declare_parameter("static_map_manifest", "")
        self.result_dir = Path(str(self.get_parameter("result_dir").value)).expanduser()
        if not str(self.result_dir):
            raise RuntimeError("result_dir is required")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.socket_path = Path(str(self.get_parameter("socket_path").value))
        self.ipc_transport = str(self.get_parameter("ipc_transport").value)
        self.tcp_bind_host = str(self.get_parameter("tcp_bind_host").value)
        self.tcp_port = int(self.get_parameter("tcp_port").value)
        self.tcp_expected_peer_ip = str(
            self.get_parameter("tcp_expected_peer_ip").value
        )
        if self.ipc_transport not in {"unix", "tcp"}:
            raise RuntimeError("ipc_transport must be unix or tcp")
        if self.ipc_transport == "tcp":
            try:
                bind_address = ipaddress.ip_address(self.tcp_bind_host)
                peer_address = ipaddress.ip_address(self.tcp_expected_peer_ip)
            except ValueError as exc:
                raise RuntimeError("controller TCP addresses must be IP literals") from exc
            if bind_address.version != 4 or peer_address.version != 4:
                raise RuntimeError("controller TCP transport currently requires IPv4")
            if bind_address.is_multicast or peer_address.is_multicast:
                raise RuntimeError("controller TCP transport rejects multicast addresses")
            if bind_address.is_unspecified:
                raise RuntimeError("controller TCP bind address cannot be unspecified")
            if peer_address.is_unspecified:
                raise RuntimeError("controller TCP expected peer cannot be unspecified")
            if not 1024 <= self.tcp_port <= 65535:
                raise RuntimeError("controller TCP port is outside the allowed range")
        self.command_timeout = float(self.get_parameter("command_timeout_sec").value)
        self.expected_control_hz = float(self.get_parameter("expected_control_hz").value)
        self.state_republish_timeout = float(
            self.get_parameter("state_republish_timeout_sec").value
        )
        if (
            self.command_timeout <= 0.0
            or not 20.0 <= self.expected_control_hz <= 50.0
            or self.state_republish_timeout <= self.command_timeout
        ):
            raise RuntimeError("invalid controller timing parameters")
        self._t5_sim_time_semantics = bool(
            os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
            and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
            and os.environ.get("INTERNNAV_T5_LANE", "") in {"a", "b"}
        )
        if self._t5_sim_time_semantics and (
            not self.has_parameter("use_sim_time")
            or not bool(self.get_parameter("use_sim_time").value)
        ):
            raise RuntimeError("T5 completion_sim requires use_sim_time=true")
        self._semantic_clock_lock = threading.Lock()
        self._last_semantic_clock_ns = 0
        self.static_map_manifest_path = Path(
            str(self.get_parameter("static_map_manifest").value)
        ).expanduser()
        if not self.static_map_manifest_path.is_file():
            raise FileNotFoundError(
                f"static_map_manifest is required: {self.static_map_manifest_path}"
            )
        self.static_map_manifest_sha256 = hashlib.sha256(
            self.static_map_manifest_path.read_bytes()
        ).hexdigest()
        self.static_map_manifest = json.loads(
            self.static_map_manifest_path.read_text(encoding="utf-8")
        )
        if int(self.static_map_manifest.get("schema_version", 0)) != 1:
            raise ValueError("unsupported static-map manifest")
        self.static_map_episode_entries = list(
            self.static_map_manifest.get("generations", [])
        )
        self.static_map_entries = dict(self.static_map_manifest.get("maps", {}))
        if not self.static_map_episode_entries or not self.static_map_entries:
            raise ValueError("static-map manifest has no generations or maps")

        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.started_unix = time.time()
        self.records_path = self.result_dir / "controller_records.jsonl"
        self.costmap_records_path = self.result_dir / "costmap_records.jsonl"
        if self.records_path.exists():
            raise FileExistsError(self.records_path)
        if self.costmap_records_path.exists():
            raise FileExistsError(self.costmap_records_path)
        restart_identity = _load_onboard_restart_identity()
        self.active_identity = (
            ("", -1, -1) if restart_identity is None else restart_identity
        )
        self.navigation_barrier_monotonic = 0.0
        self.navigation_barrier_cmd_serial = 0
        self.motion_enabled = False
        self.stop_latched = True
        self.last_safe_cmd = (0.0, 0.0)
        self.last_safe_cmd_monotonic = 0.0
        self.safe_cmd_serial = 0
        self.last_raw_cmd = (0.0, 0.0)
        self.last_raw_cmd_monotonic = 0.0
        self.active_path: list[tuple[float, float]] = []
        self.last_update_monotonic = 0.0
        self.last_update_identity: tuple[str, int, int] | None = None
        self.update_intervals: list[float] = []
        self.update_count = 0
        self.state_only_count = 0
        self.latest_state: tuple[list[float], list[float], list[float]] | None = None
        self.latest_state_monotonic = 0.0
        self.latest_pointcloud: list[tuple[float, float, float]] | None = None
        self.latest_pointcloud_monotonic = 0.0
        self.timeout_count = 0
        self.stale_count = 0
        self.reset_barrier_count = 0
        self.estop_count = 0
        self.fall_count = 0
        self.nan_count = 0
        self.depth_frame_count = 0
        self.depth_nonempty_count = 0
        self.obstacle_expected_target_count = 0
        self.obstacle_detected_target_count = 0
        self.obstacle_expected_frame_count = 0
        self.obstacle_detected_frame_count = 0
        self.latest_detected_targets_map: list[tuple[float, float, float, float, float]] = []
        self.latest_detected_target_monotonic = 0.0
        self.latest_detected_target_token = -1
        self.latest_detected_target_generation = -1
        self.latest_detected_target_episode = ""
        self._state_replay_epoch = 0
        self.last_costmap_scored_token = -1
        self.costmap_expected_frame_count = 0
        self.costmap_detected_frame_count = 0
        self.costmap_expected_target_count = 0
        self.costmap_detected_target_count = 0
        self.costmap_by_generation: dict[int, dict[str, int]] = {}
        self.static_map_generation = -1
        self.static_map_publish_count = 0
        self.static_map_selections: list[dict[str, Any]] = []
        self.static_map_cache: dict[str, list[int]] = {}
        self.collision_monitor_stop_count = 0
        self.collision_monitor_recovery_count = 0
        self.physical_collision_count = 0
        self.ipc_accepted_connection_count = 0
        self.ipc_rejected_peer_count = 0
        self._physical_collision_active = False
        self._monitor_was_stopping = False

        self.odom_publisher = self.create_publisher(Odometry, "/odom", 50)
        self.velocity_publisher = self.create_publisher(
            TwistStamped, "/go2/actual_velocity", 50
        )
        self.error_publisher = self.create_publisher(
            Float64MultiArray, "/go2/path_tracking_error", 50
        )
        self.pointcloud_publisher = self.create_publisher(
            PointCloud2, "/go2/depth/points", 10
        )
        self.static_map_publisher = self.create_publisher(
            OccupancyGrid,
            "/map",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = self.get_clock().now().to_msg()
        map_to_odom.header.frame_id = "map"
        map_to_odom.child_frame_id = "odom"
        map_to_odom.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(map_to_odom)

        # Relative names preserve T4's root topics while joining the complete
        # namespaced T5 Nav2 velocity chain.
        self.create_subscription(Twist, "cmd_vel_nav", self._on_raw_cmd, 50)
        self.create_subscription(Twist, "cmd_vel_safe", self._on_safe_cmd, 50)
        self.create_subscription(Bool, "/internvla/stop", self._on_stop, 20)
        self.create_subscription(
            Bool, "/internvla/nav2_motion_enabled", self._on_motion_enabled, 20
        )
        self.create_subscription(
            NavigationCommand,
            "/internvla/navigation_command",
            self._on_navigation,
            20,
        )
        self.create_subscription(
            NavPath, "/internvla/nav2_active_path", self._on_active_path, 20
        )
        self.create_subscription(
            OccupancyGrid, "local_costmap/costmap", self._on_local_costmap, 10
        )
        self.create_timer(1.0 / self.expected_control_hz, self._republish_state)

        self.server_thread = threading.Thread(target=self._serve, daemon=True)
        self.server_thread.start()
        self._write_summary("READY")

    def _on_raw_cmd(self, message: Twist) -> None:
        with self.lock:
            self.last_raw_cmd = (float(message.linear.x), float(message.angular.z))
            self.last_raw_cmd_monotonic = _semantic_now(self)

    def _on_safe_cmd(self, message: Twist) -> None:
        now = _semantic_now(self)
        safe = (float(message.linear.x), float(message.angular.z))
        with self.lock:
            self.last_safe_cmd = safe
            self.last_safe_cmd_monotonic = now
            self.safe_cmd_serial += 1
            raw_moving = abs(self.last_raw_cmd[0]) > 1e-4 or abs(self.last_raw_cmd[1]) > 1e-4
            safe_stopped = abs(safe[0]) <= 1e-4 and abs(safe[1]) <= 1e-4
            raw_age = _semantic_age(self, self.last_raw_cmd_monotonic)
            monitor_stopping = (
                raw_moving
                and safe_stopped
                and raw_age is not None
                and raw_age <= 0.2
            )
            if monitor_stopping and not self._monitor_was_stopping:
                self.collision_monitor_stop_count += 1
            if not monitor_stopping and self._monitor_was_stopping:
                self.collision_monitor_recovery_count += 1
            self._monitor_was_stopping = monitor_stopping

    def _publish_static_map(
        self,
        generation: int,
        pose: list[float],
        *,
        identity: tuple[str, int, int],
        replay_epoch: int,
    ) -> None:
        with self.lock:
            if generation == self.static_map_generation:
                return
        episode_entry = min(
            self.static_map_episode_entries,
            key=lambda item: math.hypot(
                float(item["start_map_xy"][0]) - pose[0],
                float(item["start_map_xy"][1]) - pose[1],
            ),
        )
        start_xy_error = math.hypot(
            float(episode_entry["start_map_xy"][0]) - pose[0],
            float(episode_entry["start_map_xy"][1]) - pose[1],
        )
        if start_xy_error > 0.05:
            raise ValueError(
                f"no static map start matches reset generation {generation}: "
                f"nearest error {start_xy_error:.3f}m"
            )
        map_key = str(episode_entry["map_key"])
        entry = self.static_map_entries.get(map_key)
        if not isinstance(entry, dict):
            raise KeyError(f"static map entry missing: {map_key}")
        map_path = self.static_map_manifest_path.parent / str(entry["file"])
        with self.lock:
            data = self.static_map_cache.get(map_key)
        if data is None:
            raw = map_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != str(entry["sha256"]):
                raise ValueError(f"static map hash mismatch: {map_key}")
            if len(raw) != int(entry["width"]) * int(entry["height"]):
                raise ValueError(f"static map size mismatch: {map_key}")
            if any(value not in (0, 100) for value in raw):
                raise ValueError(f"static map has invalid occupancy values: {map_key}")
            data = list(raw)
            with self.lock:
                self.static_map_cache[map_key] = data
        stamp = self.get_clock().now().to_msg()
        message = OccupancyGrid()
        message.header.stamp = stamp
        message.header.frame_id = "map"
        message.info.map_load_time = stamp
        message.info.resolution = float(entry["resolution_m"])
        message.info.width = int(entry["width"])
        message.info.height = int(entry["height"])
        message.info.origin.position.x = float(entry["origin_xy"][0])
        message.info.origin.position.y = float(entry["origin_xy"][1])
        message.info.origin.orientation.w = 1.0
        message.data = data
        with self.lock:
            if not _state_update_is_current(
                t5_sim_time_semantics=self._t5_sim_time_semantics,
                update_identity=identity,
                active_identity=self.active_identity,
                update_epoch=replay_epoch,
                current_epoch=self._state_replay_epoch,
            ):
                return
            self.static_map_publisher.publish(message)
            self.static_map_generation = generation
            self.static_map_publish_count += 1
            self.static_map_selections.append(
                {
                    "generation": generation,
                    "map_key": map_key,
                    "trajectory_id": str(episode_entry.get("trajectory_id", "")),
                    "episode_id": str(episode_entry.get("episode_id", "")),
                    "start_xy_error_m": start_xy_error,
                    "pose_xyz": [float(value) for value in pose[:3]],
                    "reference_height_m": float(
                        episode_entry.get("reference_height_m", 0.0)
                    ),
                }
            )

    def _on_local_costmap(self, message: OccupancyGrid) -> None:
        with self.lock:
            targets = list(self.latest_detected_targets_map)
            token = self.latest_detected_target_token
            generation = self.latest_detected_target_generation
            episode_id = self.latest_detected_target_episode
            age = _semantic_age(self, self.latest_detected_target_monotonic)
            if (
                not targets
                or token == self.last_costmap_scored_token
                or age is None
                or age < 0.05
                or age > 1.25
            ):
                return
        width = int(message.info.width)
        height = int(message.info.height)
        resolution = float(message.info.resolution)
        if width <= 0 or height <= 0 or resolution <= 0.0:
            return
        if len(message.data) != width * height:
            return
        origin = message.info.origin
        origin_yaw = _yaw(
            [
                float(origin.orientation.w),
                float(origin.orientation.x),
                float(origin.orientation.y),
                float(origin.orientation.z),
            ]
        )
        origin_cosine = math.cos(origin_yaw)
        origin_sine = math.sin(origin_yaw)
        detected_targets = 0
        for tx, ty, hx, hy, target_yaw in targets:
            delta_x = tx - float(origin.position.x)
            delta_y = ty - float(origin.position.y)
            grid_x = (origin_cosine * delta_x + origin_sine * delta_y) / resolution
            grid_y = (-origin_sine * delta_x + origin_cosine * delta_y) / resolution
            radius_cells = int(math.ceil((math.hypot(hx, hy) + 0.20) / resolution))
            target_cosine = math.cos(target_yaw)
            target_sine = math.sin(target_yaw)
            marked = False
            for row in range(
                max(0, int(math.floor(grid_y)) - radius_cells),
                min(height, int(math.floor(grid_y)) + radius_cells + 1),
            ):
                if marked:
                    break
                local_grid_y = (row + 0.5) * resolution
                for column in range(
                    max(0, int(math.floor(grid_x)) - radius_cells),
                    min(width, int(math.floor(grid_x)) + radius_cells + 1),
                ):
                    if int(message.data[row * width + column]) < 90:
                        continue
                    local_grid_x = (column + 0.5) * resolution
                    world_x = (
                        float(origin.position.x)
                        + origin_cosine * local_grid_x
                        - origin_sine * local_grid_y
                    )
                    world_y = (
                        float(origin.position.y)
                        + origin_sine * local_grid_x
                        + origin_cosine * local_grid_y
                    )
                    obstacle_x = target_cosine * (world_x - tx) + target_sine * (
                        world_y - ty
                    )
                    obstacle_y = -target_sine * (world_x - tx) + target_cosine * (
                        world_y - ty
                    )
                    if abs(obstacle_x) <= hx + 0.20 and abs(obstacle_y) <= hy + 0.20:
                        marked = True
                        break
            detected_targets += int(marked)
        record = {
            "schema_version": 1,
            "episode_id": episode_id,
            "reset_generation": generation,
            "depth_frame_token": token,
            "source_depth_age_sec": age,
            "expected_target_count": len(targets),
            "detected_target_count": detected_targets,
            "detected_frame": detected_targets > 0,
            "costmap_frame": str(message.header.frame_id),
            "costmap_resolution_m": resolution,
            "wall_time_unix": time.time(),
        }
        with self.lock:
            if token == self.last_costmap_scored_token:
                return
            self.last_costmap_scored_token = token
            self.costmap_expected_frame_count += 1
            self.costmap_detected_frame_count += int(detected_targets > 0)
            self.costmap_expected_target_count += len(targets)
            self.costmap_detected_target_count += detected_targets
            bucket = self.costmap_by_generation.setdefault(
                generation,
                {
                    "expected_frame_count": 0,
                    "detected_frame_count": 0,
                    "expected_target_count": 0,
                    "detected_target_count": 0,
                },
            )
            bucket["expected_frame_count"] += 1
            bucket["detected_frame_count"] += int(detected_targets > 0)
            bucket["expected_target_count"] += len(targets)
            bucket["detected_target_count"] += detected_targets
        with self.costmap_records_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def _on_stop(self, message: Bool) -> None:
        with self.lock:
            self.stop_latched = bool(message.data)

    def _on_motion_enabled(self, message: Bool) -> None:
        with self.lock:
            self.motion_enabled = bool(message.data)

    def _on_navigation(self, message: NavigationCommand) -> None:
        identity = (
            str(message.episode_id),
            int(message.reset_generation),
            int(message.sequence_id),
        )
        now = _semantic_now(self)
        with self.lock:
            previous = self.active_identity
            if previous[1] >= 0 and identity[1] < previous[1]:
                self.stale_count += 1
                return
            if previous[:2] != identity[:2]:
                self.reset_barrier_count += int(previous[1] >= 0)
                self._state_replay_epoch += 1
                self.last_safe_cmd = (0.0, 0.0)
                self.last_safe_cmd_monotonic = 0.0
                self.motion_enabled = False
                self.stop_latched = True
                self.latest_state = None
                self.latest_state_monotonic = 0.0
                self.latest_pointcloud = None
                self.latest_pointcloud_monotonic = 0.0
                self.latest_detected_targets_map.clear()
                self.latest_detected_target_monotonic = 0.0
                self.latest_detected_target_token = -1
                self.latest_detected_target_generation = -1
                self.latest_detected_target_episode = ""
                self.active_path.clear()
            self.active_identity = identity
            self.navigation_barrier_monotonic = now
            self.navigation_barrier_cmd_serial = self.safe_cmd_serial
            # The typed command is the authoritative STOP contract.  The
            # model client also publishes /internvla/stop, while the Oracle
            # client intentionally has no duplicate stop publisher.
            if self._t5_sim_time_semantics and now <= 0.0:
                self.motion_enabled = False
                self.stop_latched = True
            else:
                self.stop_latched = bool(message.stop)

    def _on_active_path(self, message: NavPath) -> None:
        points = [
            (float(item.pose.position.x), float(item.pose.position.y))
            for item in message.poses
        ]
        with self.lock:
            self.active_path = points

    def _republish_state(self) -> None:
        with self.lock:
            replay_epoch = self._state_replay_epoch
            state_age = _semantic_age(self, self.latest_state_monotonic)
            if (
                self.latest_state is None
                or state_age is None
                or state_age > self.state_republish_timeout
            ):
                return
            pose, linear_velocity, angular_velocity = self.latest_state
            state = (list(pose), list(linear_velocity), list(angular_velocity))
            path = list(self.active_path)
            pointcloud = None
            pointcloud_age = _semantic_age(self, self.latest_pointcloud_monotonic)
            if (
                not self.motion_enabled
                and self.latest_pointcloud is not None
                and pointcloud_age is not None
                and pointcloud_age <= self.state_republish_timeout
            ):
                pointcloud = list(self.latest_pointcloud)
        with self.lock:
            if replay_epoch != self._state_replay_epoch:
                return
        self._publish_state(*state, path)
        if pointcloud is not None:
            # Isaac physics is paused while the agent waits for Nav2, so the
            # last camera frame remains current. Stop replay immediately once
            # motion is enabled; moving operation then requires live depth.
            self._publish_pointcloud(pointcloud)

    def _serve(self) -> None:
        if self.ipc_transport == "unix":
            if self.socket_path.exists():
                self.socket_path.unlink()
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
        else:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.tcp_bind_host, self.tcp_port))
        listener.listen(1)
        listener.settimeout(0.5)
        try:
            while not self.stop_event.is_set():
                try:
                    connection, address = listener.accept()
                except socket.timeout:
                    continue
                if self.ipc_transport == "tcp" and address[0] != self.tcp_expected_peer_ip:
                    self.ipc_rejected_peer_count += 1
                    connection.close()
                    continue
                self.ipc_accepted_connection_count += 1
                with connection:
                    connection.settimeout(1.0)
                    while not self.stop_event.is_set():
                        try:
                            request = _recv_json(connection)
                            response = self._handle_ipc_request(request)
                            _send_json(connection, response)
                        except (ConnectionError, BrokenPipeError, socket.timeout):
                            break
                        except BaseException as exc:
                            try:
                                _send_json(
                                    connection,
                                    {
                                        "schema_version": 1,
                                        "status": "error",
                                        "message": repr(exc)[:512],
                                        "linear_x": 0.0,
                                        "angular_z": 0.0,
                                        "emergency_stop": True,
                                    },
                                )
                            except OSError:
                                break
        finally:
            listener.close()
            if self.ipc_transport == "unix" and self.socket_path.exists():
                self.socket_path.unlink()

    def _handle_ipc_request(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("operation") != "query_active_identity":
            return self._handle_update(request)
        if int(request.get("schema_version", 0)) != 1:
            raise ValueError("unsupported controller IPC schema")
        with self.lock:
            return _active_identity_query_response(
                self.active_identity, self._state_replay_epoch
            )

    def _handle_update(self, request: dict[str, Any]) -> dict[str, Any]:
        if int(request.get("schema_version", 0)) != 1 or request.get("operation") != "update":
            raise ValueError("unsupported controller IPC operation")
        episode_id = str(request.get("episode_id", ""))
        generation = int(request.get("reset_generation", -1))
        sequence = int(request.get("sequence_id", -1))
        pose = [float(value) for value in request.get("pose_wxyz", [])]
        linear_velocity = [float(value) for value in request.get("linear_velocity", [])]
        angular_velocity = [float(value) for value in request.get("angular_velocity", [])]
        if len(pose) != 7 or len(linear_velocity) != 3 or len(angular_velocity) != 3:
            raise ValueError("invalid controller state vector")
        finite = all(math.isfinite(value) for value in pose + linear_velocity + angular_velocity)
        identity = (episode_id, generation, sequence)
        state_only = bool(request.get("state_only", False))
        with self.lock:
            replay_epoch = self._state_replay_epoch
            state_identity_current = _state_update_is_current(
                t5_sim_time_semantics=self._t5_sim_time_semantics,
                update_identity=identity,
                active_identity=self.active_identity,
                update_epoch=replay_epoch,
                current_epoch=self._state_replay_epoch,
            )
        if generation >= 0 and finite and state_identity_current:
            self._publish_static_map(
                generation,
                pose,
                identity=identity,
                replay_epoch=replay_epoch,
            )
        request_nan = bool(request.get("nan_detected", False)) or not finite
        request_fall = bool(request.get("fallen", False))
        request_collision = bool(request.get("physical_collision", False))
        collision_force = float(request.get("maximum_collision_force", 0.0))
        wall_now = time.monotonic()
        semantic_now = _semantic_now(self)
        with self.lock:
            if (
                not state_only
                and self.last_update_monotonic
                and identity == self.last_update_identity
            ):
                interval = wall_now - self.last_update_monotonic
                minimum_paced_interval = 0.5 / self.expected_control_hz
                maximum_paced_interval = 2.5 / self.expected_control_hz
                if minimum_paced_interval <= interval <= maximum_paced_interval:
                    self.update_intervals.append(interval)
            self.last_update_monotonic = wall_now
            self.last_update_identity = identity
            self.update_count += 1
            self.state_only_count += int(state_only)
            if request_nan:
                self.nan_count += 1
            if request_fall:
                self.fall_count += 1
            state_identity_current = _state_update_is_current(
                t5_sim_time_semantics=self._t5_sim_time_semantics,
                update_identity=identity,
                active_identity=self.active_identity,
                update_epoch=replay_epoch,
                current_epoch=self._state_replay_epoch,
                state_only=state_only,
            )
            identity_valid = identity == self.active_identity
            command_age = _semantic_age(self, self.last_safe_cmd_monotonic)
            fresh = _command_is_fresh(
                safe_cmd_serial=self.safe_cmd_serial,
                barrier_cmd_serial=self.navigation_barrier_cmd_serial,
                safe_cmd_stamp=self.last_safe_cmd_monotonic,
                barrier_stamp=self.navigation_barrier_monotonic,
                command_age=command_age,
                timeout_sec=self.command_timeout,
            )
            if request_collision and not self._physical_collision_active:
                self.physical_collision_count += 1
            self._physical_collision_active = request_collision
            state_hazard = _runtime_state_hazard(
                t5_completion_sim=self._t5_sim_time_semantics,
                nan_detected=request_nan,
                fallen=request_fall,
                physical_collision=request_collision,
            )
            if state_only:
                emergency = state_hazard
                timed_out = False
                enabled = False
            else:
                emergency = state_hazard or not identity_valid
                if not identity_valid:
                    self.stale_count += 1
                timed_out = not fresh
                if timed_out:
                    self.timeout_count += 1
                enabled = (
                    identity_valid
                    and fresh
                    and self.motion_enabled
                    and not self.stop_latched
                    and not emergency
                )
            desired = self.last_safe_cmd if enabled else (0.0, 0.0)
            if emergency:
                self.estop_count += 1
            if state_identity_current:
                self.latest_state = (
                    list(pose),
                    list(linear_velocity),
                    list(angular_velocity),
                )
                self.latest_state_monotonic = semantic_now
            path = list(self.active_path)
        with self.lock:
            state_identity_current = _state_update_is_current(
                t5_sim_time_semantics=self._t5_sim_time_semantics,
                update_identity=identity,
                active_identity=self.active_identity,
                update_epoch=replay_epoch,
                current_epoch=self._state_replay_epoch,
                state_only=state_only,
            )
            if state_identity_current:
                lateral_error, heading_error = self._publish_state(
                    pose, linear_velocity, angular_velocity, path
                )
            else:
                lateral_error, heading_error = 0.0, 0.0
        yaw = _yaw([pose[3], pose[4], pose[5], pose[6]])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        body_linear_velocity = [
            cosine * linear_velocity[0] + sine * linear_velocity[1],
            -sine * linear_velocity[0] + cosine * linear_velocity[1],
            linear_velocity[2],
        ]
        (
            point_count,
            obstacle_expected,
            obstacle_detected,
            stop_polygon_point_count,
            stop_polygon_samples,
        ) = (
            self._publish_depth(
                request,
                identity=identity,
                replay_epoch=replay_epoch,
            )
            if state_identity_current
            else (0, 0, 0, 0, [])
        )
        record = {
            "schema_version": 1,
            "update_index": self.update_count - 1,
            "episode_id": episode_id,
            "reset_generation": generation,
            "sequence_id": sequence,
            "state_only": state_only,
            "identity_valid": identity_valid,
            "motion_enabled": enabled,
            "command_fresh": fresh,
            "command_timeout": timed_out,
            # T5 completion_sim uses ROS simulation time for freshness.  Keep
            # the exact per-update age so candidate analysis never has to
            # infer it from wall latency or a boolean timeout outcome.
            "command_age_sec": command_age,
            "collision_monitor_stopping": self._monitor_was_stopping,
            "desired_linear_x": desired[0],
            "desired_angular_z": desired[1],
            "actual_linear_velocity_world": linear_velocity,
            "actual_linear_velocity_base": body_linear_velocity,
            "actual_angular_velocity": angular_velocity,
            "pose_xyz_wxyz": pose,
            "lateral_error_m": lateral_error,
            "heading_error_rad": heading_error,
            "point_count": point_count,
            "stop_polygon_point_count": stop_polygon_point_count,
            "stop_polygon_samples_xyz": stop_polygon_samples,
            "robot_link_centers_base": request.get("robot_link_centers_base", []),
            "support_plane_world_z": request.get("support_plane_world_z"),
            "obstacle_expected_target_count": obstacle_expected,
            "obstacle_detected_target_count": obstacle_detected,
            "obstacle_expected_frame_count": int(obstacle_expected > 0),
            "obstacle_detected_frame_count": int(
                obstacle_expected > 0 and obstacle_detected > 0
            ),
            "obstacle_targets_base": request.get("obstacle_targets_base", []),
            "fallen": request_fall,
            "nan_detected": request_nan,
            "physical_collision": request_collision,
            "physical_collision_warn_only": bool(
                request_collision and self._t5_sim_time_semantics
            ),
            "obstacle_scenario": str(request.get("obstacle_scenario", "")),
            "maximum_collision_force": collision_force,
            "collision_pairs": request.get("collision_pairs", []),
            "emergency_stop": emergency,
            "wall_time_unix": time.time(),
        }
        self._append(record)
        if self.update_count % 20 == 0:
            self._write_summary("RUNNING")
        return {
            "schema_version": 1,
            "status": "ok",
            "linear_x": float(desired[0]),
            "angular_z": float(desired[1]),
            "emergency_stop": bool(emergency or state_only),
            "command_timeout": bool(timed_out),
            "motion_enabled": bool(enabled),
        }

    def _publish_state(
        self,
        pose: list[float],
        linear_velocity: list[float],
        angular_velocity: list[float],
        path: list[tuple[float, float]],
    ) -> tuple[float, float]:
        x, y, z, qw, qx, qy, qz = pose
        stamp = self.get_clock().now().to_msg()
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z
        transform.transform.rotation.w = qw
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        self.tf_broadcaster.sendTransform(transform)
        odom = Odometry()
        odom.header = transform.header
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        odom.pose.pose.orientation = transform.transform.rotation
        yaw = _yaw([qw, qx, qy, qz])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        odom.twist.twist.linear.x = cosine * linear_velocity[0] + sine * linear_velocity[1]
        odom.twist.twist.linear.y = -sine * linear_velocity[0] + cosine * linear_velocity[1]
        odom.twist.twist.linear.z = linear_velocity[2]
        odom.twist.twist.angular.x = angular_velocity[0]
        odom.twist.twist.angular.y = angular_velocity[1]
        odom.twist.twist.angular.z = angular_velocity[2]
        self.odom_publisher.publish(odom)
        velocity = TwistStamped()
        velocity.header.stamp = stamp
        velocity.header.frame_id = "base_link"
        velocity.twist = odom.twist.twist
        self.velocity_publisher.publish(velocity)

        lateral = 0.0
        heading = 0.0
        if path:
            nearest = min(range(len(path)), key=lambda index: math.hypot(path[index][0] - x, path[index][1] - y))
            lateral = math.hypot(path[nearest][0] - x, path[nearest][1] - y)
            if len(path) > 1:
                next_index = min(nearest + 1, len(path) - 1)
                previous_index = max(0, next_index - 1)
                dx = path[next_index][0] - path[previous_index][0]
                dy = path[next_index][1] - path[previous_index][1]
                path_yaw = math.atan2(dy, dx)
                heading = math.atan2(
                    math.sin(path_yaw - _yaw([qw, qx, qy, qz])),
                    math.cos(path_yaw - _yaw([qw, qx, qy, qz])),
                )
        error = Float64MultiArray()
        error.data = [lateral, heading]
        self.error_publisher.publish(error)
        return lateral, heading

    def _publish_pointcloud(self, points: list[tuple[float, float, float]]) -> None:
        message = PointCloud2()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.height = 1
        message.width = len(points)
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = 12
        message.row_step = 12 * len(points)
        message.is_dense = True
        message.data = b"".join(struct.pack("<fff", *point) for point in points)
        self.pointcloud_publisher.publish(message)

    def _publish_depth(
        self,
        request: dict[str, Any],
        *,
        identity: tuple[str, int, int],
        replay_epoch: int,
    ) -> tuple[int, int, int, int, list[list[float]]]:
        values = request.get("depth_values")
        if not values:
            return 0, 0, 0, 0, []
        height = int(request.get("depth_height", 0))
        width = int(request.get("depth_width", 0))
        row_stride = int(request.get("depth_row_stride", 1))
        column_stride = int(request.get("depth_column_stride", 1))
        if height <= 0 or width <= 0 or len(values) != height * width:
            raise ValueError("invalid sampled depth payload")
        raw_link_centers = request.get("robot_link_centers_base", [])
        if not isinstance(raw_link_centers, list):
            raise ValueError("robot link center evidence must be a list")
        link_centers: list[tuple[str, tuple[float, float, float]]] = []
        for item in raw_link_centers:
            if not isinstance(item, dict):
                raise ValueError("invalid robot link center evidence")
            name = str(item.get("name", ""))
            center = item.get("center_base", [])
            if not name or not isinstance(center, list) or len(center) != 3:
                raise ValueError("invalid robot link center evidence")
            center_values = tuple(float(value) for value in center)
            if not all(math.isfinite(value) for value in center_values):
                raise ValueError("non-finite robot link center evidence")
            link_centers.append((name, center_values))
        support_plane_value = request.get("support_plane_world_z")
        support_plane = (
            float(support_plane_value) if support_plane_value is not None else None
        )
        if support_plane is not None and not math.isfinite(support_plane):
            raise ValueError("non-finite support plane evidence")
        pose = [float(value) for value in request.get("pose_wxyz", [])]
        if len(pose) != 7:
            raise ValueError("support-plane filtering requires a valid pose")
        w, qx, qy, qz = pose[3:]
        gravity_row = (
            2.0 * (qx * qz - qy * w),
            2.0 * (qy * qz + qx * w),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )

        def robot_self_return(point: list[float]) -> bool:
            for name, center in link_centers:
                dx = point[0] - center[0]
                dy = point[1] - center[1]
                dz = point[2] - center[2]
                if name == "base":
                    if abs(dx) <= 0.34 and abs(dy) <= 0.18 and abs(dz) <= 0.14:
                        return True
                    continue
                radius = 0.12 if name.startswith("Head_") else 0.10
                if name.endswith("_calf"):
                    radius = 0.09
                elif name.endswith("_foot"):
                    radius = 0.08
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    return True
            return False

        points: list[tuple[float, float, float]] = []
        for row in range(height):
            pixel_v = row * row_stride
            for column in range(width):
                depth = float(values[row * width + column])
                if not math.isfinite(depth) or depth <= 0.1 or depth > 6.0:
                    continue
                pixel_u = column * column_stride
                camera_x = (pixel_u - 320.0) * depth / 585.0
                camera_y = (240.0 - pixel_v) * depth / 585.0
                camera_z = -depth
                base = []
                for axis in range(3):
                    base.append(
                        CAMERA_TRANSLATION[axis]
                        + CAMERA_ROTATION[axis][0] * camera_x
                        + CAMERA_ROTATION[axis][1] * camera_y
                        + CAMERA_ROTATION[axis][2] * camera_z
                    )
                world_z = pose[2] + sum(
                    gravity_row[axis] * base[axis] for axis in range(3)
                )
                ground_return = (
                    support_plane is not None
                    and world_z <= support_plane + 0.08
                )
                if (
                    base[0] > 0.12
                    and not ground_return
                    and not robot_self_return(base)
                ):
                    points.append((base[0], base[1], base[2]))
        targets = request.get("obstacle_targets_base", [])
        if not isinstance(targets, list):
            raise ValueError("obstacle target evidence must be a list")
        detected_targets = 0
        detected_target_values: list[tuple[float, float, float, float, float, float, float]] = []
        for target in targets:
            if not isinstance(target, list) or len(target) not in (6, 7):
                raise ValueError("invalid obstacle target evidence")
            tx, ty, tz, hx, hy, hz = [float(value) for value in target[:6]]
            target_yaw = float(target[6]) if len(target) == 7 else 0.0
            if not all(
                math.isfinite(value)
                for value in (tx, ty, tz, hx, hy, hz, target_yaw)
            ):
                raise ValueError("non-finite obstacle target evidence")
            target_cosine = math.cos(target_yaw)
            target_sine = math.sin(target_yaw)
            if any(
                abs(target_cosine * (px - tx) + target_sine * (py - ty))
                <= hx + 0.15
                and abs(-target_sine * (px - tx) + target_cosine * (py - ty))
                <= hy + 0.15
                and tz - hz + 0.08 <= pz <= tz + hz + 0.15
                for px, py, pz in points
            ):
                detected_targets += 1
                detected_target_values.append(
                    (tx, ty, tz, hx, hy, hz, target_yaw)
                )
        pose_yaw = _yaw(pose[3:])
        pose_cosine = math.cos(pose_yaw)
        pose_sine = math.sin(pose_yaw)
        detected_targets_map = [
            (
                pose[0] + pose_cosine * tx - pose_sine * ty,
                pose[1] + pose_sine * tx + pose_cosine * ty,
                hx,
                hy,
                pose_yaw + target_yaw,
            )
            for tx, ty, _tz, hx, hy, _hz, target_yaw in detected_target_values
        ]
        with self.lock:
            if not _state_update_is_current(
                t5_sim_time_semantics=self._t5_sim_time_semantics,
                update_identity=identity,
                active_identity=self.active_identity,
                update_epoch=replay_epoch,
                current_epoch=self._state_replay_epoch,
            ):
                return 0, 0, 0, 0, []
            self._publish_pointcloud(points)
            semantic_now = _semantic_now(self)
            self.latest_pointcloud = list(points)
            self.latest_pointcloud_monotonic = semantic_now
            self.depth_frame_count += 1
            self.depth_nonempty_count += int(bool(points))
            self.obstacle_expected_target_count += len(targets)
            self.obstacle_detected_target_count += detected_targets
            self.obstacle_expected_frame_count += int(bool(targets))
            self.obstacle_detected_frame_count += int(
                bool(targets) and detected_targets > 0
            )
            self.latest_detected_targets_map = detected_targets_map
            self.latest_detected_target_monotonic = semantic_now
            self.latest_detected_target_token = self.depth_frame_count
            self.latest_detected_target_generation = int(
                request.get("reset_generation", -1)
            )
            self.latest_detected_target_episode = str(request.get("episode_id", ""))
        stop_points = [
            point
            for point in points
            if -0.30 <= point[0] <= 0.60
            and -0.32 <= point[1] <= 0.32
            and 0.05 <= point[2] <= 1.5
        ]
        stop_points.sort(key=lambda point: math.hypot(point[0], point[1]))
        return (
            len(points),
            len(targets),
            detected_targets,
            len(stop_points),
            [[float(value) for value in point] for point in stop_points[:20]],
        )

    def _append(self, value: dict[str, Any]) -> None:
        with self.records_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")

    def _summary(self, status: str) -> dict[str, Any]:
        intervals = list(self.update_intervals)
        control_hz = 1.0 / (sum(intervals) / len(intervals)) if intervals else 0.0
        detection_rate = (
            self.depth_nonempty_count / self.depth_frame_count
            if self.depth_frame_count
            else 0.0
        )
        obstacle_target_detection_rate = (
            self.obstacle_detected_target_count / self.obstacle_expected_target_count
            if self.obstacle_expected_target_count
            else 0.0
        )
        obstacle_frame_detection_rate = (
            self.obstacle_detected_frame_count / self.obstacle_expected_frame_count
            if self.obstacle_expected_frame_count
            else 0.0
        )
        costmap_detection_rate = (
            self.costmap_detected_target_count / self.costmap_expected_target_count
            if self.costmap_expected_target_count
            else 0.0
        )
        return {
            "schema_version": 1,
            "status": status,
            "mode": "continuous_physics_cmd_vel",
            "started_unix": self.started_unix,
            "updated_unix": time.time(),
            "update_count": self.update_count,
            "state_only_count": self.state_only_count,
            "measured_control_hz": control_hz,
            "control_interval_sample_count": len(intervals),
            "expected_control_hz": self.expected_control_hz,
            "command_timeout_count": self.timeout_count,
            "stale_or_identity_reject_count": self.stale_count,
            "reset_barrier_count": self.reset_barrier_count,
            "estop_count": self.estop_count,
            "fall_count": self.fall_count,
            "nan_count": self.nan_count,
            "depth_frame_count": self.depth_frame_count,
            "depth_nonempty_count": self.depth_nonempty_count,
            "depth_nonempty_rate": detection_rate,
            "obstacle_expected_target_count": self.obstacle_expected_target_count,
            "obstacle_detected_target_count": self.obstacle_detected_target_count,
            "obstacle_target_detection_rate": obstacle_target_detection_rate,
            "obstacle_expected_frame_count": self.obstacle_expected_frame_count,
            "obstacle_detected_frame_count": self.obstacle_detected_frame_count,
            "obstacle_frame_detection_rate": obstacle_frame_detection_rate,
            "obstacle_sensor_frame_detection_rate": obstacle_frame_detection_rate,
            "costmap_expected_frame_count": self.costmap_expected_frame_count,
            "costmap_detected_frame_count": self.costmap_detected_frame_count,
            "costmap_expected_target_count": self.costmap_expected_target_count,
            "costmap_detected_target_count": self.costmap_detected_target_count,
            "costmap_detection_rate": costmap_detection_rate,
            "costmap_by_generation": self.costmap_by_generation,
            "static_map_manifest_sha256": self.static_map_manifest_sha256,
            "static_map_generation": self.static_map_generation,
            "static_map_publish_count": self.static_map_publish_count,
            "static_map_selection_count": len(self.static_map_selections),
            "static_map_max_start_xy_error_m": max(
                (
                    float(item["start_xy_error_m"])
                    for item in self.static_map_selections
                ),
                default=0.0,
            ),
            "static_map_selections": self.static_map_selections,
            "static_map_map_count": len(self.static_map_entries),
            # Gate metric: proportion of real depth-acquired obstacle targets
            # independently observed as occupied in Nav2's local costmap.
            "obstacle_detection_rate": costmap_detection_rate,
            "collision_monitor_stop_count": self.collision_monitor_stop_count,
            "collision_monitor_recovery_count": self.collision_monitor_recovery_count,
            "physical_collision_count": self.physical_collision_count,
            "flash_call_count": 0,
            "cmd_vel_quantization_count": 0,
            "direct_motion_bypass_count": 0,
            "ipc_transport": self.ipc_transport,
            "ipc_tcp_bind_host": (
                self.tcp_bind_host if self.ipc_transport == "tcp" else None
            ),
            "ipc_tcp_port": self.tcp_port if self.ipc_transport == "tcp" else None,
            "ipc_expected_peer_ip": (
                self.tcp_expected_peer_ip if self.ipc_transport == "tcp" else None
            ),
            "ipc_accepted_connection_count": self.ipc_accepted_connection_count,
            "ipc_rejected_peer_count": self.ipc_rejected_peer_count,
        }

    def _write_summary(self, status: str) -> None:
        temporary = self.result_dir / "controller_summary.json.tmp"
        final = self.result_dir / "controller_summary.json"
        temporary.write_text(
            json.dumps(self._summary(status), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, final)

    def finish(self) -> None:
        self.stop_event.set()
        self._write_summary("FINISHED")
        if self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = Go2ControllerBridge()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
