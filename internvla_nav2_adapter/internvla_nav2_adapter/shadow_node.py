"""TF2/costmap/direction shadow gate; this node never commands the robot."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from internvla_ros2_msgs.msg import NavigationCommand
from nav_msgs.msg import OccupancyGrid, Path as NavPath
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def _time_ns(message: Any) -> int:
    return int(message.sec) * 1_000_000_000 + int(message.nanosec)


def _yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quaternion(yaw: float) -> tuple[float, float]:
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _expected_first_action(points: np.ndarray) -> int:
    if len(points) < 2 or float(np.linalg.norm(points[0] - points[-1])) <= 0.2:
        return -1
    pos = points[0]
    goal = points[-1]
    nearest = int(np.argmin(np.linalg.norm(points - pos, axis=1)))
    target = points[min(nearest + 4, len(points) - 1)]
    vector = target - pos
    if float(np.linalg.norm(vector)) < 1e-6:
        return -1
    angle = math.atan2(float(vector[1]), float(vector[0]))
    turns = int(round(angle / math.radians(15.0)))
    if turns > 0:
        return 2
    if turns < 0:
        return 3
    next_pos = pos + np.asarray([0.25, 0.0])
    return 1 if np.linalg.norm(next_pos - goal) <= np.linalg.norm(pos - goal) else -1


class InternVLANav2Shadow(Node):
    def __init__(self) -> None:
        super().__init__("internvla_nav2_shadow")
        self.declare_parameter(
            "result_dir", os.environ.get("INTERNVLA_NAV2_SHADOW_RESULT_DIR", "")
        )
        self.declare_parameter("goal_max_distance_m", 2.0)
        self.declare_parameter("goal_min_distance_m", 0.05)
        self.declare_parameter("occupied_threshold", 50)
        result_value = str(self.get_parameter("result_dir").value)
        if not result_value:
            raise RuntimeError("result_dir is required")
        self.result_dir = Path(result_value).resolve()
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.result_dir / "records.jsonl"
        if self.records_path.exists():
            raise RuntimeError("refusing to append an existing Nav2 shadow record")
        self.summary_path = self.result_dir / "summary.json"
        self.maximum_distance = float(self.get_parameter("goal_max_distance_m").value)
        self.minimum_distance = float(self.get_parameter("goal_min_distance_m").value)
        self.occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        self.callback_group = ReentrantCallbackGroup()
        self.lock = threading.RLock()
        self.map: OccupancyGrid | None = None
        self.tf_buffer = Buffer(cache_time=Duration(seconds=60.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=16,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            NavigationCommand,
            "/internvla/navigation_command",
            self._on_command,
            qos,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            OccupancyGrid,
            "/map",
            self._on_map,
            map_qos,
            callback_group=self.callback_group,
        )
        self.goal_publisher = self.create_publisher(
            PoseStamped, "/internvla/nav2_shadow/goal", qos
        )
        self.path_publisher = self.create_publisher(
            NavPath, "/internvla/nav2_shadow/path", qos
        )
        self.started = time.time()
        self.command_count = 0
        self.trajectory_count = 0
        self.tf_success = 0
        self.tf_failure = 0
        self.stale_count = 0
        self.cross_episode_pollution = 0
        self.local_goal_valid = 0
        self.direction_checks = 0
        self.direction_matches = 0
        self.map_missing = 0
        self.current_episode = ""
        self.current_generation = -1
        self.last_sequence = -1
        self.reset_count = 0
        self._write_summary("WAITING")

    def _on_map(self, message: OccupancyGrid) -> None:
        with self.lock:
            self.map = message

    def _on_command(self, command: NavigationCommand) -> None:
        with self.lock:
            record: dict[str, Any] = {
                "schema_version": 1,
                "command_index": self.command_count,
                "episode_id": command.episode_id,
                "reset_generation": int(command.reset_generation),
                "sequence_id": int(command.sequence_id),
                "trajectory_valid": bool(command.trajectory_valid),
                "tf_success": False,
                "stale": False,
                "local_goal_valid": False,
                "direction_checked": False,
                "direction_match": False,
            }
            self.command_count += 1
            self._check_barrier(command)
            now_ns = self.get_clock().now().nanoseconds
            if now_ns > _time_ns(command.valid_until):
                self.stale_count += 1
                record["stale"] = True
            if command.trajectory_valid:
                self.trajectory_count += 1
                self._check_trajectory(command, record)
            self._append(record)
            self._write_summary("PASS_SO_FAR")

    def _check_barrier(self, command: NavigationCommand) -> None:
        generation = int(command.reset_generation)
        sequence = int(command.sequence_id)
        if self.current_generation < 0:
            if generation < 0 or sequence != 0:
                self.cross_episode_pollution += 1
        elif generation == self.current_generation:
            if command.episode_id != self.current_episode or sequence != self.last_sequence + 1:
                self.cross_episode_pollution += 1
        elif generation > self.current_generation:
            self.reset_count += 1
            if sequence != 0:
                self.cross_episode_pollution += 1
        else:
            self.cross_episode_pollution += 1
        self.current_episode = command.episode_id
        self.current_generation = generation
        self.last_sequence = sequence

    def _check_trajectory(self, command: NavigationCommand, record: dict[str, Any]) -> None:
        path = command.local_path
        points = np.asarray(
            [[pose.pose.position.x, pose.pose.position.y] for pose in path.poses],
            dtype=np.float64,
        )
        record["trajectory_points"] = int(len(points))
        if (
            command.trajectory_source != 1
            or path.header.frame_id != "base_link"
            or points.ndim != 2
            or points.shape[1:] != (2,)
            or not len(points)
            or not np.isfinite(points).all()
        ):
            self.tf_failure += 1
            record["tf_error"] = "invalid local trajectory contract"
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                Time.from_msg(command.header.stamp),
                timeout=Duration(seconds=1.0),
            )
        except TransformException as exc:
            self.tf_failure += 1
            record["tf_error"] = str(exc)[:512]
            return
        self.tf_success += 1
        record["tf_success"] = True
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        base_yaw = _yaw(rotation.x, rotation.y, rotation.z, rotation.w)
        cosine, sine = math.cos(base_yaw), math.sin(base_yaw)
        transformed = NavPath()
        transformed.header.stamp = command.header.stamp
        transformed.header.frame_id = "map"
        map_points: list[tuple[float, float]] = []
        for local_pose in path.poses:
            x = float(local_pose.pose.position.x)
            y = float(local_pose.pose.position.y)
            map_x = float(translation.x) + cosine * x - sine * y
            map_y = float(translation.y) + sine * x + cosine * y
            local_q = local_pose.pose.orientation
            pose_yaw = base_yaw + _yaw(local_q.x, local_q.y, local_q.z, local_q.w)
            z_value, w_value = _quaternion(pose_yaw)
            pose = PoseStamped()
            pose.header = transformed.header
            pose.pose.position.x = map_x
            pose.pose.position.y = map_y
            pose.pose.position.z = float(translation.z)
            pose.pose.orientation.z = z_value
            pose.pose.orientation.w = w_value
            transformed.poses.append(pose)
            map_points.append((map_x, map_y))
        self.path_publisher.publish(transformed)

        radial = np.linalg.norm(points - points[0], axis=1)
        candidates = np.nonzero(radial <= self.maximum_distance + 1e-6)[0]
        target_index = int(candidates[-1]) if len(candidates) else 0
        target_distance = float(radial[target_index])
        record["target_distance_m"] = target_distance
        expected = _expected_first_action(points)
        self.direction_checks += 1
        self.direction_matches += int(expected == int(command.discrete_action))
        record["direction_checked"] = True
        record["expected_first_action"] = expected
        record["observed_action"] = int(command.discrete_action)
        record["direction_match"] = expected == int(command.discrete_action)

        if self.map is None:
            self.map_missing += 1
            record["goal_error"] = "map missing"
            return
        target = map_points[target_index]
        start = map_points[0]
        if target_distance < self.minimum_distance or not self._line_is_free(start, target):
            record["goal_error"] = "target too short or occupied/unreachable"
            return
        goal = transformed.poses[target_index]
        self.goal_publisher.publish(goal)
        self.local_goal_valid += 1
        record["local_goal_valid"] = True
        record["goal_map_xy"] = [target[0], target[1]]

    def _line_is_free(self, start: tuple[float, float], target: tuple[float, float]) -> bool:
        assert self.map is not None
        info = self.map.info
        resolution = float(info.resolution)
        distance = math.dist(start, target)
        samples = max(2, int(math.ceil(distance / max(resolution * 0.5, 0.01))) + 1)
        for ratio in np.linspace(0.0, 1.0, samples):
            x = start[0] + (target[0] - start[0]) * float(ratio)
            y = start[1] + (target[1] - start[1]) * float(ratio)
            column = int(math.floor((x - info.origin.position.x) / resolution))
            row = int(math.floor((y - info.origin.position.y) / resolution))
            if column < 0 or row < 0 or column >= info.width or row >= info.height:
                return False
            value = int(self.map.data[row * info.width + column])
            if value < 0 or value > self.occupied_threshold:
                return False
        return True

    def _append(self, record: dict[str, Any]) -> None:
        with self.records_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _write_summary(self, status: str) -> None:
        _atomic_json(
            self.summary_path,
            {
                "schema_version": 1,
                "status": status,
                "mode": "nav2_shadow_no_control",
                "command_count": self.command_count,
                "trajectory_count": self.trajectory_count,
                "tf_success_count": self.tf_success,
                "tf_failure_count": self.tf_failure,
                "tf_success_rate": self.tf_success / self.trajectory_count
                if self.trajectory_count
                else 0.0,
                "stale_execution_count": self.stale_count,
                "cross_episode_pollution_count": self.cross_episode_pollution,
                "local_goal_valid_count": self.local_goal_valid,
                "local_goal_valid_rate": self.local_goal_valid / self.trajectory_count
                if self.trajectory_count
                else 0.0,
                "direction_check_count": self.direction_checks,
                "direction_match_count": self.direction_matches,
                "direction_match_rate": self.direction_matches / self.direction_checks
                if self.direction_checks
                else 0.0,
                "map_missing_count": self.map_missing,
                "reset_count": self.reset_count,
                "started_unix": self.started,
                "updated_unix": time.time(),
            },
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = InternVLANav2Shadow()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
