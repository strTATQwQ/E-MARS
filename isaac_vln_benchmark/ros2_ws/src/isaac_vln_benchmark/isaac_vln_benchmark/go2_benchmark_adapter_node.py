from __future__ import annotations

import json
import math
import socket
import zlib
import time
from dataclasses import dataclass
from typing import Any

from .config_loader import load_data, normalize_scene_to_robot_origin, object_by_id, scene_by_type
from .metrics import SuccessJudgeCore, bearing_deg, distance_xy, object_visible, path_obstacle_status, wrap_to_pi
from .v10_geometry_utils import (
    branch_membership,
    branch_yaw_deg,
    clamp,
    crawl_turn_command,
    expected_branch,
    intersection_center,
    rad2deg,
    target_vector,
    update_yaw_polarity,
)


TARGET_HOLD_PHASES = {"stop", "verify", "done", "await_step_track", "await_step_stop", "failsafe"}


@dataclass
class PlanarPose:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.yaw]

    def integrate(self, linear_x: float, angular_z: float, dt: float) -> None:
        if dt <= 0.0:
            return
        self.x += float(linear_x) * math.cos(self.yaw) * dt
        self.y += float(linear_x) * math.sin(self.yaw) * dt
        self.yaw = wrap_to_pi(self.yaw + float(angular_z) * dt)


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def isaac_twist_payload(linear_x: float, linear_y: float, angular_z: float, *, seq: int, timestamp: float) -> dict[str, Any]:
    return {
        "vx": float(linear_x),
        "vy": float(linear_y),
        "wz": float(angular_z),
        "seq": int(seq),
        "timestamp": float(timestamp),
        "source": "go2_benchmark_adapter",
    }


def normalize_isaac_pose(pose: list[Any], heading_sign: float = 1.0) -> PlanarPose:
    if len(pose) < 3:
        raise ValueError("Isaac pose requires x, y, and heading")
    return PlanarPose(float(pose[0]), float(pose[1]), float(heading_sign) * float(pose[2]))


def root_height_unstable(root_z: Any, threshold_m: float = 0.18) -> bool:
    return root_z is not None and float(root_z) < float(threshold_m)


def step_payload_matches_episode(payload: dict[str, Any], active_episode_id: str) -> bool:
    episode_id = str(payload.get("episode_id") or payload.get("mission_id") or "")
    return not (active_episode_id and episode_id and episode_id != active_episode_id)


def accepted_semantic_step_stop(
    payload: dict[str, Any],
    *,
    require_confirmed_track: bool = False,
    actual_distance_m: float | None = None,
    max_actual_distance_m: float | None = None,
) -> bool:
    accepted = bool(
        payload.get("stop", False)
        and payload.get("target_visible", False)
        and payload.get("estimated_distance_ok", False)
    )
    if accepted and max_actual_distance_m is not None:
        accepted = actual_distance_m is not None and actual_distance_m <= max_actual_distance_m
    if not accepted or not require_confirmed_track:
        return accepted
    track = payload.get("track") if isinstance(payload.get("track"), dict) else {}
    return bool(track.get("confirmed") and track.get("visible"))


def accepted_sensor_semantic_step_stop(payload: dict[str, Any]) -> bool:
    track = payload.get("track") if isinstance(payload.get("track"), dict) else {}
    return bool(
        payload.get("stop")
        and payload.get("target_visible")
        and payload.get("estimated_distance_ok")
        and track.get("confirmed")
        and track.get("fresh")
        and track.get("visible")
        and str(track.get("source") or "") == "actual_sensor_spatial_track"
        and str(track.get("fusion_source") or "")
        == "two_frame_grounded_sam_plus_step_confirmation"
    )


def simple_goal_stop_should_latch(task: dict[str, Any], status: dict[str, Any]) -> bool:
    if str(task.get("task_type") or "") != "simple_navigation":
        return False
    success = task.get("success") if isinstance(task.get("success"), dict) else {}
    try:
        distance = float(status.get("distance_to_target"))
        threshold = float(success.get("distance_to_target_m", 2.0))
    except (TypeError, ValueError):
        return False
    if bool(success.get("target_visible", False)) and not bool(status.get("target_visible", False)):
        return False
    return distance <= threshold


def target_max_yaw_rate(mode: dict[str, Any]) -> float:
    requested = float(mode.get("target_max_yaw_rate_radps", 0.30))
    if bool(mode.get("low_speed_primitive_stabilizer", False)):
        requested = min(requested, float(mode.get("target_low_speed_max_yaw_rate_radps", 0.15)))
    return requested


def target_phase_timeout_sec(mode: dict[str, Any]) -> float:
    requested = float(mode.get("target_max_phase_duration_sec", 20.0))
    if bool(mode.get("low_speed_primitive_stabilizer", False)):
        requested = max(requested, float(mode.get("target_low_speed_max_phase_duration_sec", 210.0)))
    return requested


def target_crawl_max_yaw_error_deg(mode: dict[str, Any]) -> float:
    requested = float(mode.get("target_crawl_max_yaw_error_deg", 60.0))
    if bool(mode.get("low_speed_primitive_stabilizer", False)):
        requested = max(requested, float(mode.get("target_low_speed_crawl_max_yaw_error_deg", 180.0)))
    return requested


def target_yaw_polarity_adaptation_enabled(mode: dict[str, Any]) -> bool:
    return bool(mode.get("target_yaw_polarity_adaptation", False))


def choose_bypass_side(robot_y: float, target_y: float, deadband_m: float = 0.25) -> float:
    delta = float(target_y) - float(robot_y)
    if abs(delta) <= float(deadband_m):
        return 1.0
    return 1.0 if delta > 0.0 else -1.0


def bypass_waypoint(
    obstacle_pose: list[Any],
    target_pose: list[Any],
    side: float,
    *,
    release_after_x_m: float,
    margin_x_m: float,
    offset_y_m: float,
) -> list[float]:
    ox, oy = float(obstacle_pose[0]), float(obstacle_pose[1])
    target_x = float(target_pose[0])
    return [
        min(target_x, ox + float(release_after_x_m) + float(margin_x_m)),
        oy + float(side) * float(offset_y_m),
        float(target_pose[2]) if len(target_pose) > 2 else 0.0,
    ]


def bypass_released(
    robot_pose: list[Any],
    obstacle_pose: list[Any],
    waypoint_pose: list[Any],
    side: float,
    *,
    release_after_x_m: float,
    clearance_y_m: float,
    reached_m: float,
) -> bool:
    robot_x, robot_y = float(robot_pose[0]), float(robot_pose[1])
    ox, oy = float(obstacle_pose[0]), float(obstacle_pose[1])
    waypoint_distance = math.hypot(float(waypoint_pose[0]) - robot_x, float(waypoint_pose[1]) - robot_y)
    return bool(
        waypoint_distance <= float(reached_m)
        or (
            robot_x >= ox + float(release_after_x_m)
            and float(side) * (robot_y - oy) >= float(clearance_y_m)
        )
    )


def branch_pre_entry_reached(
    robot_pose: list[Any],
    pre_entry_pose: list[Any],
    *,
    reached_m: float,
    lateral_tolerance_m: float,
) -> bool:
    return bool(
        distance_xy(robot_pose, pre_entry_pose) <= float(reached_m)
        or (
            float(robot_pose[0]) >= float(pre_entry_pose[0])
            and abs(float(robot_pose[1]) - float(pre_entry_pose[1])) <= float(lateral_tolerance_m)
        )
    )


def branch_verify_ready(*, entered_correct: bool, target_visible_required: bool, target_visible: bool) -> bool:
    return bool(entered_correct and (target_visible or not target_visible_required))


def rgb_color(name: str | None) -> tuple[int, int, int]:
    table = {
        "red": (220, 35, 35),
        "blue": (35, 95, 220),
        "white": (230, 230, 220),
        "brown": (145, 95, 55),
        "gray": (140, 140, 140),
        "grey": (140, 140, 140),
    }
    return table.get(str(name or "").lower(), (70, 190, 120))


def draw_rect(buf: bytearray, width: int, height: int, cx: int, cy: int, rw: int, rh: int, color: tuple[int, int, int]) -> None:
    x0 = max(0, cx - rw // 2)
    x1 = min(width - 1, cx + rw // 2)
    y0 = max(0, cy - rh // 2)
    y1 = min(height - 1, cy + rh // 2)
    for y in range(y0, y1 + 1):
        row = y * width * 3
        for x in range(x0, x1 + 1):
            idx = row + x * 3
            buf[idx : idx + 3] = bytes(color)


def draw_semantic_object(
    buf: bytearray,
    width: int,
    height: int,
    *,
    cx: int,
    cy: int,
    size: int,
    obj: dict[str, Any],
) -> None:
    color = rgb_color(obj.get("color"))
    object_class = str(obj.get("class") or "").lower()
    if object_class == "fire_extinguisher":
        body_w = max(8, int(size * 0.72))
        body_h = max(12, int(size * 1.55))
        draw_rect(buf, width, height, cx, cy, body_w, body_h, color)
        draw_rect(buf, width, height, cx, cy - body_h // 2, max(5, body_w // 2), max(3, size // 5), (25, 25, 25))
        draw_rect(buf, width, height, cx + body_w // 2, cy - body_h // 2, max(4, size // 3), max(2, size // 8), (25, 25, 25))
        draw_rect(buf, width, height, cx, cy, max(4, body_w // 2), max(4, body_h // 4), (235, 235, 225))
        return
    draw_rect(buf, width, height, cx, cy, size, max(8, int(size * 0.75)), color)


def synthesize_front_rgb(
    pose: list[float],
    objects: list[dict[str, Any]],
    obstacles: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    fov_deg: float,
    max_range_m: float,
) -> bytes:
    buf = bytearray([38, 42, 46] * width * height)
    horizon = int(height * 0.55)
    for y in range(horizon, height):
        shade = int(48 + 48 * (y - horizon) / max(1, height - horizon))
        row = y * width * 3
        for x in range(width):
            idx = row + x * 3
            buf[idx : idx + 3] = bytes((shade, shade, shade))
    for obj in objects:
        if not object_visible(pose, obj, fov_deg=fov_deg, max_range_m=max_range_m, blockers=obstacles):
            continue
        obj_pose = obj.get("pose", [0.0, 0.0, 0.0])
        brg = bearing_deg(pose, obj_pose)
        dist = max(0.25, distance_xy(pose, obj_pose))
        cx = int(width * 0.5 + (brg / (fov_deg * 0.5)) * width * 0.45)
        cy = int(height * 0.52 - min(70.0, 22.0 / dist))
        size = max(8, int(42.0 / dist))
        draw_semantic_object(buf, width, height, cx=cx, cy=cy, size=size, obj=obj)
    return bytes(buf)


def find_scene_by_id(scenes_doc: dict[str, Any], scene_id: str) -> dict[str, Any] | None:
    for scene in scenes_doc.get("scenes", []):
        if scene.get("scene_id") == scene_id:
            return scene
    return None


def find_task_by_id(tasks_doc: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    for task in tasks_doc.get("tasks", []):
        if task.get("task_id") == task_id:
            return task
    return None


def main() -> None:
    import rclpy
    from geometry_msgs.msg import TransformStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String
    from tf2_ros import TransformBroadcaster

    class Go2BenchmarkAdapter(Node):
        def __init__(self) -> None:
            super().__init__("go2_benchmark_adapter")
            self.declare_parameter("tasks_path", "configs/tasks.yaml")
            self.declare_parameter("scenes_path", "configs/scenes.yaml")
            self.declare_parameter("task_id", "")
            self.declare_parameter("scene_id", "")
            self.declare_parameter("safe_cmd_topic", "/safe_cmd_vel")
            self.declare_parameter("isaac_cmd_topic", "/go2/cmd_vel")
            self.declare_parameter("publish_rate_hz", 20.0)
            self.declare_parameter("camera_rate_hz", 5.0)
            self.declare_parameter("status_rate_hz", 5.0)
            self.declare_parameter("camera_width", 320)
            self.declare_parameter("camera_height", 240)
            self.declare_parameter("front_fov_deg", 90.0)
            self.declare_parameter("max_visible_range_m", 12.0)
            self.declare_parameter("command_timeout_sec", 0.6)
            self.declare_parameter("forward_to_isaac", True)
            self.declare_parameter("publish_isaac_cmd_topic", False)
            self.declare_parameter("publish_tf", True)
            self.declare_parameter("local_stop_distance_m", 0.5)
            self.declare_parameter("local_path_half_width_m", 0.15)
            self.declare_parameter("fallen_root_z_m", 0.18)
            self.declare_parameter("isaac_pose_udp_host", "127.0.0.1")
            self.declare_parameter("isaac_pose_udp_port", 15010)
            self.declare_parameter("isaac_pose_timeout_sec", 1.0)
            self.declare_parameter("isaac_heading_sign", 1.0)
            self.declare_parameter("isaac_angular_command_sign", 1.0)
            self.declare_parameter("isaac_control_udp_host", "127.0.0.1")
            self.declare_parameter("isaac_control_udp_port", 15011)
            self.declare_parameter("isaac_twist_udp_host", "127.0.0.1")
            self.declare_parameter("isaac_twist_udp_port", 15002)
            self.declare_parameter("isaac_camera_udp_host", "127.0.0.1")
            self.declare_parameter("isaac_camera_udp_port", 15012)
            self.declare_parameter("isaac_camera_timeout_sec", 1.0)
            self.declare_parameter("publish_synthetic_camera_fallback", True)
            self.declare_parameter("prefer_synthetic_camera_for_primary", False)
            self.declare_parameter("isaac_image_topic", "/camera/front/isaac_image")
            self.declare_parameter("camera_info_topic", "/camera/front/camera_info")
            self.declare_parameter("camera_focal_length_mm", 24.0)
            self.declare_parameter("camera_horizontal_aperture_mm", 20.955)

            self.tasks_doc = load_data(self.get_parameter("tasks_path").value)
            self.scenes_doc = load_data(self.get_parameter("scenes_path").value)
            self.task = self._initial_task()
            self.scene = self._initial_scene()
            self.pose = self._start_pose(self.scene)
            self.active_mode_name = ""
            self.active_mode_config: dict[str, Any] = {}
            self.active_episode_id = ""
            self.semantic_navigation_status: dict[str, Any] | None = None
            self.oracle_route_publish_count = 0
            self.oracle_last_route_publish = 0.0
            self.oracle_stop_publish_count = 0
            self.oracle_last_stop_publish = 0.0
            self.oracle_route_near_event_sent = False
            self.oracle_stop_visible_event_sent = False
            self.v9_last_assist_publish = 0.0
            self.v9_assist_publish_count = 0
            self.v9_route_injection_published = False
            self.v9_route_trigger_reached_sent = False
            self.v9_semantic_visible_since: float | None = None
            self.v9_semantic_threshold_reached_sent = False
            self.v9_progress_watchdog_reasons: set[str] = set()
            self.v10_phase = ""
            self.v10_phase_started = 0.0
            self.v10_phase_start_pose = self.pose.as_list()
            self.v10_decision_sent = False
            self.v10_done = False
            self.v10_stop_sent = False
            self.v10_coverage_sent = False
            self.v10_target_orienting = True
            self.v10_last_primitive_publish = 0.0
            self.v10_trace_count = 0
            self.v10_controller_enabled = False
            self.v10_step_route_choice = ""
            self.v10_step_stop_accepted = False
            self.v10_bypass_side: float | None = None
            self.v10_bypass_complete = False
            self.v10_target_yaw_polarity = 1.0
            self.v10_target_yaw_mismatch_count = 0
            self.v10_target_last_yaw = self.pose.yaw
            self.v10_target_last_desired_angular = 0.0
            self.simple_goal_stop_latched = False
            self.cmd = Twist()
            self.last_cmd_rx = 0.0
            self.last_update = time.monotonic()
            self.episode_started = time.monotonic()
            self.collision_sent = False
            self.judge = SuccessJudgeCore()

            self.forward_to_isaac = bool(self.get_parameter("forward_to_isaac").value)
            self.publish_isaac_cmd_topic = bool(self.get_parameter("publish_isaac_cmd_topic").value)
            self.publish_tf = bool(self.get_parameter("publish_tf").value)
            self.width = int(self.get_parameter("camera_width").value)
            self.height = int(self.get_parameter("camera_height").value)
            self.fov_deg = float(self.get_parameter("front_fov_deg").value)
            self.max_visible_range = float(self.get_parameter("max_visible_range_m").value)
            self.command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)
            self.local_stop_distance_m = float(self.get_parameter("local_stop_distance_m").value)
            self.local_path_half_width_m = float(self.get_parameter("local_path_half_width_m").value)
            self.fallen_root_z_m = float(self.get_parameter("fallen_root_z_m").value)
            self.pose_source = "integrated_safe_cmd"
            self.telemetry_seq = None
            self.telemetry_timestamp = None
            self.telemetry_event = ""
            self.telemetry_episode_id = ""
            self.telemetry_task_id = ""
            self.telemetry_scene_id = ""
            self.telemetry_z = None
            self.telemetry_command: dict[str, Any] = {}
            self.telemetry_linear_velocity: list[float] = []
            self.telemetry_angular_velocity: list[float] = []
            self.last_isaac_pose_rx = 0.0
            self.isaac_pose_timeout_sec = float(self.get_parameter("isaac_pose_timeout_sec").value)
            self.isaac_heading_sign = float(self.get_parameter("isaac_heading_sign").value)
            self.isaac_angular_command_sign = float(self.get_parameter("isaac_angular_command_sign").value)
            self.telemetry_sock = self.make_telemetry_socket()
            self.control_sock = self.make_control_socket()
            self.twist_sock = self.make_twist_socket()
            self.twist_seq = 0
            self.camera_sock = self.make_camera_socket()
            self.camera_chunks: dict[tuple[int, str], dict[str, Any]] = {}
            self.camera_rgb_frame: dict[str, Any] | None = None
            self.camera_depth_frame: dict[str, Any] | None = None
            self.last_isaac_camera_rx = 0.0
            self.camera_seq = None
            self.camera_rx_counts = {"rgb8": 0, "depth16": 0}
            self.camera_pub_counts = {"rgb8": 0, "isaac_rgb8": 0, "depth16": 0}
            self.last_camera_source = ""
            self.isaac_camera_timeout_sec = float(self.get_parameter("isaac_camera_timeout_sec").value)
            self.synthetic_camera_fallback = bool(self.get_parameter("publish_synthetic_camera_fallback").value)

            self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
            self.image_pub = self.create_publisher(Image, "/camera/front/image", 2)
            self.isaac_image_pub = self.create_publisher(
                Image,
                str(self.get_parameter("isaac_image_topic").value),
                2,
            )
            self.depth_pub = self.create_publisher(Image, "/camera/front/depth", 2)
            self.camera_info_pub = self.create_publisher(
                CameraInfo,
                str(self.get_parameter("camera_info_topic").value),
                2,
            )
            self.pose_pub = self.create_publisher(String, "/isaac/ground_truth_pose", 10)
            self.reset_ack_pub = self.create_publisher(String, "/isaac/reset_ack_json", 10)
            self.objects_pub = self.create_publisher(String, "/isaac/objects", 10)
            self.status_pub = self.create_publisher(String, "/isaac/episode_status", 10)
            self.collision_pub = self.create_publisher(String, "/isaac/collision_event", 10)
            self.oracle_route_pub = self.create_publisher(String, "/oracle/route_choice_json", 10)
            self.oracle_stop_pub = self.create_publisher(String, "/oracle/semantic_stop_json", 10)
            self.oracle_override_pub = self.create_publisher(String, "/oracle/control_override_json", 10)
            self.primitive_pub = self.create_publisher(String, "/primitive/command_json", 10)
            self.mission_event_pub = self.create_publisher(String, "/mission/event_json", 10)
            self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
            self.robot_state_pub = self.create_publisher(String, "/robot_state_json", 10)
            self.safety_pub = self.create_publisher(String, "/safety/local_status_json", 10)
            self.cmd_pub = self.create_publisher(Twist, str(self.get_parameter("isaac_cmd_topic").value), 10)
            self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

            self.cmd_sub = self.create_subscription(
                Twist,
                str(self.get_parameter("safe_cmd_topic").value),
                self.on_safe_cmd,
                10,
            )
            self.reset_sub = self.create_subscription(String, "/isaac/reset_episode", self.on_reset, 10)
            self.scene_sub = self.create_subscription(String, "/isaac/set_scene", self.on_set_scene, 10)
            self.mode_sub = self.create_subscription(String, "/benchmark/mode_json", self.on_benchmark_mode, 10)
            self.instruction_sub = self.create_subscription(String, "/user_instruction", self.on_instruction, 10)
            self.step_route_sub = self.create_subscription(
                String, "/step/route_choice_json", self.on_step_route_choice, 10
            )
            self.step_stop_sub = self.create_subscription(
                String, "/step/semantic_stop_json", self.on_step_semantic_stop, 10
            )
            self.step_stop_release_sub = self.create_subscription(
                String, "/scheduler/semantic_stop_release_json", self.on_step_semantic_stop, 10
            )
            self.semantic_status_sub = self.create_subscription(
                String, "/semantic_navigation/status_json", self.on_semantic_navigation_status, 10
            )

            publish_period = 1.0 / max(1.0, float(self.get_parameter("publish_rate_hz").value))
            camera_period = 1.0 / max(0.2, float(self.get_parameter("camera_rate_hz").value))
            status_period = 1.0 / max(0.2, float(self.get_parameter("status_rate_hz").value))
            self.create_timer(publish_period, self.tick_motion)
            self.create_timer(camera_period, self.publish_camera)
            self.create_timer(status_period, self.publish_status_and_oracles)
            self.get_logger().info(
                f"Go2 benchmark adapter ready task={self.task.get('task_id')} scene={self.scene.get('scene_id')} "
                f"safe_cmd={self.get_parameter('safe_cmd_topic').value} isaac_cmd={self.get_parameter('isaac_cmd_topic').value}"
            )

        def _initial_task(self) -> dict[str, Any]:
            wanted = str(self.get_parameter("task_id").value)
            if wanted:
                task = find_task_by_id(self.tasks_doc, wanted)
                if task:
                    return task
                self.get_logger().warning(f"task_id {wanted!r} not found; using first task")
            return self.tasks_doc.get("tasks", [])[0]

        def _initial_scene(self) -> dict[str, Any]:
            wanted = str(self.get_parameter("scene_id").value)
            if wanted:
                scene = find_scene_by_id(self.scenes_doc, wanted)
                if scene:
                    return normalize_scene_to_robot_origin(scene)
                self.get_logger().warning(f"scene_id {wanted!r} not found; using task scene_type")
            return normalize_scene_to_robot_origin(scene_by_type(self.scenes_doc, self.task["scene_type"]))

        @staticmethod
        def _start_pose(scene: dict[str, Any]) -> PlanarPose:
            start = scene.get("robot_start_pose", [0.0, 0.0, 0.0])
            return PlanarPose(float(start[0]), float(start[1]), float(start[2]))

        def on_safe_cmd(self, msg: Twist) -> None:
            self.cmd = msg
            self.last_cmd_rx = time.monotonic()

        def on_set_scene(self, msg: String) -> None:
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError as exc:
                self.get_logger().warning(f"bad /isaac/set_scene JSON: {exc}")
                return
            if isinstance(data, dict) and data.get("scene_id"):
                self.scene = normalize_scene_to_robot_origin(data)
                self.pose = self._start_pose(self.scene)
                self.episode_started = time.monotonic()
                self.judge = SuccessJudgeCore()
                self.collision_sent = False
                self.get_logger().info(f"set scene {self.scene.get('scene_id')}")

        def on_reset(self, msg: String) -> None:
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                data = {}
            task_id = str(data.get("task_id", ""))
            scene_id = str(data.get("scene_id", ""))
            if task_id:
                task = find_task_by_id(self.tasks_doc, task_id)
                if task:
                    self.task = task
            if scene_id:
                scene = find_scene_by_id(self.scenes_doc, scene_id)
                if scene:
                    self.scene = normalize_scene_to_robot_origin(scene)
            elif self.task:
                self.scene = normalize_scene_to_robot_origin(scene_by_type(self.scenes_doc, self.task["scene_type"]))
            overlay_only = bool(data.get("visual_overlay_only", False))
            if isinstance(data.get("objects"), list):
                visual_objects = [dict(value) for value in data["objects"] if isinstance(value, dict)]
                self.scene["objects"] = (
                    merge_scene_entities(list(self.scene.get("objects") or []), visual_objects)
                    if overlay_only
                    else visual_objects
                )
            if isinstance(data.get("obstacles"), list):
                visual_obstacles = [dict(value) for value in data["obstacles"] if isinstance(value, dict)]
                self.scene["obstacles"] = (
                    merge_scene_entities(list(self.scene.get("obstacles") or []), visual_obstacles)
                    if overlay_only
                    else visual_obstacles
                )
            if isinstance(data.get("pose"), list) and len(data["pose"]) >= 3:
                self.scene["robot_start_pose"] = [float(value) for value in data["pose"][:3]]
            self.pose = self._start_pose(self.scene)
            self.cmd = Twist()
            self.last_cmd_rx = 0.0
            self.last_update = time.monotonic()
            self.episode_started = time.monotonic()
            self.judge = SuccessJudgeCore()
            self.semantic_navigation_status = None
            self.collision_sent = False
            self.oracle_route_publish_count = 0
            self.oracle_last_route_publish = 0.0
            self.oracle_stop_publish_count = 0
            self.oracle_last_stop_publish = 0.0
            self.oracle_route_near_event_sent = False
            self.oracle_stop_visible_event_sent = False
            self.reset_v9_probe_state()
            self.last_camera_source = ""
            self.v10_controller_enabled = False
            if data.get("episode_id"):
                self.active_episode_id = str(data.get("episode_id"))
            self.get_logger().info(f"reset episode task={self.task.get('task_id')} scene={self.scene.get('scene_id')}")
            self.send_isaac_control(
                {
                    "event": "reset",
                    "task_id": self.task.get("task_id"),
                    "task_type": self.task.get("task_type"),
                    "scene_id": str(data.get("scene_id") or self.scene.get("scene_id")),
                    "episode_id": self.active_episode_id,
                    "pose": self.pose.as_list(),
                    "objects": [dict(obj) for obj in self.scene.get("objects", [])],
                    "obstacles": [dict(obj) for obj in self.scene.get("obstacles", [])],
                    "dynamic_obstacles": [dict(obj) for obj in self.scene.get("dynamic_obstacles", [])],
                    "random_seed": int(self.scene.get("random_seed", data.get("random_seed", 0)) or 0),
                    "timestamp": time.time(),
                    "visual_case_id": str(data.get("visual_case_id") or ""),
                    "visual_variant": str(data.get("visual_variant") or ""),
                    "brightness": data.get("brightness"),
                    "background_usd": str(data.get("background_usd") or ""),
                    "background_pose": list(data.get("background_pose") or [0.0, 0.0, 0.0]),
                    "background_scale": list(data.get("background_scale") or [1.0, 1.0, 1.0]),
                    "background_visual_only": bool(data.get("background_visual_only", True)),
                }
            )

        def on_benchmark_mode(self, msg: String) -> None:
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            if not isinstance(data, dict):
                return
            episode_id = str(data.get("episode_id") or "")
            if episode_id and episode_id != self.active_episode_id:
                self.active_episode_id = episode_id
                self.semantic_navigation_status = None
                self.oracle_route_publish_count = 0
                self.oracle_last_route_publish = 0.0
                self.oracle_stop_publish_count = 0
                self.oracle_last_stop_publish = 0.0
                self.oracle_route_near_event_sent = False
                self.oracle_stop_visible_event_sent = False
                self.reset_v9_probe_state()
                self.v10_controller_enabled = False
            mode_config = data.get("mode_config") if isinstance(data.get("mode_config"), dict) else {}
            self.active_mode_name = str(data.get("mode") or "")
            self.active_mode_config = dict(mode_config)

        def on_semantic_navigation_status(self, msg: String) -> None:
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            if not isinstance(data, dict):
                return
            episode_id = str(data.get("episode_id") or "")
            task_id = str(data.get("task_id") or "")
            if self.active_episode_id and episode_id and episode_id != self.active_episode_id:
                return
            if task_id and task_id != str(self.task.get("task_id") or ""):
                return
            self.semantic_navigation_status = dict(data)

        def on_instruction(self, msg: String) -> None:
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                data = {}
            mission_id = str(data.get("mission_id") or "") if isinstance(data, dict) else ""
            if mission_id and mission_id != self.active_episode_id:
                return
            if str(self.active_mode_config.get("v10_mode") or "") not in {
                "branch_entry_controller",
                "target_relative_approach",
                "geometric_watchdog",
            }:
                return
            self.reset_v10_controller_state()
            decision_source = str(self.active_mode_config.get("controller_decision_source") or "oracle")
            task_type = str(self.task.get("task_type") or "")
            route_relevant = task_type in {"turn_choice", "turn_microbench"} and bool(
                self.active_mode_config.get("route_controller_enabled", True)
            )
            target_relevant = task_type == "semantic_target" and bool(
                self.active_mode_config.get("target_controller_enabled", True)
            )
            self.v10_controller_enabled = bool(route_relevant or target_relevant)
            self.publish_metric(
                "v10_controller_enabled",
                episode_id=self.active_episode_id,
                task_id=self.task.get("task_id"),
                pose=self.pose.as_list(),
                controller_decision_source=decision_source,
                enabled=self.v10_controller_enabled,
            )

        def on_step_route_choice(self, msg: String) -> None:
            if str(self.active_mode_config.get("controller_decision_source") or "") != "step":
                return
            if not bool(self.active_mode_config.get("route_controller_enabled", True)):
                return
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            if not isinstance(data, dict) or self.step_payload_episode_mismatch(data):
                return
            route_choice = str(data.get("route_choice") or "").strip().lower()
            if route_choice not in {"left", "right"}:
                self.publish_metric(
                    "v10_step_route_choice_rejected",
                    route_choice=route_choice,
                    reason="unsupported_composite_branch",
                )
                return
            if self.v10_step_route_choice in {"left", "right"}:
                self.publish_metric(
                    "v10_step_route_choice_duplicate_ignored",
                    route_choice=route_choice,
                    latched_route_choice=self.v10_step_route_choice,
                    source="step",
                )
                return
            self.v10_step_route_choice = route_choice
            self.v10_controller_enabled = True
            self.v10_target_orienting = True
            self.publish_metric(
                "v10_step_route_choice_latched",
                route_choice=route_choice,
                source="step",
                preserved_phase=self.v10_phase,
                bypass_complete=self.v10_bypass_complete,
            )

        def on_step_semantic_stop(self, msg: String) -> None:
            sensor_only = bool(self.active_mode_config.get("sensor_only_planning", False))
            if sensor_only:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    return
                if not isinstance(data, dict) or self.step_payload_episode_mismatch(data):
                    return
                accepted = accepted_sensor_semantic_step_stop(data)
                self.publish_metric(
                    "sensor_step_semantic_stop_judge",
                    decision=data,
                    source="actual_sensor_fusion",
                    result="accepted" if accepted else "rejected",
                )
                if accepted:
                    self.v10_step_stop_accepted = True
                    self.v10_stop_sent = True
                return
            if str(self.active_mode_config.get("controller_decision_source") or "") != "step":
                return
            if not bool(self.active_mode_config.get("target_controller_enabled", True)):
                return
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            if not isinstance(data, dict) or self.step_payload_episode_mismatch(data):
                return
            target = object_by_id(self.scene, self.task["target_object"])
            actual_distance = distance_xy(self.pose.as_list(), target.get("pose", [0.0, 0.0, 0.0]))
            max_actual_distance = float(self.active_mode_config.get("target_stop_threshold_m", 2.0))
            accepted = accepted_semantic_step_stop(
                data,
                require_confirmed_track=bool(
                    self.active_mode_config.get("semantic_stop_require_confirmed_track", False)
                ),
                actual_distance_m=actual_distance,
                max_actual_distance_m=max_actual_distance,
            )
            if not accepted:
                if bool((data.get("track") or {}).get("confirmed")) and actual_distance > max_actual_distance:
                    self.v10_set_phase("approach", time.monotonic() - self.episode_started, self.pose.as_list())
                self.publish_metric(
                    "v10_step_semantic_stop_rejected",
                    decision=data,
                    source="step",
                    actual_distance_m=round(actual_distance, 4),
                    stop_threshold_m=max_actual_distance,
                )
                return
            self.v10_step_stop_accepted = True
            self.v10_stop_sent = True
            self.v10_set_phase("stop", time.monotonic() - self.episode_started, self.pose.as_list())
            self.publish_metric("v10_step_semantic_stop_latched", decision=data, source="step")

        def step_payload_episode_mismatch(self, payload: dict[str, Any]) -> bool:
            return not step_payload_matches_episode(payload, self.active_episode_id)

        def reset_v9_probe_state(self) -> None:
            self.v9_last_assist_publish = 0.0
            self.v9_assist_publish_count = 0
            self.v9_route_injection_published = False
            self.v9_route_trigger_reached_sent = False
            self.v9_semantic_visible_since = None
            self.v9_semantic_threshold_reached_sent = False
            self.v9_progress_watchdog_reasons = set()
            self.reset_v10_controller_state()

        def reset_v10_controller_state(self) -> None:
            self.v10_phase = ""
            self.v10_phase_started = 0.0
            self.v10_phase_start_pose = self.pose.as_list()
            self.v10_decision_sent = False
            self.v10_done = False
            self.v10_stop_sent = False
            self.v10_coverage_sent = False
            self.v10_target_orienting = True
            self.v10_last_primitive_publish = 0.0
            self.v10_trace_count = 0
            self.v10_step_route_choice = ""
            self.v10_step_stop_accepted = False
            self.v10_bypass_side = None
            self.v10_bypass_complete = False
            self.v10_target_yaw_polarity = 1.0
            self.v10_target_yaw_mismatch_count = 0
            self.v10_target_last_yaw = self.pose.yaw
            self.v10_target_last_desired_angular = 0.0
            self.simple_goal_stop_latched = False

        def active_cmd(self) -> Twist:
            if self.last_cmd_rx <= 0.0 or time.monotonic() - self.last_cmd_rx > self.command_timeout_sec:
                return Twist()
            return self.cmd

        def make_control_socket(self):
            host = str(self.get_parameter("isaac_control_udp_host").value)
            port = int(self.get_parameter("isaac_control_udp_port").value)
            if not host or port <= 0:
                return None
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.control_addr = (host, port)
            self.get_logger().info(f"Sending Isaac benchmark control to udp://{host}:{port}")
            return sock

        def make_twist_socket(self):
            host = str(self.get_parameter("isaac_twist_udp_host").value)
            port = int(self.get_parameter("isaac_twist_udp_port").value)
            if not host or port <= 0:
                return None
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.twist_addr = (host, port)
            self.get_logger().info(f"Sending Isaac Twist control to udp://{host}:{port}")
            return sock

        def send_isaac_control(self, payload: dict[str, Any]) -> None:
            if self.control_sock is None:
                return
            try:
                self.control_sock.sendto(
                    json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
                    self.control_addr,
                )
            except OSError as exc:
                self.get_logger().warning(f"failed to send Isaac control packet: {exc!r}")

        def send_isaac_twist(self, cmd: Twist) -> None:
            if self.twist_sock is None:
                return
            self.twist_seq += 1
            payload = isaac_twist_payload(
                cmd.linear.x,
                cmd.linear.y,
                self.isaac_angular_command_sign * cmd.angular.z,
                seq=self.twist_seq,
                timestamp=time.time(),
            )
            try:
                self.twist_sock.sendto(
                    json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
                    self.twist_addr,
                )
            except OSError as exc:
                self.get_logger().warning(f"failed to send Isaac Twist packet: {exc!r}")

        def make_telemetry_socket(self):
            host = str(self.get_parameter("isaac_pose_udp_host").value)
            port = int(self.get_parameter("isaac_pose_udp_port").value)
            if not host or port <= 0:
                return None
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            sock.bind((host, port))
            sock.setblocking(False)
            self.get_logger().info(f"Listening for Isaac Go2 root pose telemetry on udp://{host}:{port}")
            return sock

        def make_camera_socket(self):
            host = str(self.get_parameter("isaac_camera_udp_host").value)
            port = int(self.get_parameter("isaac_camera_udp_port").value)
            if not host or port <= 0:
                return None
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
            sock.bind((host, port))
            sock.setblocking(False)
            self.get_logger().info(f"Listening for Isaac front camera frames on udp://{host}:{port}")
            return sock

        def drain_isaac_pose_telemetry(self) -> None:
            if self.telemetry_sock is None:
                return
            while True:
                try:
                    payload, _addr = self.telemetry_sock.recvfrom(16384)
                except BlockingIOError:
                    return
                try:
                    data = json.loads(payload.decode("utf-8"))
                    pose = data.get("pose", [])
                    if len(pose) >= 3:
                        self.pose = normalize_isaac_pose(pose, self.isaac_heading_sign)
                        self.pose_source = str(data.get("source", "isaaclab_go2_udp"))
                        self.telemetry_seq = data.get("seq")
                        self.telemetry_timestamp = data.get("timestamp")
                        self.telemetry_event = str(data.get("event", "pose"))
                        self.telemetry_episode_id = str(data.get("episode_id", ""))
                        self.telemetry_task_id = str(data.get("task_id", ""))
                        self.telemetry_scene_id = str(data.get("scene_id", ""))
                        self.telemetry_z = data.get("z")
                        self.telemetry_command = dict(data.get("cmd") or {})
                        self.telemetry_linear_velocity = list(data.get("linear_velocity") or [])
                        self.telemetry_angular_velocity = list(data.get("angular_velocity") or [])
                        if len(self.telemetry_angular_velocity) >= 3:
                            self.telemetry_angular_velocity[2] = self.isaac_heading_sign * float(
                                self.telemetry_angular_velocity[2]
                            )
                        self.last_isaac_pose_rx = time.monotonic()
                        if (
                            self.telemetry_event == "reset"
                            and self.telemetry_episode_id
                        ):
                            self.reset_ack_pub.publish(
                                String(
                                    data=json.dumps(
                                        {
                                            "episode_id": self.telemetry_episode_id,
                                            "task_id": self.telemetry_task_id,
                                            "scene_id": self.telemetry_scene_id,
                                            "telemetry_seq": self.telemetry_seq,
                                            "telemetry_timestamp": self.telemetry_timestamp,
                                            "pose": self.pose.as_list(),
                                            "z": self.telemetry_z,
                                        }
                                    )
                                )
                            )
                except Exception as exc:
                    self.get_logger().warning(f"bad Isaac telemetry packet: {exc!r}")

        def have_fresh_isaac_pose(self) -> bool:
            return self.last_isaac_pose_rx > 0.0 and time.monotonic() - self.last_isaac_pose_rx <= self.isaac_pose_timeout_sec

        def drain_isaac_camera_telemetry(self) -> None:
            if self.camera_sock is None:
                return
            now = time.monotonic()
            while True:
                try:
                    packet, _addr = self.camera_sock.recvfrom(65535)
                except BlockingIOError:
                    break
                try:
                    header_bytes, sep, chunk = packet.partition(b"\n")
                    if not sep:
                        continue
                    header = json.loads(header_bytes.decode("utf-8"))
                    if header.get("magic") != "isaac_cam_v1":
                        continue
                    seq = int(header["seq"])
                    kind = str(header["kind"])
                    chunk_index = int(header["chunk"])
                    chunk_count = int(header["chunks"])
                    if chunk_count <= 0 or chunk_index < 0 or chunk_index >= chunk_count:
                        continue
                    key = (seq, kind)
                    state = self.camera_chunks.get(key)
                    if state is None:
                        state = {
                            "header": header,
                            "created": now,
                            "parts": [None] * chunk_count,
                        }
                        self.camera_chunks[key] = state
                    if len(state["parts"]) != chunk_count:
                        continue
                    state["parts"][chunk_index] = chunk
                    if all(part is not None for part in state["parts"]):
                        raw = b"".join(state["parts"])
                        if str(state["header"].get("compression") or "") == "zlib":
                            raw = zlib.decompress(raw)
                            expected_bytes = int(state["header"].get("raw_bytes", len(raw)))
                            if len(raw) != expected_bytes:
                                raise ValueError(
                                    f"camera frame size {len(raw)} != expected {expected_bytes}"
                                )
                        self.store_camera_frame(state["header"], raw)
                        self.camera_chunks.pop(key, None)
                except Exception as exc:
                    self.get_logger().warning(f"bad Isaac camera packet: {exc!r}")
            for key, state in list(self.camera_chunks.items()):
                if now - float(state.get("created", now)) > 1.0:
                    self.camera_chunks.pop(key, None)

        def store_camera_frame(self, header: dict[str, Any], raw: bytes) -> None:
            width = int(header.get("width", 0))
            height = int(header.get("height", 0))
            kind = str(header.get("kind", ""))
            if width <= 0 or height <= 0:
                return
            frame = {
                "width": width,
                "height": height,
                "encoding": str(header.get("encoding", "")),
                "frame_id": str(header.get("frame_id", "isaac_front_camera")),
                "seq": header.get("seq"),
                "timestamp": header.get("timestamp"),
                "received": time.monotonic(),
                "focal_length_mm": float(header.get("focal_length_mm", self.get_parameter("camera_focal_length_mm").value)),
                "horizontal_aperture_mm": float(
                    header.get(
                        "horizontal_aperture_mm",
                        self.get_parameter("camera_horizontal_aperture_mm").value,
                    )
                ),
                "camera_translation_base_m": list(header.get("camera_translation_base_m") or [0.35, 0.0, 0.20]),
                "camera_rotation_base_xyzw": list(
                    header.get("camera_rotation_base_xyzw") or [-0.5, 0.5, -0.5, 0.5]
                ),
            }
            if kind == "rgb8":
                expected = width * height * 3
                if len(raw) < expected:
                    return
                frame["encoding"] = "rgb8"
                frame["step"] = width * 3
                frame["data"] = bytes(raw[:expected])
                self.camera_rgb_frame = frame
                self.camera_rx_counts["rgb8"] += 1
            elif kind == "depth16":
                expected = width * height * 2
                if len(raw) < expected:
                    return
                frame["encoding"] = "16UC1"
                frame["step"] = width * 2
                frame["data"] = bytes(raw[:expected])
                self.camera_depth_frame = frame
                self.camera_rx_counts["depth16"] += 1
            else:
                return
            self.last_isaac_camera_rx = time.monotonic()
            self.camera_seq = header.get("seq")

        def have_fresh_camera_frame(self, frame: dict[str, Any] | None) -> bool:
            return frame is not None and time.monotonic() - float(frame.get("received", 0.0)) <= self.isaac_camera_timeout_sec

        def tick_motion(self) -> None:
            now = time.monotonic()
            dt = min(0.2, max(0.0, now - self.last_update))
            self.last_update = now
            self.drain_isaac_pose_telemetry()
            self.drain_isaac_camera_telemetry()
            cmd = self.active_cmd()
            if not self.have_fresh_isaac_pose():
                self.pose_source = "integrated_safe_cmd"
                self.pose.integrate(cmd.linear.x, cmd.angular.z, dt)
            if self.forward_to_isaac:
                if self.publish_isaac_cmd_topic:
                    self.cmd_pub.publish(cmd)
                self.send_isaac_twist(cmd)
            self.publish_odom(cmd)
            if self.tf_broadcaster is not None:
                self.publish_tf_msg()

        def publish_odom(self, cmd: Twist) -> None:
            stamp = self.get_clock().now().to_msg()
            msg = Odometry()
            msg.header.stamp = stamp
            msg.header.frame_id = "odom"
            msg.child_frame_id = "base_link"
            msg.pose.pose.position.x = self.pose.x
            msg.pose.pose.position.y = self.pose.y
            msg.pose.pose.position.z = 0.0
            qx, qy, qz, qw = yaw_to_quaternion(self.pose.yaw)
            msg.pose.pose.orientation.x = qx
            msg.pose.pose.orientation.y = qy
            msg.pose.pose.orientation.z = qz
            msg.pose.pose.orientation.w = qw
            msg.twist.twist.linear.x = cmd.linear.x
            msg.twist.twist.angular.z = cmd.angular.z
            self.odom_pub.publish(msg)

        def publish_tf_msg(self) -> None:
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "odom"
            transform.child_frame_id = "base_link"
            transform.transform.translation.x = self.pose.x
            transform.transform.translation.y = self.pose.y
            transform.transform.translation.z = 0.0
            qx, qy, qz, qw = yaw_to_quaternion(self.pose.yaw)
            transform.transform.rotation.x = qx
            transform.transform.rotation.y = qy
            transform.transform.rotation.z = qz
            transform.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(transform)
            camera = TransformStamped()
            camera.header.stamp = transform.header.stamp
            camera.header.frame_id = "base_link"
            camera.child_frame_id = "isaac_front_camera"
            camera.transform.translation.x = 0.35
            camera.transform.translation.y = 0.0
            camera.transform.translation.z = 0.20
            camera.transform.rotation.x = -0.5
            camera.transform.rotation.y = 0.5
            camera.transform.rotation.z = -0.5
            camera.transform.rotation.w = 0.5
            self.tf_broadcaster.sendTransform(camera)

        def publish_camera(self) -> None:
            self.drain_isaac_camera_telemetry()
            isaac_rgb_frame = self.camera_rgb_frame if self.have_fresh_camera_frame(self.camera_rgb_frame) else None
            source_stamp = self.get_clock().now().to_msg()
            if isaac_rgb_frame is not None and isaac_rgb_frame.get("timestamp") is not None:
                timestamp = float(isaac_rgb_frame["timestamp"])
                source_stamp.sec = int(timestamp)
                source_stamp.nanosec = int((timestamp % 1.0) * 1.0e9)
            if isaac_rgb_frame is not None:
                self.isaac_image_pub.publish(self.image_message(isaac_rgb_frame, stamp=source_stamp))
                self.camera_pub_counts["isaac_rgb8"] += 1
            prefer_synthetic = bool(self.get_parameter("prefer_synthetic_camera_for_primary").value)
            rgb_frame = None if prefer_synthetic else isaac_rgb_frame
            if rgb_frame is None:
                if not self.synthetic_camera_fallback:
                    return
                objects = self.current_objects()
                obstacles = self.current_obstacles()
                data = synthesize_front_rgb(
                    self.pose.as_list(),
                    objects,
                    obstacles,
                    width=self.width,
                    height=self.height,
                    fov_deg=self.fov_deg,
                    max_range_m=self.max_visible_range,
                )
                rgb_frame = {
                    "width": self.width,
                    "height": self.height,
                    "encoding": "rgb8",
                    "step": self.width * 3,
                    "frame_id": "front_camera",
                    "data": data,
                }
            camera_source = (
                "real_isaac_render_product_udp_15012"
                if isaac_rgb_frame is not None and not prefer_synthetic
                else "synthetic_adapter_camera"
            )
            if camera_source != self.last_camera_source:
                self.last_camera_source = camera_source
                self.publish_metric(
                    "camera_frame_source",
                    source=camera_source,
                    width=int(rgb_frame.get("width", 0) or 0),
                    height=int(rgb_frame.get("height", 0) or 0),
                    synthetic_fallback=camera_source != "real_isaac_render_product_udp_15012",
                )
            msg = self.image_message(rgb_frame, stamp=source_stamp)
            self.image_pub.publish(msg)
            self.camera_pub_counts["rgb8"] += 1

            depth_frame = self.camera_depth_frame if self.have_fresh_camera_frame(self.camera_depth_frame) else None
            if (
                depth_frame is not None
                and isaac_rgb_frame is not None
                and depth_frame.get("seq") != isaac_rgb_frame.get("seq")
            ):
                depth_frame = None
            if depth_frame is not None:
                depth_msg = Image()
                depth_msg.header.stamp = msg.header.stamp
                depth_msg.header.frame_id = str(depth_frame.get("frame_id", "isaac_front_camera"))
                depth_msg.height = int(depth_frame["height"])
                depth_msg.width = int(depth_frame["width"])
                depth_msg.encoding = str(depth_frame["encoding"])
                depth_msg.is_bigendian = 0
                depth_msg.step = int(depth_frame["step"])
                depth_msg.data = bytes(depth_frame["data"])
                self.depth_pub.publish(depth_msg)
                self.camera_pub_counts["depth16"] += 1
            if isaac_rgb_frame is not None:
                self.camera_info_pub.publish(self.camera_info_message(isaac_rgb_frame, stamp=source_stamp))

        def image_message(self, frame: dict[str, Any], *, stamp=None) -> Image:
            msg = Image()
            msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
            msg.header.frame_id = str(frame.get("frame_id", "front_camera"))
            msg.height = int(frame["height"])
            msg.width = int(frame["width"])
            msg.encoding = str(frame["encoding"])
            msg.is_bigendian = 0
            msg.step = int(frame["step"])
            msg.data = bytes(frame["data"])
            return msg

        def camera_info_message(self, frame: dict[str, Any], *, stamp=None) -> CameraInfo:
            msg = CameraInfo()
            msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
            msg.header.frame_id = str(frame.get("frame_id", "isaac_front_camera"))
            msg.width = int(frame["width"])
            msg.height = int(frame["height"])
            focal = float(frame.get("focal_length_mm", self.get_parameter("camera_focal_length_mm").value))
            aperture = float(
                frame.get(
                    "horizontal_aperture_mm",
                    self.get_parameter("camera_horizontal_aperture_mm").value,
                )
            )
            fx = msg.width * focal / aperture
            fy = fx
            cx = (msg.width - 1.0) * 0.5
            cy = (msg.height - 1.0) * 0.5
            msg.distortion_model = "plumb_bob"
            msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            return msg

        def publish_status_and_oracles(self) -> None:
            now_ros = self.get_clock().now().nanoseconds / 1e9
            elapsed = time.monotonic() - self.episode_started
            objects = self.current_objects()
            obstacles = self.current_obstacles()
            pose = self.pose.as_list()
            pose_payload = {
                "timestamp": now_ros,
                "scene_id": self.scene.get("scene_id"),
                "task_id": self.task.get("task_id"),
                "pose": pose,
                "robot_pose": pose,
                "source": self.pose_source,
                "telemetry_seq": self.telemetry_seq,
                "telemetry_timestamp": self.telemetry_timestamp,
                "isaac_heading_sign": self.isaac_heading_sign,
                "isaac_angular_command_sign": self.isaac_angular_command_sign,
                "isaac_transport": "udp_direct" if not self.publish_isaac_cmd_topic else "udp_direct_and_ros_topic",
                "telemetry_event": self.telemetry_event,
                "telemetry_episode_id": self.telemetry_episode_id,
                "telemetry_task_id": self.telemetry_task_id,
                "telemetry_scene_id": self.telemetry_scene_id,
                "z": self.telemetry_z,
                "command": self.telemetry_command,
                "linear_velocity": self.telemetry_linear_velocity,
                "angular_velocity": self.telemetry_angular_velocity,
                "telemetry_age_sec": None
                if self.last_isaac_pose_rx <= 0.0
                else round(time.monotonic() - self.last_isaac_pose_rx, 3),
            }
            objects_payload = {
                "timestamp": now_ros,
                "scene_id": self.scene.get("scene_id"),
                "objects": objects,
                "obstacles": obstacles,
                "dynamic_obstacles": self.current_dynamic_obstacles(),
                "semantic_zones": self.scene.get("semantic_zones", []),
            }
            camera_age = None if self.last_isaac_camera_rx <= 0.0 else round(time.monotonic() - self.last_isaac_camera_rx, 3)
            self.pose_pub.publish(String(data=json.dumps(pose_payload)))
            self.objects_pub.publish(String(data=json.dumps(objects_payload)))
            self.robot_state_pub.publish(
                String(
                    data=json.dumps(
                        {
                            "timestamp": now_ros,
                            "pose": pose,
                            "velocity": self.telemetry_linear_velocity,
                            "angular_velocity": self.telemetry_angular_velocity,
                            "z": self.telemetry_z,
                            "camera": {
                                "source": "isaac_render_product" if camera_age is not None else "synthetic_or_unavailable",
                                "seq": self.camera_seq,
                                "age_sec": camera_age,
                                "rx_counts": self.camera_rx_counts,
                                "pub_counts": self.camera_pub_counts,
                            },
                        }
                    )
                )
            )
            safety = self.safety_status(obstacles)
            self.safety_pub.publish(String(data=json.dumps(safety)))
            if str(self.task.get("task_type") or "") == "semantic_navigation":
                status = dict(
                    self.semantic_navigation_status
                    or {
                        "success": False,
                        "clean": False,
                        "done": False,
                        "reason": "waiting_for_semantic_marker_judge",
                        "qualification_evidence": False,
                    }
                )
                status.update(
                    {
                        "task_id": self.task.get("task_id"),
                        "task_type": "semantic_navigation",
                        "scene_id": self.scene.get("scene_id"),
                        "episode_id": self.active_episode_id,
                        "collision": max(int(bool(status.get("collision", 0))), int(bool(safety.get("collision", False)))),
                        "instrumented_semantic_markers": True,
                        "geometry_judge_is_oracle_only": True,
                        "qualification_evidence": False,
                    }
                )
                if status["collision"]:
                    status.update({"success": False, "clean": False, "done": True, "reason": "collision"})
                if elapsed > float(self.task.get("timeout_sec", 120)) and not status.get("done"):
                    status.update(
                        {"success": False, "clean": False, "done": True, "reason": "timeout", "time_sec": round(elapsed, 3)}
                    )
                self.status_pub.publish(String(data=json.dumps(status)))
                if safety.get("collision") and not self.collision_sent:
                    self.collision_sent = True
                    self.collision_pub.publish(
                        String(data=json.dumps({"timestamp": now_ros, "event": "collision", "pose": pose}))
                    )
                return
            try:
                target = object_by_id(self.scene, self.task["target_object"])
            except KeyError as exc:
                status = {
                    "task_id": self.task.get("task_id"),
                    "scene_id": self.scene.get("scene_id"),
                    "episode_id": self.active_episode_id,
                    "success": False,
                    "done": False,
                    "reason": "configuration_mismatch",
                    "configuration_error": str(exc),
                }
                self.status_pub.publish(String(data=json.dumps(status)))
                return
            status = self.judge.evaluate(
                self.task,
                self.scene,
                pose,
                target,
                now_sec=elapsed,
                cmd_vel={"linear_x": self.active_cmd().linear.x, "angular_z": self.active_cmd().angular.z},
                collision=bool(safety.get("collision", False)),
                fatal_failsafe=bool(safety.get("robot_fallen_or_unstable", False)),
            )
            if (
                str(self.task.get("task_type") or "") == "semantic_target"
                and bool(self.active_mode_config.get("semantic_stop_requires_step", False))
                and not self.v10_step_stop_accepted
                and bool(status.get("success", False))
            ):
                status.update({"success": False, "done": False, "reason": "never_stopped"})
            status.update(
                {
                    "task_id": self.task.get("task_id"),
                    "task_type": self.task.get("task_type"),
                    "scene_id": self.scene.get("scene_id"),
                    "episode_id": self.active_episode_id,
                }
            )
            if elapsed > float(self.task.get("timeout_sec", 120)) and not status.get("done"):
                status.update({"success": False, "done": True, "reason": "timeout", "time_sec": round(elapsed, 3)})
            self.status_pub.publish(String(data=json.dumps(status)))
            self.publish_simple_goal_stop(status=status, elapsed=elapsed)
            self.publish_v10_controller_actions(status=status, safety=safety, pose=pose, target=target, elapsed=elapsed)
            self.publish_v9_probe_actions(status=status, safety=safety, pose=pose, target=target, elapsed=elapsed)
            self.publish_v7_oracles(status=status, safety=safety, pose=pose, target=target, elapsed=elapsed)
            if safety.get("collision") and not self.collision_sent:
                self.collision_sent = True
                self.collision_pub.publish(
                    String(data=json.dumps({"timestamp": now_ros, "event": "collision", "pose": pose}))
                )

        def publish_v10_controller_actions(
            self,
            *,
            status: dict[str, Any],
            safety: dict[str, Any],
            pose: list[float],
            target: dict[str, Any],
            elapsed: float,
        ) -> None:
            if not self.v10_controller_enabled:
                return
            mode = self.active_mode_config
            v10_mode = str(mode.get("v10_mode") or "")
            task_type = str(self.task.get("task_type") or "")
            relevant = (
                task_type in {"turn_choice", "turn_microbench"}
                and v10_mode in {"branch_entry_controller", "geometric_watchdog"}
                and bool(mode.get("route_controller_enabled", True))
            ) or (
                task_type == "semantic_target"
                and v10_mode in {"target_relative_approach", "geometric_watchdog"}
                and bool(mode.get("target_controller_enabled", True))
            )
            if not v10_mode or not relevant:
                return
            if bool(status.get("done")):
                self.publish_oracle_override(False, phase="done")
                return
            self.publish_oracle_override(not self.v10_done, phase=self.v10_phase or "starting")
            if task_type in {"turn_choice", "turn_microbench"} and v10_mode in {"branch_entry_controller", "geometric_watchdog"}:
                self.publish_v10_branch_controller(safety=safety, pose=pose, target=target, elapsed=elapsed)
                if self.v10_done:
                    self.publish_oracle_override(False, phase="done")
            elif task_type == "semantic_target" and v10_mode in {"target_relative_approach", "geometric_watchdog"}:
                self.publish_v10_target_controller(status=status, safety=safety, pose=pose, target=target, elapsed=elapsed)

        def publish_oracle_override(self, active: bool, *, phase: str) -> None:
            self.oracle_override_pub.publish(
                String(
                    data=json.dumps(
                        {
                            "active": bool(active),
                            "episode_id": self.active_episode_id,
                            "task_id": self.task.get("task_id"),
                            "phase": phase,
                            "source": str(self.active_mode_config.get("oracle_source") or "geometric_oracle"),
                        }
                    )
                )
            )

        def publish_v10_branch_controller(
            self,
            *,
            safety: dict[str, Any],
            pose: list[float],
            target: dict[str, Any],
            elapsed: float,
        ) -> None:
            mode = self.active_mode_config
            decision_source = str(mode.get("controller_decision_source") or "oracle")
            waiting_for_step_route = False
            if decision_source == "step":
                direction = self.v10_step_route_choice
                if direction not in {"left", "right"}:
                    waiting_for_step_route = True
                    direction = "front"
            else:
                direction = expected_branch(self.task, self.scene)
                if direction not in {"left", "right"}:
                    direction = self.route_choice_for_task(pose, target)
            target_yaw_deg = 0.0 if waiting_for_step_route else branch_yaw_deg(
                direction, float(mode.get("branch_target_yaw_deg", 75.0))
            )
            target_yaw_rad = math.radians(target_yaw_deg)
            yaw_error = wrap_to_pi(target_yaw_rad - float(pose[2]))
            yaw_error_deg = rad2deg(yaw_error)
            cx, _ = intersection_center(self.scene)
            pre_entry_x = cx - float(mode.get("branch_pre_entry_distance_m", 0.0))
            pre_entry_pose = [pre_entry_x, 0.0, 0.0]
            centerline_lookahead_pose = [cx + float(mode.get("branch_centerline_lookahead_m", 1.0)), 0.0, 0.0]
            advance_target_pose = self.v10_branch_navigation_pose(pose, centerline_lookahead_pose, direction)
            advance_vec = target_vector(pose, advance_target_pose)
            verify_vec = target_vector(pose, target.get("pose", [0.0, 0.0, 0.0]))
            verify_yaw_error = float(verify_vec["yaw_error_rad"])
            target_visible_required = bool((self.task.get("success") or {}).get("target_visible", False))
            target_visible_for_verify = object_visible(pose, target, blockers=self.scene.get("obstacles", []))
            membership = branch_membership(pose, self.scene)
            entered_correct = False if waiting_for_step_route else bool(membership.get(direction))
            phase_timeouts = {
                "advance": float(mode.get("branch_advance_timeout_sec", mode.get("branch_max_phase_duration_sec", 45.0))),
                "rotate": float(mode.get("branch_rotate_timeout_sec", mode.get("branch_max_phase_duration_sec", 12.0))),
                "enter": float(mode.get("branch_enter_timeout_sec", mode.get("branch_max_phase_duration_sec", 12.0))),
                "verify": float(mode.get("branch_verify_timeout_sec", 20.0)),
            }
            rotate_stop = float(mode.get("branch_rotate_stop_error_deg", 5.0))
            post_forward = float(mode.get("branch_post_turn_forward_m", 0.8))
            max_enter_forward = float(mode.get("branch_max_enter_forward_m", post_forward + 0.7))
            max_yaw_rate = float(mode.get("branch_max_yaw_rate_radps", 0.35))
            forward_speed = float(mode.get("branch_forward_speed_mps", 0.20))
            advance_heading = float(advance_vec["desired_yaw_rad"])
            advance_yaw_error = float(advance_vec["yaw_error_rad"])
            advance_yaw_start = float(mode.get("branch_advance_yaw_start_deg", 15.0))
            advance_yaw_stop = float(mode.get("branch_advance_yaw_stop_deg", 8.0))
            lateral_tolerance = float(mode.get("branch_pre_entry_lateral_tolerance_m", 0.4))

            if not self.v10_phase:
                self.v10_set_phase("advance", elapsed, pose)
            phase_age = elapsed - self.v10_phase_started
            phase_timeout = phase_timeouts.get(self.v10_phase, float(mode.get("branch_max_phase_duration_sec", 45.0)))
            if phase_age > phase_timeout and self.v10_phase not in {"done", "failsafe"}:
                self.v10_set_phase("failsafe", elapsed, pose)
            near_gate = bool(mode.get("branch_rotate_on_near_intersection", False)) and bool(safety.get("near_intersection"))
            at_pre_entry = branch_pre_entry_reached(
                pose,
                pre_entry_pose,
                reached_m=float(mode.get("branch_pre_entry_reached_m", 0.3)),
                lateral_tolerance_m=lateral_tolerance,
            )
            if self.v10_phase == "advance" and (at_pre_entry or near_gate):
                self.v10_set_phase("await_route_choice" if waiting_for_step_route else "rotate", elapsed, pose)
            if (
                not waiting_for_step_route
                and self.v10_phase in {"rotate", "enter", "verify", "done"}
                and not self.v10_decision_sent
            ):
                self.v10_decision_sent = True
                controller_source = "step" if decision_source == "step" else str(
                    mode.get("oracle_source") or "geometric_oracle"
                )
                if decision_source != "step":
                    self.oracle_route_pub.publish(
                        String(
                            data=json.dumps(
                                {
                                    "route_choice": direction,
                                    "source": controller_source,
                                    "episode_id": self.active_episode_id,
                                    "task_id": self.task.get("task_id"),
                                }
                            )
                        )
                    )
                self.publish_metric(
                    "v10_branch_decision_published",
                    direction=direction,
                    route_choice=direction,
                    source=controller_source,
                    target_yaw_deg=round(target_yaw_deg, 3),
                    post_turn_forward_m=post_forward,
                    pose=pose,
                    elapsed_sec=round(elapsed, 3),
                )
            if self.v10_phase == "rotate" and abs(yaw_error_deg) <= rotate_stop:
                self.v10_set_phase("enter", elapsed, pose)
            if self.v10_phase == "enter":
                moved = distance_xy(pose, self.v10_phase_start_pose)
                if (entered_correct and moved >= post_forward) or moved >= max_enter_forward:
                    self.v10_set_phase("verify", elapsed, pose)
            phase_age = elapsed - self.v10_phase_started
            if self.v10_phase == "verify" and branch_verify_ready(
                entered_correct=entered_correct,
                target_visible_required=target_visible_required,
                target_visible=target_visible_for_verify,
            ):
                self.v10_done = True
                self.v10_phase = "done"

            linear = 0.0
            angular = 0.0
            if self.v10_phase == "advance":
                if self.v10_target_orienting and abs(rad2deg(advance_yaw_error)) <= advance_yaw_stop:
                    self.v10_target_orienting = False
                elif not self.v10_target_orienting and abs(rad2deg(advance_yaw_error)) >= advance_yaw_start:
                    self.v10_target_orienting = True
                linear = (
                    float(mode.get("branch_advance_orient_linear_x_mps", 0.0))
                    if self.v10_target_orienting
                    else forward_speed
                )
                angular = clamp(float(mode.get("branch_advance_yaw_gain", 1.0)) * advance_yaw_error, -max_yaw_rate, max_yaw_rate)
            elif self.v10_phase == "rotate":
                linear = float(mode.get("branch_rotate_linear_x_mps", 0.0))
                angular = math.copysign(max_yaw_rate, yaw_error) if abs(yaw_error_deg) > rotate_stop else 0.0
            elif self.v10_phase == "enter":
                linear = forward_speed
                angular = clamp(0.8 * yaw_error, -max_yaw_rate, max_yaw_rate)
            elif self.v10_phase == "verify":
                linear = 0.0
                angular = clamp(
                    float(mode.get("branch_verify_yaw_gain", 1.0)) * verify_yaw_error,
                    -max_yaw_rate,
                    max_yaw_rate,
                )
            elif self.v10_phase in {"done", "failsafe"}:
                linear = 0.0
                angular = 0.0
            elif self.v10_phase == "await_route_choice":
                linear = 0.0
                angular = 0.0

            primitive = {
                "primitive": "enter_branch",
                "phase": self.v10_phase,
                "direction": "pending" if waiting_for_step_route else direction,
                "target_yaw_deg": round(target_yaw_deg, 3),
                "desired_yaw_rad": round(target_yaw_rad, 6),
                "yaw_error_rad": round(yaw_error, 6),
                "post_turn_forward_m": post_forward,
                "max_enter_forward_m": max_enter_forward,
                "linear_x_mps": round(linear, 4),
                "angular_z_radps": round(angular, 4),
                "max_linear_x_mps": forward_speed,
                "max_yaw_rate_radps": max_yaw_rate,
                "ttl_sec": float(mode.get("v10_primitive_ttl_sec", 0.5)),
                "source": (
                    "step_geometric_approach" if waiting_for_step_route else "step_geometric_controller"
                ) if decision_source == "step" else str(
                    mode.get("oracle_source") or "geometric_oracle"
                ),
                "episode_id": self.active_episode_id,
                "task_id": self.task.get("task_id"),
            }
            self.publish_v10_controller_trace(
                "v10_branch_controller_trace",
                primitive=primitive,
                pose=pose,
                phase=self.v10_phase,
                elapsed=elapsed,
                direction="pending" if waiting_for_step_route else direction,
                target_yaw_deg=target_yaw_deg,
                post_turn_forward_m=post_forward,
                yaw_error_deg=yaw_error_deg,
                advance_heading_deg=rad2deg(advance_heading),
                advance_yaw_error_deg=rad2deg(advance_yaw_error),
                advance_target_pose=advance_target_pose,
                pre_entry_pose=pre_entry_pose,
                centerline_lookahead_pose=centerline_lookahead_pose,
                at_pre_entry=at_pre_entry,
                bypass_active=not self.v10_bypass_complete and advance_target_pose != pre_entry_pose,
                inside_left_polygon=membership["left"],
                inside_right_polygon=membership["right"],
                entered_correct_branch=entered_correct,
                target_visible_required=target_visible_required,
                target_visible_for_verify=target_visible_for_verify,
                verify_yaw_error_deg=rad2deg(verify_yaw_error),
            )
            self.publish_v10_structured_primitive(primitive, elapsed=elapsed)

        def v10_branch_navigation_pose(
            self,
            pose: list[float],
            centerline_lookahead_pose: list[float],
            direction: str,
        ) -> list[float]:
            mode = self.active_mode_config
            if not bool(mode.get("branch_bypass_enabled", True)) or self.v10_bypass_complete:
                return list(centerline_lookahead_pose)
            obstacles = self.scene.get("obstacles", [])
            if not obstacles:
                return list(centerline_lookahead_pose)
            if self.v10_bypass_side is None:
                self.v10_bypass_side = 1.0 if direction == "left" else -1.0
            for obstacle in obstacles:
                obstacle_pose = obstacle.get("pose", [999.0, 999.0, 0.0])
                if float(obstacle_pose[0]) > float(centerline_lookahead_pose[0]) + 0.2:
                    continue
                release_after = float(mode.get("branch_bypass_release_after_x_m", 0.8))
                waypoint = bypass_waypoint(
                    obstacle_pose,
                    centerline_lookahead_pose,
                    self.v10_bypass_side,
                    release_after_x_m=release_after,
                    margin_x_m=float(mode.get("branch_bypass_waypoint_margin_x_m", 0.2)),
                    offset_y_m=float(mode.get("branch_bypass_offset_y_m", 1.1)),
                )
                if bypass_released(
                    pose,
                    obstacle_pose,
                    waypoint,
                    self.v10_bypass_side,
                    release_after_x_m=release_after,
                    clearance_y_m=float(mode.get("branch_bypass_clearance_y_m", 0.8)),
                    reached_m=float(mode.get("branch_bypass_waypoint_reached_m", 0.30)),
                ):
                    self.v10_bypass_complete = True
                    self.v10_target_orienting = True
                    return list(centerline_lookahead_pose)
                return waypoint
            self.v10_bypass_complete = True
            return list(centerline_lookahead_pose)

        def publish_v10_target_controller(
            self,
            *,
            status: dict[str, Any],
            safety: dict[str, Any],
            pose: list[float],
            target: dict[str, Any],
            elapsed: float,
        ) -> None:
            mode = self.active_mode_config
            yaw_delta = wrap_to_pi(float(pose[2]) - float(self.v10_target_last_yaw))
            previous_polarity = self.v10_target_yaw_polarity
            if target_yaw_polarity_adaptation_enabled(mode):
                (
                    self.v10_target_yaw_polarity,
                    self.v10_target_yaw_mismatch_count,
                    polarity_flipped,
                ) = update_yaw_polarity(
                    self.v10_target_yaw_polarity,
                    self.v10_target_yaw_mismatch_count,
                    self.v10_target_last_desired_angular,
                    yaw_delta,
                )
                if polarity_flipped:
                    self.publish_metric(
                        "v10_target_yaw_polarity_flip",
                        old_polarity=previous_polarity,
                        new_polarity=self.v10_target_yaw_polarity,
                        desired_angular_radps=self.v10_target_last_desired_angular,
                        actual_yaw_delta_rad=yaw_delta,
                    )
            self.v10_target_last_yaw = float(pose[2])
            decision_source = str(mode.get("controller_decision_source") or "oracle")
            controller = str(mode.get("target_controller") or "rotate_then_forward")
            target_pose = target.get("pose", [0.0, 0.0, 0.0])
            real_vec = target_vector(pose, target_pose)
            nav_target_pose = self.v10_target_navigation_pose(pose, target_pose)
            vec = target_vector(pose, nav_target_pose)
            distance = float(real_vec["distance_2d_m"])
            nav_distance = float(vec["distance_2d_m"])
            yaw_error = float(vec["yaw_error_rad"])
            yaw_error_deg = rad2deg(yaw_error)
            coverage_threshold = float(mode.get("target_coverage_threshold_m", 2.5))
            threshold = float(mode.get("target_stop_threshold_m", 2.0))
            max_linear = float(mode.get("target_max_linear_x_mps", 0.20))
            max_yaw = target_max_yaw_rate(mode)
            yaw_start = float(mode.get("target_yaw_start_deg", 15.0))
            yaw_stop = float(mode.get("target_yaw_stop_deg", 8.0))
            max_phase = target_phase_timeout_sec(mode)
            if not self.v10_phase:
                self.v10_set_phase("orient", elapsed, pose)
            if elapsed - self.v10_phase_started > max_phase and self.v10_phase not in TARGET_HOLD_PHASES:
                self.v10_set_phase("failsafe" if decision_source == "step" else "stop", elapsed, pose)

            if distance <= coverage_threshold and not self.v10_coverage_sent:
                self.v10_coverage_sent = True
                self.publish_metric(
                    "semantic_stop_coverage_entered",
                    target_id=self.task.get("target_object"),
                    distance_m=round(distance, 3),
                    target_visible=bool(status.get("target_visible", False)),
                    elapsed_sec=round(elapsed, 3),
                )
                if decision_source == "step" and bool(mode.get("semantic_coverage_step_trigger", False)):
                    self.v10_set_phase("await_step_track", elapsed, pose)
                    self.mission_event_pub.publish(
                        String(
                            data=json.dumps(
                                {
                                    "type": "candidate_goal_reached",
                                    "reason": "semantic_visual_coverage_entered",
                                    "source": "isaac_semantic_coverage",
                                    "episode_id": self.active_episode_id,
                                    "mission_id": self.active_episode_id,
                                    "instruction": self.task.get("instruction", ""),
                                    "current_subgoal": self.task.get("instruction", ""),
                                    "distance_to_target": round(distance, 3),
                                    "target_visible": bool(status.get("target_visible", False)),
                                    "multimodal": False,
                                }
                            )
                        )
                    )
                    self.publish_metric(
                        "semantic_coverage_step_trigger",
                        result="candidate_goal_reached",
                        distance_m=round(distance, 3),
                        target_visible=bool(status.get("target_visible", False)),
                    )

            if distance <= threshold:
                if decision_source == "step":
                    if not self.v10_step_stop_accepted:
                        self.v10_set_phase("await_step_stop", elapsed, pose)
                else:
                    self.v10_set_phase("stop", elapsed, pose)
                    if not self.v10_stop_sent:
                        self.v10_stop_sent = True
                        stop_status = dict(status)
                        stop_status["target_visible"] = bool(status.get("target_visible", False))
                        stop_status["distance_to_target"] = round(distance, 3)
                        self.publish_stop_oracle(status=stop_status, elapsed=elapsed)
            if self.v10_phase == "stop" and self.v10_stop_sent and elapsed - self.v10_phase_started >= float(mode.get("target_verify_hold_sec", 1.0)):
                self.v10_set_phase("verify", elapsed, pose)

            linear = 0.0
            angular = 0.0
            phase = self.v10_phase
            if self.v10_phase not in TARGET_HOLD_PHASES:
                if controller == "pure_forward":
                    phase = "approach"
                    linear = max_linear
                    angular = 0.0
                elif controller == "pure_rotate":
                    phase = "orient"
                    linear = 0.0
                    angular = 0.0 if abs(yaw_error_deg) <= yaw_stop else math.copysign(max_yaw, yaw_error)
                elif controller == "crawl_turn_waypoint":
                    phase, self.v10_target_orienting, linear, angular = crawl_turn_command(
                        yaw_error,
                        orienting=self.v10_target_orienting,
                        local_costmap_clear=bool(safety.get("local_costmap_clear", False)),
                        yaw_start_deg=yaw_start,
                        yaw_stop_deg=yaw_stop,
                        orient_linear_x_mps=float(mode.get("target_orient_linear_x_mps", 0.12)),
                        blocked_reverse_speed_mps=float(mode.get("target_blocked_reverse_speed_mps", 0.08)),
                        crawl_max_yaw_error_deg=target_crawl_max_yaw_error_deg(mode),
                        base_speed_mps=max_linear,
                        max_yaw_rate_radps=max_yaw,
                        k_yaw=float(mode.get("target_yaw_kp", 0.8)),
                    )
                elif controller in {"proportional", "waypoint"}:
                    if self.v10_target_orienting:
                        self.v10_target_orienting = abs(yaw_error_deg) > yaw_stop
                    elif abs(yaw_error_deg) > yaw_start:
                        self.v10_target_orienting = True
                    angular = clamp(0.8 * yaw_error, -max_yaw, max_yaw)
                    if self.v10_target_orienting:
                        phase = "orient"
                        linear = 0.0
                    else:
                        phase = "approach"
                        linear = max_linear * max(0.0, math.cos(yaw_error))
                else:
                    if self.v10_target_orienting:
                        self.v10_target_orienting = abs(yaw_error_deg) > yaw_stop
                    elif abs(yaw_error_deg) > yaw_start:
                        self.v10_target_orienting = True
                    if self.v10_target_orienting:
                        phase = "orient"
                        angular = math.copysign(max_yaw, yaw_error)
                        linear = 0.0
                    else:
                        phase = "approach"
                        linear = max_linear
                        angular = clamp(0.8 * yaw_error, -max_yaw, max_yaw)
                if phase != self.v10_phase:
                    self.v10_set_phase(phase, elapsed, pose)
            else:
                phase = self.v10_phase

            desired_angular = angular
            if target_yaw_polarity_adaptation_enabled(mode):
                angular *= self.v10_target_yaw_polarity
            self.v10_target_last_desired_angular = desired_angular

            primitive = {
                "primitive": "target_relative_approach",
                "phase": phase,
                "controller": controller,
                "target_id": self.task.get("target_object"),
                "desired_yaw_rad": round(float(vec["desired_yaw_rad"]), 6),
                "yaw_error_rad": round(yaw_error, 6),
                "distance_m": round(distance, 3),
                "nav_distance_m": round(nav_distance, 3),
                "nav_target_pose": nav_target_pose,
                "linear_x_mps": round(linear, 4),
                "angular_z_radps": round(angular, 4),
                "desired_angular_z_radps": round(desired_angular, 4),
                "yaw_polarity": self.v10_target_yaw_polarity,
                "max_linear_x_mps": max_linear,
                "max_yaw_rate_radps": max_yaw,
                "stop_threshold_m": threshold,
                "coverage_threshold_m": coverage_threshold,
                "ttl_sec": float(mode.get("v10_primitive_ttl_sec", 0.5)),
                "source": "step_geometric_controller" if decision_source == "step" else str(
                    mode.get("oracle_source") or "geometric_oracle"
                ),
                "episode_id": self.active_episode_id,
                "task_id": self.task.get("task_id"),
            }
            self.publish_v10_controller_trace(
                "v10_target_controller_trace",
                primitive=primitive,
                pose=pose,
                phase=phase,
                elapsed=elapsed,
                controller=controller,
                target_id=self.task.get("target_object"),
                target_pose=target_pose,
                nav_target_pose=nav_target_pose,
                desired_yaw_deg=rad2deg(float(vec["desired_yaw_rad"])),
                yaw_error_deg=yaw_error_deg,
                distance_m=distance,
                nav_distance_m=nav_distance,
                target_visible=bool(status.get("target_visible", False)),
                yaw_polarity=self.v10_target_yaw_polarity,
                yaw_polarity_mismatch_count=self.v10_target_yaw_mismatch_count,
                target_visible_id=self.task.get("target_object") if bool(status.get("target_visible", False)) else "",
                success_judge_target_id=self.task.get("target_object"),
                safe_cmd_linear_x_mps=round(float(self.active_cmd().linear.x), 4),
                safe_cmd_angular_z_radps=round(float(self.active_cmd().angular.z), 4),
                actual_linear_velocity=self.telemetry_linear_velocity,
                actual_angular_velocity=self.telemetry_angular_velocity,
                robot_z=self.telemetry_z,
                robot_fallen_or_unstable=bool(safety.get("robot_fallen_or_unstable", False)),
                collision=bool(safety.get("collision", False)),
                local_costmap_clear=bool(safety.get("local_costmap_clear", False)),
                obstacle_distance_m=safety.get("obstacle_distance_m"),
                path_obstacle_distance_m=safety.get("path_obstacle_distance_m"),
                telemetry_event=self.telemetry_event,
                telemetry_episode_id=self.telemetry_episode_id,
            )
            if not self.v10_stop_sent:
                self.publish_v10_structured_primitive(primitive, elapsed=elapsed)

        def v10_target_navigation_pose(self, pose: list[float], target_pose: list[float]) -> list[float]:
            if not bool(self.active_mode_config.get("target_bypass_enabled", True)):
                return list(target_pose)
            if self.v10_bypass_complete:
                return list(target_pose)
            obstacles = self.scene.get("obstacles", [])
            if not obstacles:
                return list(target_pose)
            robot_x = float(pose[0])
            robot_y = float(pose[1])
            target_x = float(target_pose[0])
            target_y = float(target_pose[1])
            if self.v10_bypass_side is None:
                self.v10_bypass_side = choose_bypass_side(
                    robot_y,
                    target_y,
                    float(self.active_mode_config.get("target_bypass_side_deadband_m", 0.25)),
                )
            direction_y = self.v10_bypass_side
            for obstacle in obstacles:
                obstacle_pose = obstacle.get("pose", [999.0, 999.0, 0.0])
                ox = float(obstacle_pose[0])
                oy = float(obstacle_pose[1])
                if ox > target_x + 0.2:
                    continue
                release_after = float(self.active_mode_config.get("target_bypass_release_after_x_m", 1.0))
                waypoint = bypass_waypoint(
                    obstacle_pose,
                    target_pose,
                    direction_y,
                    release_after_x_m=release_after,
                    margin_x_m=float(self.active_mode_config.get("target_bypass_waypoint_margin_x_m", 0.2)),
                    offset_y_m=float(self.active_mode_config.get("target_bypass_offset_y_m", 1.25)),
                )
                if bypass_released(
                    pose,
                    obstacle_pose,
                    waypoint,
                    direction_y,
                    release_after_x_m=release_after,
                    clearance_y_m=float(self.active_mode_config.get("target_bypass_clearance_y_m", 0.9)),
                    reached_m=float(self.active_mode_config.get("target_bypass_waypoint_reached_m", 0.30)),
                ):
                    self.v10_bypass_complete = True
                    return list(target_pose)
                return waypoint
            return list(target_pose)

        def v10_set_phase(self, phase: str, elapsed: float, pose: list[float]) -> None:
            if phase == self.v10_phase:
                return
            self.v10_phase = phase
            self.v10_phase_started = elapsed
            self.v10_phase_start_pose = list(pose)
            self.publish_metric(
                "v10_controller_phase",
                phase=phase,
                pose=pose,
                elapsed_sec=round(elapsed, 3),
            )

        def publish_v10_structured_primitive(self, primitive: dict[str, Any], *, elapsed: float) -> bool:
            now_mono = time.monotonic()
            interval = float(self.active_mode_config.get("v10_primitive_interval_sec", 0.2))
            if now_mono - self.v10_last_primitive_publish < interval:
                return False
            self.v10_last_primitive_publish = now_mono
            self.v10_trace_count += 1
            ros_now = self.get_clock().now().nanoseconds / 1e9
            payload = dict(primitive)
            payload.update(
                {
                    "request_id": f"v10_{payload.get('primitive')}_{self.v10_trace_count:04d}",
                    "created_ros_time_sec": ros_now,
                    "source_stamp_sec": ros_now,
                    "elapsed_sec": round(elapsed, 3),
                }
            )
            self.primitive_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
            return True

        def publish_simple_goal_stop(self, *, status: dict[str, Any], elapsed: float) -> None:
            if not bool(
                self.active_mode_config.get("simple_goal_stop_enabled", False)
                or self.active_mode_config.get("low_speed_primitive_stabilizer", False)
            ):
                return
            if str(self.task.get("task_type") or "") != "simple_navigation":
                return
            if bool(status.get("done", False)):
                if self.simple_goal_stop_latched:
                    self.publish_oracle_override(False, phase="simple_goal_done")
                self.simple_goal_stop_latched = False
                return
            if simple_goal_stop_should_latch(self.task, status):
                if not self.simple_goal_stop_latched:
                    self.publish_metric(
                        "simple_goal_stop_latched",
                        distance_to_target=status.get("distance_to_target"),
                        target_visible=bool(status.get("target_visible", False)),
                    )
                self.simple_goal_stop_latched = True
            if not self.simple_goal_stop_latched:
                return
            self.publish_oracle_override(True, phase="simple_goal_stop")
            self.publish_v10_structured_primitive(
                {
                    "primitive": "stop",
                    "phase": "simple_goal_stop",
                    "source": "goal_tolerance_controller",
                    "ttl_sec": 0.5,
                    "episode_id": self.active_episode_id,
                    "task_id": self.task.get("task_id"),
                },
                elapsed=elapsed,
            )

        def publish_v10_controller_trace(
            self,
            event_type: str,
            *,
            primitive: dict[str, Any],
            pose: list[float],
            phase: str,
            elapsed: float,
            **kwargs: Any,
        ) -> None:
            payload = {
                "phase": phase,
                "pose": pose,
                "robot_pose": pose,
                "primitive": primitive,
                "elapsed_sec": round(elapsed, 3),
            }
            payload.update(kwargs)
            self.publish_metric(event_type, **payload)

        def publish_v9_probe_actions(
            self,
            *,
            status: dict[str, Any],
            safety: dict[str, Any],
            pose: list[float],
            target: dict[str, Any],
            elapsed: float,
        ) -> None:
            mode = self.active_mode_config
            v9_mode = str(mode.get("v9_mode") or "")
            if not v9_mode:
                return
            if bool(status.get("done")):
                return
            task_type = str(self.task.get("task_type") or "")
            if task_type in {"turn_choice", "turn_microbench", "failure_recovery"}:
                self.publish_v9_route_probe(safety=safety, pose=pose, target=target, elapsed=elapsed)
            if task_type == "semantic_target":
                self.publish_v9_semantic_probe(status=status, pose=pose, elapsed=elapsed)

        def publish_v9_route_probe(
            self,
            *,
            safety: dict[str, Any],
            pose: list[float],
            target: dict[str, Any],
            elapsed: float,
        ) -> None:
            mode = self.active_mode_config
            v9_mode = str(mode.get("v9_mode") or "")
            probe_mode = str(mode.get("v9_probe_mode") or "")
            near_intersection = bool(safety.get("near_intersection") or safety.get("near_doorway"))
            trigger_x = self.v9_route_trigger_x(mode)
            reached_trigger = bool(float(pose[0]) >= trigger_x or near_intersection)
            if reached_trigger and not self.v9_route_trigger_reached_sent:
                self.v9_route_trigger_reached_sent = True
                self.publish_metric(
                    "v9_route_trigger_reached",
                    v9_mode=v9_mode,
                    probe_mode=probe_mode,
                    pose=pose,
                    elapsed_sec=round(elapsed, 3),
                    trigger_x_m=round(trigger_x, 3),
                    near_intersection_detected=near_intersection,
                    reached_injection_window=True,
                )

            if v9_mode == "route_trigger_reachability":
                if probe_mode == "forced_forward_until_trigger" and not reached_trigger:
                    self.publish_v9_assist_primitive(
                        primitive="move_forward",
                        distance_m=float(mode.get("progress_assist_distance_m", 0.5)),
                        ttl_sec=float(mode.get("progress_assist_ttl_sec", 0.9)),
                        source="v9_forced_forward_until_trigger",
                        reason="route_trigger_not_reached",
                        elapsed=elapsed,
                        pose=pose,
                    )
                elif probe_mode == "omninav_with_progress_assist":
                    reason = self.v9_route_watchdog_reason(pose=pose, elapsed=elapsed)
                    if reason:
                        self.publish_v9_watchdog_metric(reason=reason, pose=pose, elapsed=elapsed)
                        self.publish_v9_assist_primitive(
                            primitive="move_forward",
                            distance_m=float(mode.get("progress_assist_distance_m", 0.5)),
                            ttl_sec=float(mode.get("progress_assist_ttl_sec", 0.9)),
                            source="v9_progress_assist_forward",
                            reason=reason,
                            elapsed=elapsed,
                            pose=pose,
                        )
                return

            if v9_mode == "route_injection_sweep":
                window_ok, window_name = self.v9_route_injection_window_satisfied(safety=safety, pose=pose, mode=mode)
                if not window_ok and bool(mode.get("route_progress_assist_enabled", True)):
                    self.publish_v9_assist_primitive(
                        primitive="move_forward",
                        distance_m=float(mode.get("progress_assist_distance_m", 0.5)),
                        ttl_sec=float(mode.get("progress_assist_ttl_sec", 0.9)),
                        source="v9_route_injection_sweep_forward",
                        reason=f"before_{window_name}",
                        elapsed=elapsed,
                        pose=pose,
                    )
                if window_ok and not self.v9_route_injection_published:
                    self.v9_route_injection_published = True
                    gate = {
                        "v9_mode": v9_mode,
                        "trigger_window": window_name,
                        "trigger_x_m": round(trigger_x, 3),
                        "pose": pose,
                        "elapsed_sec": round(elapsed, 3),
                        "near_intersection_detected": bool(near_intersection),
                        "route_oracle_triggered": True,
                        "route_oracle_json_published": True,
                    }
                    self.publish_metric("v9_route_injection_window", **gate)
                    self.publish_route_oracle(
                        pose=pose,
                        target=target,
                        elapsed=elapsed,
                        source="deterministic_oracle",
                        gate=gate,
                    )
                return

            if v9_mode == "progress_to_trigger_watchdog":
                reason = self.v9_route_watchdog_reason(pose=pose, elapsed=elapsed)
                if reason:
                    self.publish_v9_watchdog_metric(reason=reason, pose=pose, elapsed=elapsed)
                    self.publish_v9_assist_primitive(
                        primitive="move_forward",
                        distance_m=float(mode.get("progress_assist_distance_m", 0.5)),
                        ttl_sec=float(mode.get("progress_assist_ttl_sec", 0.9)),
                        source="v9_progress_assist_forward",
                        reason=reason,
                        elapsed=elapsed,
                        pose=pose,
                    )

        def publish_v9_semantic_probe(self, *, status: dict[str, Any], pose: list[float], elapsed: float) -> None:
            mode = self.active_mode_config
            v9_mode = str(mode.get("v9_mode") or "")
            probe_mode = str(mode.get("v9_probe_mode") or "")
            visible = bool(status.get("target_visible", False))
            try:
                distance = float(status.get("distance_to_target"))
            except (TypeError, ValueError):
                distance = None
            threshold = float(mode.get("stop_oracle_distance_m", 2.5))
            if visible and self.v9_semantic_visible_since is None:
                self.v9_semantic_visible_since = elapsed
                self.publish_metric(
                    "v9_semantic_target_visible",
                    v9_mode=v9_mode,
                    probe_mode=probe_mode,
                    pose=pose,
                    elapsed_sec=round(elapsed, 3),
                    distance_to_target=distance,
                )
            if visible and distance is not None and distance <= threshold and not self.v9_semantic_threshold_reached_sent:
                self.v9_semantic_threshold_reached_sent = True
                self.publish_metric(
                    "v9_semantic_stop_threshold_reached",
                    v9_mode=v9_mode,
                    probe_mode=probe_mode,
                    pose=pose,
                    elapsed_sec=round(elapsed, 3),
                    distance_to_target=round(distance, 3),
                    threshold_m=threshold,
                )

            should_assist = False
            reason = ""
            if visible and distance is not None and distance > threshold:
                if v9_mode == "semantic_approach_to_stop_threshold" and probe_mode == "forced_approach":
                    should_assist = True
                    reason = "forced_approach_visible_outside_stop_threshold"
                elif v9_mode == "semantic_approach_to_stop_threshold" and probe_mode == "omninav_with_approach_assist":
                    visible_age = 0.0 if self.v9_semantic_visible_since is None else elapsed - self.v9_semantic_visible_since
                    if visible_age >= float(mode.get("semantic_watchdog_visible_delay_sec", 5.0)):
                        should_assist = True
                        reason = "semantic_visible_distance_gt_threshold_5s"
                elif v9_mode == "progress_to_trigger_watchdog":
                    visible_age = 0.0 if self.v9_semantic_visible_since is None else elapsed - self.v9_semantic_visible_since
                    if visible_age >= float(mode.get("semantic_watchdog_visible_delay_sec", 5.0)):
                        should_assist = True
                        reason = "semantic_visible_distance_gt_threshold_5s"

            if should_assist:
                if v9_mode == "progress_to_trigger_watchdog":
                    self.publish_v9_watchdog_metric(reason=reason, pose=pose, elapsed=elapsed, distance_to_target=distance)
                self.publish_v9_assist_primitive(
                    primitive="move_forward",
                    distance_m=float(mode.get("semantic_approach_distance_m", 0.3)),
                    ttl_sec=float(mode.get("semantic_approach_ttl_sec", 0.8)),
                    source="v9_approach_assist_forward" if v9_mode == "progress_to_trigger_watchdog" else f"v9_{probe_mode}",
                    reason=reason,
                    elapsed=elapsed,
                    pose=pose,
                    distance_to_target=distance,
                )

        def publish_v9_assist_primitive(
            self,
            *,
            primitive: str,
            distance_m: float,
            ttl_sec: float,
            source: str,
            reason: str,
            elapsed: float,
            pose: list[float],
            distance_to_target: float | None = None,
        ) -> bool:
            mode = self.active_mode_config
            now_mono = time.monotonic()
            interval = float(mode.get("v9_assist_interval_sec", mode.get("progress_assist_interval_sec", 0.8)))
            if now_mono - self.v9_last_assist_publish < interval:
                return False
            self.v9_last_assist_publish = now_mono
            self.v9_assist_publish_count += 1
            ros_now = self.get_clock().now().nanoseconds / 1e9
            payload = {
                "primitive": primitive,
                "distance_m": round(float(distance_m), 3),
                "speed_mps": round(float(mode.get("progress_assist_speed_mps", 0.2)), 3),
                "ttl_sec": round(float(ttl_sec), 3),
                "source": source,
                "reason": reason,
                "episode_id": self.active_episode_id,
                "task_id": self.task.get("task_id"),
                "request_id": f"{source}_{self.v9_assist_publish_count:04d}",
                "created_ros_time_sec": ros_now,
                "source_stamp_sec": ros_now,
                "elapsed_sec": round(elapsed, 3),
            }
            self.primitive_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
            self.publish_metric(
                "v9_progress_assist_forward",
                source=source,
                reason=reason,
                primitive=payload,
                pose=pose,
                distance_to_target=distance_to_target,
                elapsed_sec=round(elapsed, 3),
                assist_publish_count=self.v9_assist_publish_count,
            )
            return True

        def publish_v9_watchdog_metric(
            self,
            *,
            reason: str,
            pose: list[float],
            elapsed: float,
            distance_to_target: float | None = None,
        ) -> None:
            if reason in self.v9_progress_watchdog_reasons:
                return
            self.v9_progress_watchdog_reasons.add(reason)
            self.publish_metric(
                "v9_progress_watchdog",
                assist_triggered=True,
                assist_reason=reason,
                pose=pose,
                elapsed_sec=round(elapsed, 3),
                distance_to_target=distance_to_target,
            )

        def v9_route_watchdog_reason(self, *, pose: list[float], elapsed: float) -> str:
            x = float(pose[0])
            if elapsed > 20.0 and x < 2.0:
                return "elapsed_gt_20_x_lt_2"
            if elapsed > 40.0 and x < 3.0:
                return "elapsed_gt_40_x_lt_3"
            return ""

        def v9_route_injection_window_satisfied(
            self, *, safety: dict[str, Any], pose: list[float], mode: dict[str, Any]
        ) -> tuple[bool, str]:
            near_intersection = bool(safety.get("near_intersection") or safety.get("near_doorway"))
            window = str(mode.get("route_injection_window") or "")
            if not window:
                window = "near_intersection" if mode.get("route_injection_window_type") == "near_intersection" else "x>=3.0"
            if window == "near_intersection":
                return near_intersection, window
            if window == "intersection_x-0.8":
                return float(pose[0]) >= self.v9_route_trigger_x(mode), window
            threshold = mode.get("route_injection_x_m")
            if threshold is None and window.startswith("x>="):
                try:
                    threshold = float(window.split(">=", 1)[1])
                except (TypeError, ValueError):
                    threshold = 3.0
            threshold_f = float(threshold if threshold is not None else 3.0)
            return float(pose[0]) >= threshold_f, f"x>={threshold_f:g}"

        def v9_route_trigger_x(self, mode: dict[str, Any]) -> float:
            explicit = mode.get("route_trigger_x_m")
            if explicit is not None:
                try:
                    return float(explicit)
                except (TypeError, ValueError):
                    pass
            centers = []
            for zone in self.scene.get("semantic_zones", []):
                if str(zone.get("class") or "") in {"intersection", "doorway"}:
                    center = zone.get("center", [0.0, 0.0])
                    try:
                        centers.append(float(center[0]))
                    except (TypeError, ValueError, IndexError):
                        pass
            center_x = min(centers) if centers else 4.0
            return center_x - float(mode.get("route_trigger_distance_before_intersection_m", 0.8))

        def publish_v7_oracles(
            self,
            *,
            status: dict[str, Any],
            safety: dict[str, Any],
            pose: list[float],
            target: dict[str, Any],
            elapsed: float,
        ) -> None:
            route_gate = self.route_oracle_gate_status(safety=safety, pose=pose, elapsed=elapsed)
            if self.should_emit_route_coverage(route_gate):
                self.publish_metric("route_oracle_coverage", **route_gate)
            if route_gate["route_oracle_triggered"]:
                self.publish_route_oracle(
                    pose=pose,
                    target=target,
                    elapsed=elapsed,
                    source=str(route_gate["route_oracle_source"]),
                    gate=route_gate,
                )
            stop_gate = self.stop_oracle_gate_status(status=status, elapsed=elapsed)
            if self.should_emit_stop_coverage(stop_gate):
                self.publish_metric("semantic_stop_coverage", **stop_gate)
            if stop_gate["semantic_stop_oracle_triggered"]:
                self.publish_stop_oracle(status=status, elapsed=elapsed)

        def should_publish_route_oracle(self, *, safety: dict[str, Any], pose: list[float], elapsed: float) -> bool:
            return bool(self.route_oracle_gate_status(safety=safety, pose=pose, elapsed=elapsed)["route_oracle_triggered"])

        def route_oracle_gate_status(self, *, safety: dict[str, Any], pose: list[float], elapsed: float) -> dict[str, Any]:
            mode = self.active_mode_config
            task_type = str(self.task.get("task_type") or "")
            near_intersection = bool(safety.get("near_intersection") or safety.get("near_doorway"))
            distance_to_intersection = self.distance_to_route_zone(pose)
            deterministic = str(mode.get("route_injection_mode") or "").lower() == "deterministic" or str(
                mode.get("v8_mode") or ""
            ) in {"forced_route_injection", "deterministic_route_injection"}
            injection_distance = float(mode.get("route_injection_distance_to_intersection_m", 0.8))
            injection_band = float(mode.get("route_injection_band_m", 0.35))
            longitudinal = self.longitudinal_distance_to_route_zone(pose)
            deterministic_window = (
                deterministic
                and longitudinal is not None
                and longitudinal >= 0.0
                and abs(float(longitudinal) - injection_distance) <= injection_band
            )
            approaching_intersection = elapsed >= 1.0 and 2.6 <= float(pose[0]) <= 4.8
            trigger_window = deterministic_window if deterministic else (near_intersection or approaching_intersection)
            enabled = bool(mode.get("oracle_route_choice", False))
            task_ok = task_type in {"turn_choice", "turn_microbench", "failure_recovery"}
            limit_ok = self.oracle_route_publish_count < int(mode.get("route_oracle_max_publishes", 3))
            now_mono = time.monotonic()
            cooldown_ok = now_mono - self.oracle_last_route_publish >= float(mode.get("route_oracle_min_interval_sec", 0.6))
            triggered = enabled and task_ok and limit_ok and cooldown_ok and trigger_window
            failure_stage = None
            if not enabled:
                failure_stage = "route_oracle_disabled"
            elif not task_ok:
                failure_stage = "task_type_not_route"
            elif not limit_ok:
                failure_stage = "route_oracle_publish_limit"
            elif not cooldown_ok:
                failure_stage = "route_oracle_cooldown"
            elif not trigger_window:
                failure_stage = "route_trigger_window_false"
            return {
                "episode_id": self.active_episode_id,
                "task_id": self.task.get("task_id"),
                "task_type": task_type,
                "mode": self.active_mode_name,
                "v8_mode": mode.get("v8_mode"),
                "pose": pose,
                "elapsed_sec": round(elapsed, 3),
                "near_intersection_detected": near_intersection,
                "near_doorway_detected": bool(safety.get("near_doorway")),
                "approaching_intersection": approaching_intersection,
                "deterministic_injection_window": deterministic_window,
                "distance_to_intersection_m": None if distance_to_intersection is None else round(distance_to_intersection, 3),
                "longitudinal_distance_to_intersection_m": None if longitudinal is None else round(float(longitudinal), 3),
                "injection_distance_to_intersection_m": injection_distance if deterministic else None,
                "route_oracle_triggered": triggered,
                "route_oracle_json_published": triggered,
                "route_oracle_source": "deterministic_oracle" if deterministic else "forced_oracle",
                "route_oracle_publish_count_before": self.oracle_route_publish_count,
                "failure_stage": failure_stage,
            }

        def should_emit_route_coverage(self, gate: dict[str, Any]) -> bool:
            v8_mode = str(self.active_mode_config.get("v8_mode") or "")
            if not v8_mode and not bool(self.active_mode_config.get("oracle_route_choice", False)):
                return False
            if gate.get("route_oracle_triggered"):
                return True
            if gate.get("near_intersection_detected") or gate.get("deterministic_injection_window"):
                if not self.oracle_route_near_event_sent:
                    self.oracle_route_near_event_sent = True
                    return True
            return False

        def distance_to_route_zone(self, pose: list[float]) -> float | None:
            distances = []
            for zone in self.scene.get("semantic_zones", []):
                if str(zone.get("class") or "") not in {"intersection", "doorway"}:
                    continue
                center = zone.get("center", [0.0, 0.0])
                distances.append(distance_xy(pose, center))
            return min(distances) if distances else None

        def longitudinal_distance_to_route_zone(self, pose: list[float]) -> float | None:
            best: float | None = None
            yaw = float(pose[2])
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            for zone in self.scene.get("semantic_zones", []):
                if str(zone.get("class") or "") not in {"intersection", "doorway"}:
                    continue
                center = zone.get("center", [0.0, 0.0])
                dx = float(center[0]) - float(pose[0])
                dy = float(center[1]) - float(pose[1])
                longitudinal = cos_yaw * dx + sin_yaw * dy
                if best is None or abs(longitudinal) < abs(best):
                    best = longitudinal
            return best

        def route_choice_for_task(self, pose: list[float], target: dict[str, Any]) -> str:
            text = str(self.task.get("instruction") or "").lower()
            if "left" in text:
                return "left"
            if "right" in text:
                return "right"
            target_pose = target.get("pose", [0.0, 0.0, 0.0])
            lateral = float(target_pose[1]) - float(pose[1])
            if lateral > 0.35:
                return "left"
            if lateral < -0.35:
                return "right"
            return "front"

        def publish_route_oracle(
            self,
            *,
            pose: list[float],
            target: dict[str, Any],
            elapsed: float,
            source: str = "forced_oracle",
            gate: dict[str, Any] | None = None,
        ) -> None:
            route = self.route_choice_for_task(pose, target)
            payload = {
                "route_choice": route,
                "confidence": 1.0,
                "source": source,
                "episode_id": self.active_episode_id,
                "task_id": self.task.get("task_id"),
                "target_object": self.task.get("target_object"),
                "visible_in_view": "left" if route == "left" else "right" if route == "right" else "front",
                "elapsed_sec": round(elapsed, 3),
            }
            if gate and gate.get("trigger_window"):
                payload["trigger_window"] = gate.get("trigger_window")
            self.oracle_route_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
            self.publish_metric(
                "route_oracle_json_published",
                route_choice=route,
                source=source,
                payload=payload,
                gate=gate or {},
                episode_id=self.active_episode_id,
                task_id=self.task.get("task_id"),
            )
            self.oracle_route_publish_count += 1
            self.oracle_last_route_publish = time.monotonic()

        def should_publish_stop_oracle(self, *, status: dict[str, Any], elapsed: float) -> bool:
            return bool(self.stop_oracle_gate_status(status=status, elapsed=elapsed)["semantic_stop_oracle_triggered"])

        def stop_oracle_gate_status(self, *, status: dict[str, Any], elapsed: float) -> dict[str, Any]:
            mode = self.active_mode_config
            v7_forced = str(mode.get("v7_mode") or "") == "forced_route_stop_oracle"
            v8_forced = str(mode.get("v8_mode") or "") in {
                "forced_route_stop_oracle",
                "semantic_stop_coverage",
                "forced_stop_oracle",
            }
            enabled = v7_forced or v8_forced or bool(mode.get("stop_oracle_enabled", False))
            max_publishes = int(mode.get("stop_oracle_max_publishes", 8))
            min_interval = float(mode.get("stop_oracle_min_interval_sec", 0.4))
            limit_ok = self.oracle_stop_publish_count < max_publishes
            cooldown_ok = time.monotonic() - self.oracle_last_stop_publish >= min_interval
            done = bool(status.get("done"))
            try:
                distance = float(status.get("distance_to_target"))
            except (TypeError, ValueError):
                distance = None
            visible = bool(status.get("target_visible", False))
            distance_ok = distance is not None and distance <= float(mode.get("stop_oracle_distance_m", 2.5))
            triggered = enabled and limit_ok and cooldown_ok and not done and visible and distance_ok and elapsed >= 0.5
            failure_stage = None
            if not enabled:
                failure_stage = "semantic_stop_oracle_disabled"
            elif not limit_ok:
                failure_stage = "semantic_stop_publish_limit"
            elif not cooldown_ok:
                failure_stage = "semantic_stop_cooldown"
            elif done:
                failure_stage = "episode_done"
            elif not visible:
                failure_stage = "target_not_visible"
            elif not distance_ok:
                failure_stage = "distance_not_ok"
            elif elapsed < 0.5:
                failure_stage = "startup_grace"
            return {
                "episode_id": self.active_episode_id,
                "task_id": self.task.get("task_id"),
                "task_type": self.task.get("task_type"),
                "mode": self.active_mode_name,
                "v8_mode": mode.get("v8_mode"),
                "elapsed_sec": round(elapsed, 3),
                "target_visible": visible,
                "distance_to_target": distance,
                "estimated_distance_ok": distance_ok,
                "semantic_stop_oracle_triggered": triggered,
                "semantic_stop_json_published": triggered,
                "semantic_stop_publish_count_before": self.oracle_stop_publish_count,
                "failure_stage": failure_stage,
            }

        def should_emit_stop_coverage(self, gate: dict[str, Any]) -> bool:
            v8_mode = str(self.active_mode_config.get("v8_mode") or "")
            if not v8_mode and not bool(self.active_mode_config.get("stop_oracle_enabled", False)):
                return bool(gate.get("semantic_stop_oracle_triggered"))
            if gate.get("semantic_stop_oracle_triggered"):
                return True
            if gate.get("target_visible") and not self.oracle_stop_visible_event_sent:
                self.oracle_stop_visible_event_sent = True
                return True
            return False

        def publish_stop_oracle(self, *, status: dict[str, Any], elapsed: float) -> None:
            payload = {
                "stop": True,
                "confidence": 1.0,
                "source": "forced_oracle",
                "episode_id": self.active_episode_id,
                "task_id": self.task.get("task_id"),
                "target_object": self.task.get("target_object"),
                "target_visible": bool(status.get("target_visible", False)),
                "distance_to_target": status.get("distance_to_target"),
                "elapsed_sec": round(elapsed, 3),
                "reason": "target_visible_within_2p5m",
            }
            self.oracle_stop_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
            self.publish_metric(
                "semantic_stop_json_published",
                source="forced_oracle",
                payload=payload,
                episode_id=self.active_episode_id,
                task_id=self.task.get("task_id"),
            )
            self.oracle_stop_publish_count += 1
            self.oracle_last_stop_publish = time.monotonic()

        def publish_metric(self, event_type: str, **kwargs: Any) -> None:
            payload = {
                "event_type": event_type,
                "model": "go2_benchmark_adapter",
                "episode_id": self.active_episode_id,
                "task_id": self.task.get("task_id"),
                "scene_id": self.scene.get("scene_id"),
                "timestamp": self.get_clock().now().nanoseconds / 1e9,
            }
            payload.update(kwargs)
            self.metric_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

        def current_objects(self) -> list[dict[str, Any]]:
            return [dict(obj) for obj in self.scene.get("objects", [])]

        def current_dynamic_obstacles(self) -> list[dict[str, Any]]:
            elapsed = time.monotonic() - self.episode_started
            result = []
            for obstacle in self.scene.get("dynamic_obstacles", []):
                item = dict(obstacle)
                path = item.get("path", [])
                if len(path) >= 2:
                    p0, p1 = path[0], path[1]
                    seg_len = max(1.0e-6, math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])))
                    speed = max(0.01, float(item.get("speed_mps", 0.2)))
                    phase = (elapsed * speed / seg_len) % 2.0
                    frac = phase if phase <= 1.0 else 2.0 - phase
                    x = float(p0[0]) + (float(p1[0]) - float(p0[0])) * frac
                    y = float(p0[1]) + (float(p1[1]) - float(p0[1])) * frac
                    item["pose"] = [x, y, 0.9]
                result.append(item)
            return result

        def current_obstacles(self) -> list[dict[str, Any]]:
            obstacles = [dict(item) for item in self.scene.get("obstacles", [])]
            for item in self.current_dynamic_obstacles():
                obstacles.append(
                    {
                        "id": item.get("id"),
                        "class": item.get("class", "dynamic_obstacle"),
                        "pose": item.get("pose", [0.0, 0.0, 0.0]),
                        "size": [2.0 * float(item.get("radius", 0.45))] * 3,
                        "dynamic": True,
                    }
                )
            return obstacles

        def safety_status(self, obstacles: list[dict[str, Any]]) -> dict[str, Any]:
            pose = self.pose.as_list()
            fallen = root_height_unstable(self.telemetry_z, self.fallen_root_z_m)
            distances = [distance_xy(pose, item.get("pose", [999.0, 999.0, 0.0])) for item in obstacles]
            obstacle_distance = min(distances) if distances else None
            path_status = path_obstacle_status(
                pose,
                obstacles,
                stop_distance_m=self.local_stop_distance_m,
                path_half_width_m=self.local_path_half_width_m,
            )
            human_distances = [
                distance_xy(pose, item.get("pose", [999.0, 999.0, 0.0]))
                for item in obstacles
                if item.get("class") == "human_dummy" or str(item.get("id", "")).startswith("human")
            ]
            human_distance = min(human_distances) if human_distances else None
            near_classes = []
            for zone in self.scene.get("semantic_zones", []):
                if distance_xy(pose, zone.get("center", [999.0, 999.0])) <= float(zone.get("radius", 0.0)):
                    near_classes.append(zone.get("class"))
            collision = obstacle_distance is not None and obstacle_distance < 0.25
            return {
                "timestamp": self.get_clock().now().nanoseconds / 1e9,
                "local_costmap_clear": bool(path_status["local_costmap_clear"]),
                "obstacle_distance_m": None if obstacle_distance is None else round(obstacle_distance, 3),
                "path_obstacle_distance_m": None
                if path_status["path_obstacle_distance_m"] is None
                else round(float(path_status["path_obstacle_distance_m"]), 3),
                "human_distance_m": None if human_distance is None else round(human_distance, 3),
                "on_slope_or_stairs": False,
                "near_doorway": "doorway" in near_classes,
                "near_intersection": "intersection" in near_classes,
                "dynamic_obstacle": human_distance is not None and human_distance < 2.0,
                "estop": False,
                "deadman": True,
                "robot_fallen_or_unstable": bool(fallen),
                "battery_ok": True,
                "collision": collision,
            }

    rclpy.init()
    node = Go2BenchmarkAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        shutdown_error = exc.__class__.__name__ in {"ExternalShutdownException", "RCLError"} or "context is not valid" in repr(exc)
        if not shutdown_error:
            raise
    finally:
        if getattr(node, "telemetry_sock", None) is not None:
            node.telemetry_sock.close()
        if getattr(node, "control_sock", None) is not None:
            node.control_sock.close()
        if getattr(node, "twist_sock", None) is not None:
            node.twist_sock.close()
        if getattr(node, "camera_sock", None) is not None:
            node.camera_sock.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
def merge_scene_entities(existing: list[dict[str, Any]], overlays: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Overlay entities by id while preserving benchmark judge geometry."""
    overlay_ids = {str(row.get("id") or "") for row in overlays if row.get("id")}
    return [dict(row) for row in existing if str(row.get("id") or "") not in overlay_ids] + [
        dict(row) for row in overlays
    ]
