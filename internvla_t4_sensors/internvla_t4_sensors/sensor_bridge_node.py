"""T4 online-mapping bridge layered on the frozen T3 safety controller."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import struct
import threading
import time
import zlib
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from internvla_go2_controller import bridge_node as t3_bridge
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.clock import Clock as RclpyClock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2, PointField
from std_msgs.msg import Bool, Float64MultiArray, Int32, String

from .depth_codec import decode_depth_request


_BRIDGE_LINK_CENTER_NAMES = (
    "base",
    "FL_thigh",
    "FL_calf",
    "FL_foot",
    "FR_thigh",
    "FR_calf",
    "FR_foot",
    "RL_thigh",
    "RL_calf",
    "RL_foot",
    "RR_thigh",
    "RR_calf",
    "RR_foot",
)


def _project_bridge_link_centers(
    raw_centers: list[object],
) -> list[tuple[float, float, float]]:
    """Project the full articulation inventory onto the frozen 13-center wire set.

    The navigation-only planar profile still exposes all 19 rigid bodies so the
    local depth self-filter can use them.  The downstream bridge wire contract,
    however, is intentionally frozen to the base plus each thigh, calf, and
    foot.  Select that named subset here instead of depending on articulation
    iteration order or publishing a malformed 19-pose array.
    """

    selected: dict[str, tuple[float, float, float]] = {}
    required = set(_BRIDGE_LINK_CENTER_NAMES)
    for item in raw_centers:
        if not isinstance(item, dict):
            raise ValueError("invalid robot link center evidence")
        name = item.get("name")
        center = item.get("center_base")
        if not isinstance(name, str) or not name:
            raise ValueError("robot link center evidence has no name")
        if not isinstance(center, list) or len(center) != 3:
            raise ValueError("invalid robot link center evidence")
        values = tuple(float(value) for value in center)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite robot link center evidence")
        if name not in required:
            continue
        if name in selected:
            raise ValueError(f"duplicate required robot link center: {name}")
        selected[name] = values
    missing = [name for name in _BRIDGE_LINK_CENTER_NAMES if name not in selected]
    if missing:
        raise ValueError(
            "required robot link centers are missing: " + ",".join(missing)
        )
    return [selected[name] for name in _BRIDGE_LINK_CENTER_NAMES]


def _lane_identity_prefix() -> str:
    """Return the configured T5 Lane prefix while preserving T4's empty default."""

    prefix = os.environ.get("INTERNNAV_T5_ID_PREFIX", "")
    if prefix not in {"", "a::", "b::"}:
        raise RuntimeError("INTERNNAV_T5_ID_PREFIX must be empty, a::, or b::")
    return prefix


def _t5_revc_sensor_extensions_enabled() -> bool:
    """Keep Rev-C/sim-IMU payload handling out of T4 and real-robot modes."""

    raw_flag = os.environ.get("INTERNVLA_T5_REVC_ENABLE", "0")
    if raw_flag not in {"0", "1"}:
        raise RuntimeError("INTERNVLA_T5_REVC_ENABLE must be exactly 0 or 1")
    if raw_flag == "0":
        return False
    lane = os.environ.get("INTERNNAV_T5_LANE", "")
    exact_scope = (
        os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
        and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
        and lane in {"a", "b"}
        and os.environ.get("INTERNNAV_T5_ID_PREFIX", "") == f"{lane}::"
    )
    if not exact_scope:
        raise RuntimeError(
            "T5 Rev-C bridge requires exact completion_sim/isaac/lane/prefix scope"
        )
    return True


def _stereo_feed_enabled(
    *,
    requested: bool,
    pose_source: str,
    lane_identity_prefix: str,
) -> bool:
    """Resolve the opt-in T5 stereo shadow without changing frozen T4.

    T4 historically enabled its odometry stereo pair whenever
    ``pose_source=external_odometry``.  Preserve that empty-Lane behavior.
    A T5 Lane instead requires an explicit parameter so it can publish the
    stereo pair while GT remains the sole navigation pose source.
    """

    if not lane_identity_prefix:
        return pose_source == "external_odometry"
    if lane_identity_prefix not in {"a::", "b::"}:
        raise RuntimeError("invalid T5 Lane identity prefix")
    if not requested:
        return False
    lane = lane_identity_prefix[0]
    exact_scope = (
        os.environ.get("INTERNNAV_RUNTIME_POLICY", "") == "completion_sim"
        and os.environ.get("INTERNNAV_SIMULATION_TARGET", "") == "isaac"
        and os.environ.get("INTERNNAV_T5_LANE", "") == lane
        and os.environ.get("INTERNNAV_T5_ID_PREFIX", "") == lane_identity_prefix
    )
    if not exact_scope:
        raise RuntimeError(
            "T5 stereo feed requires exact completion_sim/isaac/lane/prefix scope"
        )
    return True


def _static_map_dataset_episode_id(
    runtime_episode_id: str,
    identity_prefix: str,
    *,
    state_only: bool = False,
) -> str:
    """Map a Lane-scoped runtime identity to the frozen dataset identity.

    Runtime identities stay Lane-prefixed end to end.  Only the immutable
    static-map lookup uses the unprefixed dataset key.  An empty prefix is the
    frozen T4 behavior; a configured T5 Lane rejects unscoped and foreign-Lane
    identities instead of accidentally selecting their map.
    """

    episode_id = str(runtime_episode_id)
    if not episode_id:
        raise ValueError("episode identity is empty")
    if identity_prefix not in {"", "a::", "b::"}:
        raise ValueError("invalid Lane identity prefix")
    if not identity_prefix:
        return episode_id
    if (
        state_only
        and episode_id.startswith("bootstrap-episode-")
        and episode_id.removeprefix("bootstrap-episode-").isdigit()
    ):
        # Isaac publishes motion-disabled warm-up state before the evaluator
        # binds the first Lane-scoped episode.  _publish_static_map recognizes
        # this exact sentinel and defers selection; it is never a dataset key.
        return episode_id
    if not episode_id.startswith(identity_prefix):
        raise ValueError("episode identity does not belong to this Lane")
    dataset_episode_id = episode_id[len(identity_prefix) :]
    if not dataset_episode_id:
        raise ValueError("Lane-prefixed episode identity has no dataset id")
    return dataset_episode_id


def _rotation_to_quaternion(
    matrix: tuple[tuple[float, float, float], ...]
) -> tuple[float, float, float, float]:
    """Return xyzw for a proper 3x3 rotation matrix."""
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
            0.25 * scale,
        )
    diagonal = [matrix[0][0], matrix[1][1], matrix[2][2]]
    index = max(range(3), key=diagonal.__getitem__)
    if index == 0:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        return (
            0.25 * scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
            (matrix[2][1] - matrix[1][2]) / scale,
        )
    if index == 1:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        return (
            (matrix[0][1] + matrix[1][0]) / scale,
            0.25 * scale,
            (matrix[1][2] + matrix[2][1]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
        )
    scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
    return (
        (matrix[0][2] + matrix[2][0]) / scale,
        (matrix[1][2] + matrix[2][1]) / scale,
        0.25 * scale,
        (matrix[1][0] - matrix[0][1]) / scale,
    )


class T4SensorBridge(t3_bridge.Go2ControllerBridge):
    """Publish calibrated depth while denying every T3 static-map write."""

    def __init__(self) -> None:
        # The frozen T3 base constructor persists an initial summary and calls
        # this virtual method before T4 parameters and counters exist.  Keep
        # that bootstrap write strictly on the frozen schema, then persist the
        # complete T4 schema once this constructor has finished.
        self._t4_initialized = False
        self.identity_safe_stop_count = 0
        self.stale_motion_execution_count = 0
        self.lane_identity_prefix = _lane_identity_prefix()
        super().__init__()
        self.declare_parameter("camera_height_above_support_m", 0.62)
        self.declare_parameter("base_height_above_support_m", 0.42)
        self.declare_parameter("camera_forward_m", 0.20)
        self.declare_parameter("camera_pitch_down_deg", 20.0)
        self.declare_parameter("camera_hfov_deg", 69.4)
        self.declare_parameter("camera_vfov_deg", 42.5)
        self.declare_parameter("depth_height_above_support_m", 0.62)
        self.declare_parameter("depth_forward_m", 0.20)
        self.declare_parameter("depth_pitch_down_deg", 20.0)
        self.declare_parameter("depth_hfov_deg", 87.0)
        self.declare_parameter("depth_vfov_deg", 58.0)
        self.declare_parameter("depth_minimum_m", 0.28)
        self.declare_parameter("camera_frame", "go2_depth_optical_frame")
        self.declare_parameter("map_source", "nvblox_online")
        self.declare_parameter("pose_source", "ground_truth")
        self.declare_parameter("enable_stereo_feed", False)
        self.declare_parameter(
            "external_odometry_topic", "/visual_slam/tracking/odometry"
        )
        self.declare_parameter("external_odometry_timeout_sec", 0.30)
        self.declare_parameter("stereo_baseline_m", 0.12)
        self.declare_parameter("stereo_height_above_support_m", 0.62)
        self.declare_parameter("stereo_forward_m", 0.20)
        self.declare_parameter("stereo_pitch_down_deg", 10.0)
        self.declare_parameter("stereo_hfov_deg", 90.0)
        self.camera_height = float(
            self.get_parameter("camera_height_above_support_m").value
        )
        self.base_height = float(
            self.get_parameter("base_height_above_support_m").value
        )
        self.camera_forward = float(self.get_parameter("camera_forward_m").value)
        self.pitch_down = float(self.get_parameter("camera_pitch_down_deg").value)
        self.hfov = float(self.get_parameter("camera_hfov_deg").value)
        self.vfov = float(self.get_parameter("camera_vfov_deg").value)
        self.depth_height = float(
            self.get_parameter("depth_height_above_support_m").value
        )
        self.depth_forward = float(self.get_parameter("depth_forward_m").value)
        self.depth_pitch_down = float(
            self.get_parameter("depth_pitch_down_deg").value
        )
        self.depth_hfov = float(self.get_parameter("depth_hfov_deg").value)
        self.depth_vfov = float(self.get_parameter("depth_vfov_deg").value)
        self.depth_minimum = float(self.get_parameter("depth_minimum_m").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.map_source = str(self.get_parameter("map_source").value)
        self.pose_source = str(self.get_parameter("pose_source").value)
        self.enable_stereo_feed = _stereo_feed_enabled(
            requested=bool(self.get_parameter("enable_stereo_feed").value),
            pose_source=self.pose_source,
            lane_identity_prefix=self.lane_identity_prefix,
        )
        self.external_odometry_topic = str(
            self.get_parameter("external_odometry_topic").value
        )
        self.external_odometry_timeout = float(
            self.get_parameter("external_odometry_timeout_sec").value
        )
        self.stereo_baseline = float(self.get_parameter("stereo_baseline_m").value)
        self.stereo_height = float(
            self.get_parameter("stereo_height_above_support_m").value
        )
        self.stereo_forward = float(self.get_parameter("stereo_forward_m").value)
        self.stereo_pitch_down = float(
            self.get_parameter("stereo_pitch_down_deg").value
        )
        self.stereo_hfov = float(self.get_parameter("stereo_hfov_deg").value)
        if self.map_source not in {"nvblox_online", "static_map"}:
            raise RuntimeError("map_source must be nvblox_online or static_map")
        if self.pose_source not in {"ground_truth", "external_odometry"}:
            raise RuntimeError("pose_source must be ground_truth or external_odometry")
        if self.external_odometry_timeout <= 0.0:
            raise RuntimeError("external odometry timeout must be positive")
        # The parent creates this publisher as part of the frozen T3 graph.
        # Destroy it as well as overriding the call site, so T4.2 has no `/map`
        # publisher to accidentally satisfy Nav2 from a latent/static source.
        if self.map_source == "nvblox_online":
            if not self.destroy_publisher(self.static_map_publisher):
                raise RuntimeError("failed to remove inherited static-map publisher")
        if not (
            0.5 <= self.camera_height <= 1.35
            and 0.0 <= self.camera_forward <= 0.4
            and 0.0 <= self.pitch_down <= 60.0
            and 60.0 <= self.hfov <= 120.0
            and 30.0 <= self.vfov <= 90.0
            and 0.5 <= self.depth_height <= 1.35
            and 0.0 <= self.depth_forward <= 0.4
            and 0.0 <= self.depth_pitch_down <= 60.0
            and 60.0 <= self.depth_hfov <= 120.0
            and 30.0 <= self.depth_vfov <= 90.0
            and 0.1 <= self.depth_minimum <= 1.0
        ):
            raise RuntimeError("invalid audited camera calibration")
        self.camera_translation = (
            self.camera_forward,
            0.0,
            self.camera_height - self.base_height,
        )
        self.depth_translation = (
            self.depth_forward,
            0.0,
            self.depth_height - self.base_height,
        )
        pitch = math.radians(self.depth_pitch_down)
        # Camera-local coordinates used by the frozen T3 point-cloud filter:
        # +X image-right, +Y image-up, -Z view direction.
        camera_to_base = (
            (0.0, math.sin(pitch), -math.cos(pitch)),
            (-1.0, 0.0, 0.0),
            (0.0, math.cos(pitch), math.sin(pitch)),
        )
        self.depth_camera_to_base = camera_to_base
        t3_bridge.CAMERA_TRANSLATION = self.depth_translation
        t3_bridge.CAMERA_ROTATION = camera_to_base
        self.depth_publisher = self.create_publisher(
            Image,
            "/go2/depth/image_rect",
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=4,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self.camera_info_publisher = self.create_publisher(
            CameraInfo,
            "/go2/depth/camera_info",
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=4,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=4,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        # R3 keeps simulator transport private and exposes only standard ROS 2
        # sensor messages to the independent Go2 bridge.
        self.d435i_depth_publisher = self.create_publisher(
            Image, "/go2/d435i/depth/image_rect", sensor_qos
        )
        self.d435i_depth_info_publisher = self.create_publisher(
            CameraInfo, "/go2/d435i/depth/camera_info", sensor_qos
        )
        self.d435i_rgb_publisher = self.create_publisher(
            Image, "/go2/d435i/color/image_raw", sensor_qos
        )
        self.d435i_rgb_info_publisher = self.create_publisher(
            CameraInfo, "/go2/d435i/color/camera_info", sensor_qos
        )
        self.raw_lidar_publisher = self.create_publisher(
            PointCloud2, "/internvla_t4/go2/lidar_raw", sensor_qos
        )
        self.raw_front_rgb_publisher = self.create_publisher(
            Image, "/internvla_t4/go2/front_rgb_raw", sensor_qos
        )
        self.raw_front_info_publisher = self.create_publisher(
            CameraInfo, "/internvla_t4/go2/front_rgb_camera_info_raw", sensor_qos
        )
        self.raw_imu_publisher = self.create_publisher(
            Imu, "/internvla_t4/go2/imu_raw", sensor_qos
        )
        self.raw_link_centers_publisher = self.create_publisher(
            PoseArray, "/internvla_t4/go2/self_filter_link_centers", sensor_qos
        )
        self.pose_valid_publisher = self.create_publisher(
            Bool, "/internvla_t4/pose_valid", 10
        )
        self.collision_monitor_clear_publisher = self.create_publisher(
            Bool, "/internvla_t4/collision_monitor_clear", 10
        )
        self.r3_lidar_frame_count = 0
        self.r3_front_rgb_frame_count = 0
        self.r3_d435i_rgb_frame_count = 0
        self.r3_imu_frame_count = 0
        self.map_reset_publisher = self.create_publisher(
            Int32,
            "/internvla_t4/map_reset_generation",
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self.create_subscription(
            String,
            "/internvla_t4/episode_prime",
            self._on_episode_prime,
            10,
        )
        self.suppressed_static_map_calls = 0
        self.last_reset_generation = -1
        self._pending_episode_id = ""
        self._pending_dataset_episode_id = ""
        self._active_start_map_pose: tuple[float, float, float] | None = None
        self._external_lock = threading.RLock()
        self._external_odometry: Odometry | None = None
        self._external_odometry_monotonic = 0.0
        self.external_odometry_received_count = 0
        self.external_odometry_stale_stop_count = 0
        self.external_odometry_nav_publish_count = 0
        self.map_ready_generation = -1
        self.map_not_ready_stop_count = 0
        self._truth_start_by_generation: dict[int, tuple[float, float, float]] = {}
        self._last_odometry_evaluation_monotonic = 0.0
        self.odometry_evaluation_path = (
            Path(self.result_dir) / "odometry_evaluation_records.jsonl"
        )
        if self.odometry_evaluation_path.exists():
            raise FileExistsError(self.odometry_evaluation_path)
        if self.pose_source == "external_odometry":
            self.create_subscription(
                Odometry,
                self.external_odometry_topic,
                self._on_external_odometry,
                QoSProfile(
                    history=HistoryPolicy.KEEP_LAST,
                    depth=10,
                    reliability=ReliabilityPolicy.RELIABLE,
                ),
            )
        if self.map_source == "nvblox_online":
            self.create_subscription(
                Int32,
                "/internvla_t4/map_ready_generation",
                self._on_map_ready,
                10,
            )
        self.metric_depth_frame_count = 0
        self.metric_depth_valid_count = 0
        self.metric_depth_transport_latency_ms: list[float] = []
        self.stereo_frame_count = 0
        self.sensor_audit_path = Path(self.result_dir) / "sensor_frames.jsonl"
        if self.sensor_audit_path.exists():
            raise FileExistsError(self.sensor_audit_path)
        self.costmap_class_audit_path = (
            Path(self.result_dir) / "nvblox_costmap_classes.jsonl"
        )
        if self.costmap_class_audit_path.exists():
            raise FileExistsError(self.costmap_class_audit_path)
        self._last_costmap_class_audit_monotonic: dict[str, float] = {
            "local": 0.0,
            "global": 0.0,
        }
        self.costmap_class_sample_count = 0
        self.costmap_class_sample_count_by_source = {"local": 0, "global": 0}
        self.costmap_class_generations: dict[str, set[int]] = {
            "free": set(),
            "occupied": set(),
            "unknown": set(),
        }
        self.create_subscription(
            OccupancyGrid,
            "global_costmap/costmap",
            lambda message: self._audit_costmap(message, "global"),
            10,
        )

        # ROS optical coordinates: +X right, +Y down, +Z forward.
        optical_to_base = (
            (0.0, -math.sin(pitch), math.cos(pitch)),
            (-1.0, 0.0, 0.0),
            (0.0, -math.cos(pitch), -math.sin(pitch)),
        )
        qx, qy, qz, qw = _rotation_to_quaternion(optical_to_base)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "base_link"
        transform.child_frame_id = self.camera_frame
        transform.transform.translation.x = self.depth_translation[0]
        transform.transform.translation.y = self.depth_translation[1]
        transform.transform.translation.z = self.depth_translation[2]
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.static_tf_broadcaster.sendTransform(transform)
        # Explicit R3 fixed extrinsics.  LiDAR points stay in their physical
        # sensor frame; downstream consumers transform them through this TF.
        for child, translation in (
            ("go2_l1_lidar", (0.25, 0.0, 0.18)),
            ("go2_imu_link", (0.0, 0.0, 0.0)),
        ):
            fixed = TransformStamped()
            fixed.header.stamp = self.get_clock().now().to_msg()
            fixed.header.frame_id = "base_link"
            fixed.child_frame_id = child
            fixed.transform.translation.x = translation[0]
            fixed.transform.translation.y = translation[1]
            fixed.transform.translation.z = translation[2]
            fixed.transform.rotation.w = 1.0
            self.static_tf_broadcaster.sendTransform(fixed)
        for child, translation, pitch_degrees in (
            (
                "go2_d435i_color_optical_frame",
                self.camera_translation,
                self.pitch_down,
            ),
            ("go2_front_rgb_optical_frame", (0.29, 0.0, -0.06), 8.0),
        ):
            sensor_pitch = math.radians(pitch_degrees)
            optical_to_base_sensor = (
                (0.0, -math.sin(sensor_pitch), math.cos(sensor_pitch)),
                (-1.0, 0.0, 0.0),
                (0.0, -math.cos(sensor_pitch), -math.sin(sensor_pitch)),
            )
            sx, sy, sz, sw = _rotation_to_quaternion(optical_to_base_sensor)
            fixed = TransformStamped()
            fixed.header.stamp = self.get_clock().now().to_msg()
            fixed.header.frame_id = "base_link"
            fixed.child_frame_id = child
            fixed.transform.translation.x = translation[0]
            fixed.transform.translation.y = translation[1]
            fixed.transform.translation.z = translation[2]
            fixed.transform.rotation.x = sx
            fixed.transform.rotation.y = sy
            fixed.transform.rotation.z = sz
            fixed.transform.rotation.w = sw
            self.static_tf_broadcaster.sendTransform(fixed)
        self.stereo_publishers: dict[str, tuple[Any, Any]] = {}
        if self.enable_stereo_feed:
            if not (
                0.06 <= self.stereo_baseline <= 0.30
                and 0.50 <= self.stereo_height <= 1.20
                and 0.0 <= self.stereo_forward <= 0.40
                and 0.0 <= self.stereo_pitch_down <= 30.0
                and 60.0 <= self.stereo_hfov <= 120.0
            ):
                raise RuntimeError("invalid audited stereo calibration")
            stereo_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=8,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            for side in ("left", "right"):
                self.stereo_publishers[side] = (
                    self.create_publisher(
                        Image, f"/go2/stereo/{side}/image_rect", stereo_qos
                    ),
                    self.create_publisher(
                        CameraInfo, f"/go2/stereo/{side}/camera_info", stereo_qos
                    ),
                )
            stereo_pitch = math.radians(self.stereo_pitch_down)
            stereo_optical_to_base = (
                (0.0, -math.sin(stereo_pitch), math.cos(stereo_pitch)),
                (-1.0, 0.0, 0.0),
                (0.0, -math.cos(stereo_pitch), -math.sin(stereo_pitch)),
            )
            sqx, sqy, sqz, sqw = _rotation_to_quaternion(stereo_optical_to_base)
            for side, lateral in (
                ("left", self.stereo_baseline / 2.0),
                ("right", -self.stereo_baseline / 2.0),
            ):
                stereo_tf = TransformStamped()
                stereo_tf.header.stamp = self.get_clock().now().to_msg()
                stereo_tf.header.frame_id = "base_link"
                stereo_tf.child_frame_id = f"go2_stereo_{side}_optical_frame"
                stereo_tf.transform.translation.x = self.stereo_forward
                stereo_tf.transform.translation.y = lateral
                stereo_tf.transform.translation.z = self.stereo_height - self.base_height
                stereo_tf.transform.rotation.x = sqx
                stereo_tf.transform.rotation.y = sqy
                stereo_tf.transform.rotation.z = sqz
                stereo_tf.transform.rotation.w = sqw
                self.static_tf_broadcaster.sendTransform(stereo_tf)
        self._bootstrap_tf_lock = threading.Lock()
        self._real_pose_published = False
        self.bootstrap_tf_publish_count = 0
        if self.lane_identity_prefix:
            # T5 keeps data stamps on simulation time, but its sole /clock
            # source starts only after the complete DGX Lane is healthy.  A
            # steady control timer lets this motion-disabled identity TF
            # unblock Nav2 lifecycle activation without creating a clock
            # source or changing the transform's ROS-time stamp.
            self._bootstrap_timer_clock = RclpyClock(
                clock_type=ClockType.STEADY_TIME
            )
            self.create_timer(
                0.05,
                self._publish_bootstrap_tf,
                clock=self._bootstrap_timer_clock,
            )
        else:
            # Empty Lane identity is the frozen T4 path.
            self.create_timer(0.05, self._publish_bootstrap_tf)
        self._t4_initialized = True
        self._write_summary("READY")

    def _publish_bootstrap_tf(self) -> None:
        """Keep Nav2 configurable before Isaac supplies the first safe state.

        No client or map-ready signal exists at this point, so this identity TF
        can only unblock lifecycle activation; motion remains fail-closed.
        """
        with self._bootstrap_tf_lock:
            if self._real_pose_published:
                return
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "odom"
            transform.child_frame_id = "base_link"
            transform.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(transform)
            self.bootstrap_tf_publish_count += 1

    def _handle_update(self, request: dict[str, Any]) -> dict[str, Any]:
        runtime_episode_id = str(request.get("episode_id", ""))
        dataset_episode_id = _static_map_dataset_episode_id(
            runtime_episode_id,
            self.lane_identity_prefix,
            state_only=bool(request.get("state_only", False)),
        )
        self._pending_episode_id = runtime_episode_id
        self._pending_dataset_episode_id = dataset_episode_id
        stale_before = self.stale_count
        response = super()._handle_update(request)
        if self.pose_source == "external_odometry":
            with self._external_lock:
                age = time.monotonic() - self._external_odometry_monotonic
                fresh = (
                    self._external_odometry is not None
                    and age <= self.external_odometry_timeout
                )
            if not fresh and bool(response.get("motion_enabled", False)):
                response["linear_x"] = 0.0
                response["angular_z"] = 0.0
                response["emergency_stop"] = True
                response["external_odometry_stale"] = True
                self.external_odometry_stale_stop_count += 1
        if (
            self.map_source == "nvblox_online"
            and self.map_ready_generation != int(request.get("reset_generation", -1))
            and bool(response.get("motion_enabled", False))
        ):
            response["linear_x"] = 0.0
            response["angular_z"] = 0.0
            response["emergency_stop"] = True
            response["online_map_not_ready"] = True
            self.map_not_ready_stop_count += 1
        rejected = max(0, self.stale_count - stale_before)
        if rejected:
            stale_motion = bool(response.get("motion_enabled", False)) or any(
                abs(float(response.get(name, 0.0))) > 1.0e-9
                for name in ("linear_x", "angular_z")
            )
            if stale_motion:
                self.stale_motion_execution_count += rejected
            else:
                self.identity_safe_stop_count += rejected
        pose = request.get("pose_wxyz", [])
        pose_valid = (
            isinstance(pose, list)
            and len(pose) == 7
            and all(math.isfinite(float(value)) for value in pose)
            and not bool(request.get("nan_detected", False))
            and not bool(request.get("fallen", False))
        )
        pose_message = Bool()
        pose_message.data = bool(pose_valid)
        self.pose_valid_publisher.publish(pose_message)
        monitor_message = Bool()
        monitor_message.data = bool(
            pose_valid
            and not bool(request.get("physical_collision", False))
            and not bool(getattr(self, "_monitor_was_stopping", False))
        )
        self.collision_monitor_clear_publisher.publish(monitor_message)
        return response

    def _on_map_ready(self, message: Int32) -> None:
        generation = int(message.data)
        if generation >= self.map_ready_generation:
            self.map_ready_generation = generation

    def _on_episode_prime(self, message: String) -> None:
        """Select the static map before the first model/Nav2 resolution call.

        The message carries dataset identity only. No evaluator pose is used;
        map selection remains the frozen episode-id lookup used by Oracle.
        """

        value = json.loads(message.data)
        if int(value.get("schema_version", 0)) != 1:
            raise ValueError("unsupported episode-prime schema")
        episode_id = str(value.get("episode_id", ""))
        generation = int(value.get("reset_generation", -1))
        if not episode_id or generation < 0:
            raise ValueError("invalid episode-prime identity")
        dataset_episode_id = _static_map_dataset_episode_id(
            episode_id, self.lane_identity_prefix
        )
        semantic_now = t3_bridge._semantic_now(self)
        with self.lock:
            self._state_replay_epoch += 1
            self.active_identity = (episode_id, generation, -1)
            self.navigation_barrier_monotonic = semantic_now
            self.navigation_barrier_cmd_serial = self.safe_cmd_serial
            self.motion_enabled = False
            self.stop_latched = True
            self.last_safe_cmd = (0.0, 0.0)
            self.last_safe_cmd_monotonic = 0.0
            self.latest_state = None
            self.latest_state_monotonic = 0.0
            self.latest_pointcloud = None
            self.latest_pointcloud_monotonic = 0.0
            self.active_path = []
            self.latest_detected_targets_map = []
            self.latest_detected_target_monotonic = 0.0
            self.latest_detected_target_token = -1
            self.latest_detected_target_generation = generation
            self.latest_detected_target_episode = episode_id
        self._pending_episode_id = episode_id
        self._pending_dataset_episode_id = dataset_episode_id
        self._publish_static_map(generation, [])

    def _publish_static_map(
        self,
        generation: int,
        pose: list[float],
        *,
        identity: tuple[str, int, int] | None = None,
        replay_epoch: int | None = None,
    ) -> None:
        """Suppress static maps for T4.2 or select them without runtime truth in T4.3."""
        del pose, identity, replay_epoch
        if generation != self.last_reset_generation:
            message = Int32()
            message.data = int(generation)
            self.map_reset_publisher.publish(message)
            self.last_reset_generation = int(generation)
        if self.map_source == "static_map":
            if self._pending_episode_id.startswith("bootstrap-episode-"):
                # The frozen T3 bootstrap deliberately hides the upcoming task
                # identity. Defer map selection until the typed episode arrives
                # rather than consulting the runtime ground-truth pose.
                return
            with self.lock:
                if generation == self.static_map_generation:
                    return
            matches = [
                item
                for item in self.static_map_episode_entries
                if str(item.get("episode_id", ""))
                == self._pending_dataset_episode_id
            ]
            if len(matches) != 1:
                raise ValueError(
                    "static-map selection requires one dataset episode match: "
                    f"runtime_episode={self._pending_episode_id!r} "
                    f"dataset_episode={self._pending_dataset_episode_id!r} "
                    f"matches={len(matches)}"
                )
            episode_entry = matches[0]
            if "start_map_yaw_rad" not in episode_entry:
                raise ValueError("truth-isolated static manifest lacks start_map_yaw_rad")
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
            self.static_map_publisher.publish(message)
            start_xy = episode_entry["start_map_xy"]
            self._active_start_map_pose = (
                float(start_xy[0]),
                float(start_xy[1]),
                float(episode_entry["start_map_yaw_rad"]),
            )
            with self.lock:
                self.static_map_generation = generation
                self.static_map_publish_count += 1
                self.static_map_selections.append(
                    {
                        "generation": generation,
                        "map_key": map_key,
                        "trajectory_id": str(episode_entry.get("trajectory_id", "")),
                        "episode_id": self._pending_episode_id,
                        "dataset_episode_id": self._pending_dataset_episode_id,
                        "selection": "dataset_episode_id_no_runtime_pose",
                        "start_xy_error_m": 0.0,
                    }
                )
            return
        self.suppressed_static_map_calls += 1

    def _on_external_odometry(self, message: Odometry) -> None:
        with self._external_lock:
            self._external_odometry = message
            self._external_odometry_monotonic = time.monotonic()
            self.external_odometry_received_count += 1

    def _on_local_costmap(self, message: OccupancyGrid) -> None:
        super()._on_local_costmap(message)
        self._audit_costmap(message, "local")

    def _audit_costmap(self, message: OccupancyGrid, source: str) -> None:
        if self.map_source != "nvblox_online":
            return
        generation = int(self.last_reset_generation)
        if generation < 0:
            # Lifecycle activation publishes a bootstrap master costmap before
            # the first episode reset. It belongs to no map epoch and must not
            # count toward reset coverage or map-pollution evidence.
            return
        now = time.monotonic()
        if now - self._last_costmap_class_audit_monotonic[source] < 0.5:
            return
        data = [int(value) for value in message.data]
        if not data:
            return
        counts = {
            "unknown": sum(value < 0 for value in data),
            "free": sum(0 <= value < 50 for value in data),
            "inflated": sum(50 <= value < 90 for value in data),
            "occupied": sum(value >= 90 for value in data),
        }
        for name in ("free", "occupied", "unknown"):
            if counts[name] > 0:
                self.costmap_class_generations[name].add(generation)
        self.costmap_class_sample_count += 1
        self.costmap_class_sample_count_by_source[source] += 1
        self._last_costmap_class_audit_monotonic[source] = now
        record = {
            "schema_version": 1,
            "costmap_name": source,
            "reset_generation": generation,
            "frame_id": str(message.header.frame_id),
            "width": int(message.info.width),
            "height": int(message.info.height),
            "resolution_m": float(message.info.resolution),
            "cell_counts": counts,
            "wall_time_unix": time.time(),
        }
        with self.costmap_class_audit_path.open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def _publish_state(
        self,
        pose: list[float],
        linear_velocity: list[float],
        angular_velocity: list[float],
        path: list[tuple[float, float]],
    ) -> tuple[float, float]:
        stamp = self.get_clock().now().to_msg()
        self._t4_frame_stamp = stamp
        with self._bootstrap_tf_lock:
            self._real_pose_published = True
        if self.pose_source == "ground_truth":
            x, y, z, qw, qx, qy, qz = pose
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
            yaw = t3_bridge._yaw([qw, qx, qy, qz])
            cosine, sine = math.cos(yaw), math.sin(yaw)
            odom.twist.twist.linear.x = (
                cosine * linear_velocity[0] + sine * linear_velocity[1]
            )
            odom.twist.twist.linear.y = (
                -sine * linear_velocity[0] + cosine * linear_velocity[1]
            )
            odom.twist.twist.linear.z = linear_velocity[2]
            odom.twist.twist.angular.x = angular_velocity[0]
            odom.twist.twist.angular.y = angular_velocity[1]
            odom.twist.twist.angular.z = angular_velocity[2]
            self.odom_publisher.publish(odom)
            velocity = t3_bridge.TwistStamped()
            velocity.header.stamp = stamp
            velocity.header.frame_id = "base_link"
            velocity.twist = odom.twist.twist
            self.velocity_publisher.publish(velocity)
            lateral = 0.0
            heading = 0.0
            if path:
                nearest = min(
                    range(len(path)),
                    key=lambda index: math.hypot(
                        path[index][0] - x, path[index][1] - y
                    ),
                )
                lateral = math.hypot(
                    path[nearest][0] - x, path[nearest][1] - y
                )
                if len(path) > 1:
                    next_index = min(nearest + 1, len(path) - 1)
                    previous_index = max(0, next_index - 1)
                    path_yaw = math.atan2(
                        path[next_index][1] - path[previous_index][1],
                        path[next_index][0] - path[previous_index][0],
                    )
                    heading = math.atan2(
                        math.sin(path_yaw - yaw), math.cos(path_yaw - yaw)
                    )
            error = Float64MultiArray()
            error.data = [lateral, heading]
            self.error_publisher.publish(error)
            return lateral, heading
        truth_pose = list(pose)
        del linear_velocity, angular_velocity
        with self._external_lock:
            message = self._external_odometry
            age = time.monotonic() - self._external_odometry_monotonic
        if (
            message is None
            or age > self.external_odometry_timeout
            or (
                self.map_source == "static_map"
                and self._active_start_map_pose is None
            )
        ):
            return 0.0, 0.0
        relative = message.pose.pose
        relative_yaw = t3_bridge._yaw(
            [
                relative.orientation.w,
                relative.orientation.x,
                relative.orientation.y,
                relative.orientation.z,
            ]
        )
        if self.map_source == "static_map":
            assert self._active_start_map_pose is not None
            start_x, start_y, start_yaw = self._active_start_map_pose
        else:
            start_x, start_y, start_yaw = 0.0, 0.0, 0.0
        cosine, sine = math.cos(start_yaw), math.sin(start_yaw)
        x = start_x + cosine * relative.position.x - sine * relative.position.y
        y = start_y + sine * relative.position.x + cosine * relative.position.y
        yaw = start_yaw + relative_yaw
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = relative.position.z
        transform.transform.rotation.z = math.sin(yaw / 2.0)
        transform.transform.rotation.w = math.cos(yaw / 2.0)
        self.tf_broadcaster.sendTransform(transform)
        odom = Odometry()
        odom.header = transform.header
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = relative.position.z
        odom.pose.pose.orientation = transform.transform.rotation
        odom.pose.covariance = list(message.pose.covariance)
        odom.twist = message.twist
        self.odom_publisher.publish(odom)
        velocity = t3_bridge.TwistStamped()
        velocity.header.stamp = stamp
        velocity.header.frame_id = "base_link"
        velocity.twist = odom.twist.twist
        self.velocity_publisher.publish(velocity)
        lateral = 0.0
        heading = 0.0
        if path:
            nearest = min(
                range(len(path)),
                key=lambda index: math.hypot(path[index][0] - x, path[index][1] - y),
            )
            lateral = math.hypot(path[nearest][0] - x, path[nearest][1] - y)
            if len(path) > 1:
                next_index = min(nearest + 1, len(path) - 1)
                previous_index = max(0, next_index - 1)
                path_yaw = math.atan2(
                    path[next_index][1] - path[previous_index][1],
                    path[next_index][0] - path[previous_index][0],
                )
                heading = math.atan2(math.sin(path_yaw - yaw), math.cos(path_yaw - yaw))
        error = Float64MultiArray()
        error.data = [lateral, heading]
        self.error_publisher.publish(error)
        self.external_odometry_nav_publish_count += 1
        self._record_odometry_evaluation(truth_pose, x, y, yaw, message)
        return lateral, heading

    def _record_odometry_evaluation(
        self,
        truth_pose: list[float],
        estimated_x: float,
        estimated_y: float,
        estimated_yaw: float,
        odometry: Odometry,
    ) -> None:
        """Record simulator truth for scoring only; no value feeds the navigation graph."""
        now = time.monotonic()
        if now - self._last_odometry_evaluation_monotonic < 0.09:
            return
        generation = int(self.last_reset_generation)
        truth_yaw = t3_bridge._yaw(truth_pose[3:])
        anchor = self._truth_start_by_generation.setdefault(
            generation, (float(truth_pose[0]), float(truth_pose[1]), truth_yaw)
        )
        if self.map_source == "static_map":
            truth_x, truth_y, scored_truth_yaw = (
                float(truth_pose[0]),
                float(truth_pose[1]),
                truth_yaw,
            )
        else:
            dx = float(truth_pose[0]) - anchor[0]
            dy = float(truth_pose[1]) - anchor[1]
            cosine, sine = math.cos(anchor[2]), math.sin(anchor[2])
            truth_x = cosine * dx + sine * dy
            truth_y = -sine * dx + cosine * dy
            scored_truth_yaw = math.atan2(
                math.sin(truth_yaw - anchor[2]), math.cos(truth_yaw - anchor[2])
            )
        record = {
            "schema_version": 1,
            "episode_id": self._pending_episode_id,
            "reset_generation": generation,
            "monotonic_sec": now,
            "truth_usage": "evaluation_only_not_published_or_consumed_by_navigation",
            "truth_xy_yaw": [truth_x, truth_y, scored_truth_yaw],
            "estimated_xy_yaw": [estimated_x, estimated_y, estimated_yaw],
            "xy_error_m": math.hypot(estimated_x - truth_x, estimated_y - truth_y),
            "yaw_error_rad": abs(
                math.atan2(
                    math.sin(estimated_yaw - scored_truth_yaw),
                    math.cos(estimated_yaw - scored_truth_yaw),
                )
            ),
            "tf_age_sec": max(
                0.0,
                self.get_clock().now().nanoseconds / 1e9
                - (
                    float(odometry.header.stamp.sec)
                    + float(odometry.header.stamp.nanosec) / 1e9
                ),
            ),
            "wall_time_unix": time.time(),
        }
        with self.odometry_evaluation_path.open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._last_odometry_evaluation_monotonic = now

    def _publish_rgb8_payload(
        self,
        request: dict[str, Any],
        *,
        prefix: str,
        frame_id: str,
        hfov_deg: float,
        image_publisher: Any,
        info_publisher: Any,
    ) -> bool:
        width = int(request.get(f"{prefix}_width", 0))
        height = int(request.get(f"{prefix}_height", 0))
        encoded = request.get(f"{prefix}_rgb8_zlib_b64")
        if width <= 0 or height <= 0 or not isinstance(encoded, str) or not encoded:
            return False
        try:
            raw = zlib.decompress(base64.b64decode(encoded, validate=True))
        except (ValueError, zlib.error) as exc:
            raise ValueError(f"invalid {prefix} RGB payload") from exc
        if len(raw) != width * height * 3:
            raise ValueError(f"invalid {prefix} RGB frame size")
        stamp = getattr(self, "_t4_frame_stamp", self.get_clock().now().to_msg())
        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = frame_id
        image.height = height
        image.width = width
        image.encoding = "rgb8"
        image.is_bigendian = False
        image.step = width * 3
        image.data = raw
        focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        info = CameraInfo()
        info.header = image.header
        info.height = height
        info.width = width
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [focal, 0.0, width / 2.0, 0.0, focal, height / 2.0, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [
            focal,
            0.0,
            width / 2.0,
            0.0,
            0.0,
            focal,
            height / 2.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        image_publisher.publish(image)
        info_publisher.publish(info)
        return True

    def _publish_r3_sensors(self, request: dict[str, Any]) -> None:
        if self._publish_rgb8_payload(
            request,
            prefix="d435i_rgb",
            frame_id="go2_d435i_color_optical_frame",
            hfov_deg=self.hfov,
            image_publisher=self.d435i_rgb_publisher,
            info_publisher=self.d435i_rgb_info_publisher,
        ):
            self.r3_d435i_rgb_frame_count += 1
        if self._publish_rgb8_payload(
            request,
            prefix="go2_front_rgb",
            frame_id="go2_front_rgb_optical_frame",
            hfov_deg=120.0,
            image_publisher=self.raw_front_rgb_publisher,
            info_publisher=self.raw_front_info_publisher,
        ):
            self.r3_front_rgb_frame_count += 1

        width = int(request.get("go2_lidar_width", 0))
        height = int(request.get("go2_lidar_height", 0))
        values = request.get("go2_lidar_points_lidar")
        if width > 0 and height > 0 and isinstance(values, list):
            if len(values) != width * height:
                raise ValueError("invalid organized Go2 LiDAR payload")
            packed: list[bytes] = []
            dense = True
            for item in values:
                if item is None:
                    packed.append(struct.pack("<fff", math.nan, math.nan, math.nan))
                    dense = False
                    continue
                if not isinstance(item, list) or len(item) != 3:
                    raise ValueError("invalid Go2 LiDAR point")
                point = tuple(float(value) for value in item)
                if not all(math.isfinite(value) for value in point):
                    packed.append(struct.pack("<fff", math.nan, math.nan, math.nan))
                    dense = False
                    continue
                packed.append(struct.pack("<fff", *point))
            cloud = PointCloud2()
            cloud.header.stamp = getattr(
                self, "_t4_frame_stamp", self.get_clock().now().to_msg()
            )
            cloud.header.frame_id = "go2_l1_lidar"
            cloud.height = height
            cloud.width = width
            cloud.fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            cloud.is_bigendian = False
            cloud.point_step = 12
            cloud.row_step = 12 * width
            cloud.is_dense = dense
            cloud.data = b"".join(packed)
            self.raw_lidar_publisher.publish(cloud)
            self.r3_lidar_frame_count += 1

        centers = request.get("robot_link_centers_base", [])
        if isinstance(centers, list) and centers:
            bridge_centers = _project_bridge_link_centers(centers)
            array = PoseArray()
            array.header.stamp = getattr(
                self, "_t4_frame_stamp", self.get_clock().now().to_msg()
            )
            array.header.frame_id = "base_link"
            for center in bridge_centers:
                pose = Pose()
                pose.position.x = float(center[0])
                pose.position.y = float(center[1])
                pose.position.z = float(center[2])
                pose.orientation.w = 1.0
                array.poses.append(pose)
            self.raw_link_centers_publisher.publish(array)

        pose = request.get("pose_wxyz", [])
        angular = request.get("angular_velocity", [])
        if (
            isinstance(pose, list)
            and len(pose) == 7
            and isinstance(angular, list)
            and len(angular) == 3
        ):
            imu = Imu()
            imu.header.stamp = getattr(
                self, "_t4_frame_stamp", self.get_clock().now().to_msg()
            )
            imu.header.frame_id = "go2_imu_link"
            imu.orientation.w = float(pose[3])
            imu.orientation.x = float(pose[4])
            imu.orientation.y = float(pose[5])
            imu.orientation.z = float(pose[6])
            imu.angular_velocity.x = float(angular[0])
            imu.angular_velocity.y = float(angular[1])
            imu.angular_velocity.z = float(angular[2])
            imu.orientation_covariance[0] = 1.0e-4
            imu.orientation_covariance[4] = 1.0e-4
            imu.orientation_covariance[8] = 1.0e-4
            imu.angular_velocity_covariance[0] = 1.0e-3
            imu.angular_velocity_covariance[4] = 1.0e-3
            imu.angular_velocity_covariance[8] = 1.0e-3
            acceleration = request.get("go2_imu_linear_acceleration_mps2")
            acceleration_available = request.get(
                "go2_imu_linear_acceleration_available"
            )
            acceleration_stamp_ns = request.get(
                "go2_imu_linear_acceleration_sample_sim_stamp_ns"
            )
            request_sim_stamp_ns = request.get("t5_revc_sim_sample_stamp_ns")
            frame_stamp_ns = (
                int(imu.header.stamp.sec) * 1_000_000_000
                + int(imu.header.stamp.nanosec)
            )
            if (
                _t5_revc_sensor_extensions_enabled()
                and acceleration_available is True
                and isinstance(acceleration, list)
                and len(acceleration) == 3
                and all(math.isfinite(float(value)) for value in acceleration)
                and isinstance(acceleration_stamp_ns, int)
                and not isinstance(acceleration_stamp_ns, bool)
                and acceleration_stamp_ns > 0
                and isinstance(request_sim_stamp_ns, int)
                and not isinstance(request_sim_stamp_ns, bool)
                and acceleration_stamp_ns == request_sim_stamp_ns
                and acceleration_stamp_ns == frame_stamp_ns
            ):
                imu.header.stamp.sec = acceleration_stamp_ns // 1_000_000_000
                imu.header.stamp.nanosec = acceleration_stamp_ns % 1_000_000_000
                imu.linear_acceleration.x = float(acceleration[0])
                imu.linear_acceleration.y = float(acceleration[1])
                imu.linear_acceleration.z = float(acceleration[2])
                imu.linear_acceleration_covariance[0] = 1.0e-2
                imu.linear_acceleration_covariance[4] = 1.0e-2
                imu.linear_acceleration_covariance[8] = 1.0e-2
            else:
                # The generated simulator overlay deliberately marks the first
                # post-reset finite-difference sample unavailable.  This is a
                # sensor-contract precondition only; no LIO backend is enabled.
                imu.linear_acceleration_covariance[0] = -1.0
            self.raw_imu_publisher.publish(imu)
            self.r3_imu_frame_count += 1

    def _publish_metric_depth(self, request: dict[str, Any]) -> None:
        raw_values = request.get("depth_values")
        if not raw_values:
            return
        values = [
            value
            if math.isfinite(value := float(raw))
            and self.depth_minimum <= value <= 6.0
            else 0.0
            for raw in raw_values
        ]
        height = int(request.get("depth_height", 0))
        width = int(request.get("depth_width", 0))
        row_stride = int(request.get("depth_row_stride", 1))
        column_stride = int(request.get("depth_column_stride", 1))
        if height <= 0 or width <= 0 or len(values) != height * width:
            raise ValueError("invalid metric depth image")
        if row_stride != column_stride or row_stride <= 0:
            raise ValueError("metric depth sampling must use equal positive strides")
        stamp = getattr(self, "_t4_frame_stamp", self.get_clock().now().to_msg())
        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = self.camera_frame
        image.height = height
        image.width = width
        image.encoding = "32FC1"
        image.is_bigendian = False
        image.step = width * 4
        image.data = b"".join(struct.pack("<f", value) for value in values)
        full_fx = 320.0 / math.tan(math.radians(self.depth_hfov) / 2.0)
        full_fy = 240.0 / math.tan(math.radians(self.depth_vfov) / 2.0)
        fx = full_fx / column_stride
        fy = full_fy / row_stride
        cx = 319.5 / column_stride
        cy = 239.5 / row_stride
        info = CameraInfo()
        info.header = image.header
        info.height = height
        info.width = width
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.depth_publisher.publish(image)
        self.camera_info_publisher.publish(info)
        self.d435i_depth_publisher.publish(image)
        self.d435i_depth_info_publisher.publish(info)
        valid = sum(value > 0.0 for value in values)
        self.metric_depth_frame_count += 1
        self.metric_depth_valid_count += valid
        capture_wall_time = float(request.get("depth_capture_wall_time_unix", time.time()))
        transport_latency_ms = max(0.0, (time.time() - capture_wall_time) * 1000.0)
        self.metric_depth_transport_latency_ms.append(transport_latency_ms)
        if self.metric_depth_frame_count % 10 == 1:
            record = {
                "schema_version": 1,
                "frame_index": self.metric_depth_frame_count - 1,
                "reset_generation": int(request.get("reset_generation", -1)),
                "height": height,
                "width": width,
                "stride": row_stride,
                "valid_depth_count": valid,
                "valid_depth_fraction": valid / len(values),
                "camera_frame": self.camera_frame,
                "camera_model": "Intel RealSense D435i depth imager",
                "minimum_depth_m": self.depth_minimum,
                "hfov_deg": self.depth_hfov,
                "vfov_deg": self.depth_vfov,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "ros_stamp_ns": int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec),
                "pose_depth_sync_error_ms": 0.0,
                "transport_latency_ms": transport_latency_ms,
                "wall_time_unix": time.time(),
            }
            with self.sensor_audit_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def _publish_stereo(self, request: dict[str, Any]) -> None:
        if not self.enable_stereo_feed:
            return
        width = int(request.get("stereo_width", 0))
        height = int(request.get("stereo_height", 0))
        if width <= 0 or height <= 0:
            return
        decoded: dict[str, bytes] = {}
        for side in ("left", "right"):
            encoded = request.get(f"stereo_{side}_zlib_b64")
            if not isinstance(encoded, str) or not encoded:
                return
            try:
                raw = zlib.decompress(base64.b64decode(encoded, validate=True))
            except (ValueError, zlib.error) as exc:
                raise ValueError(f"invalid compressed stereo {side} frame") from exc
            if len(raw) != width * height:
                raise ValueError(f"invalid stereo {side} frame size")
            decoded[side] = raw
        stamp = getattr(self, "_t4_frame_stamp", self.get_clock().now().to_msg())
        fx = (width / 2.0) / math.tan(math.radians(self.stereo_hfov) / 2.0)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        for side in ("left", "right"):
            image = Image()
            image.header.stamp = stamp
            image.header.frame_id = f"go2_stereo_{side}_optical_frame"
            image.height = height
            image.width = width
            image.encoding = "mono8"
            image.is_bigendian = False
            image.step = width
            image.data = decoded[side]
            info = CameraInfo()
            info.header = image.header
            info.height = height
            info.width = width
            info.distortion_model = "plumb_bob"
            info.d = [0.0] * 5
            info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            image_publisher, info_publisher = self.stereo_publishers[side]
            image_publisher.publish(image)
            info_publisher.publish(info)
        self.stereo_frame_count += 1

    def _publish_depth(
        self,
        request: dict[str, Any],
        *,
        identity: tuple[str, int, int] | None = None,
        replay_epoch: int | None = None,
    ) -> tuple[int, int, int, int, list[list[float]]]:
        del identity, replay_epoch
        decoded_depth = decode_depth_request(request)
        if decoded_depth is not None:
            request = dict(request)
            request["depth_values"] = decoded_depth
        self._publish_r3_sensors(request)
        self._publish_metric_depth(request)
        self._publish_stereo(request)
        values = request.get("depth_values")
        if not values:
            return 0, 0, 0, 0, []
        height = int(request.get("depth_height", 0))
        width = int(request.get("depth_width", 0))
        row_stride = int(request.get("depth_row_stride", 1))
        column_stride = int(request.get("depth_column_stride", 1))
        if height <= 0 or width <= 0 or len(values) != height * width:
            raise ValueError("invalid sampled D435i depth payload")
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

        full_fx = 320.0 / math.tan(math.radians(self.depth_hfov) / 2.0)
        full_fy = 240.0 / math.tan(math.radians(self.depth_vfov) / 2.0)
        points: list[tuple[float, float, float]] = []
        for row in range(height):
            pixel_v = row * row_stride
            for column in range(width):
                depth = float(values[row * width + column])
                if (
                    not math.isfinite(depth)
                    or depth < self.depth_minimum
                    or depth > 6.0
                ):
                    continue
                pixel_u = column * column_stride
                camera_x = (pixel_u - 319.5) * depth / full_fx
                camera_y = (239.5 - pixel_v) * depth / full_fy
                camera_z = -depth
                base = [
                    self.depth_translation[axis]
                    + self.depth_camera_to_base[axis][0] * camera_x
                    + self.depth_camera_to_base[axis][1] * camera_y
                    + self.depth_camera_to_base[axis][2] * camera_z
                    for axis in range(3)
                ]
                world_z = pose[2] + sum(
                    gravity_row[axis] * base[axis] for axis in range(3)
                )
                ground_return = (
                    support_plane is not None and world_z <= support_plane + 0.08
                )
                if (
                    base[0] > 0.12
                    and not ground_return
                    and not robot_self_return(base)
                ):
                    points.append((base[0], base[1], base[2]))
        self._publish_pointcloud(points)
        targets = request.get("obstacle_targets_base", [])
        if not isinstance(targets, list):
            raise ValueError("obstacle target evidence must be a list")
        detected_targets = 0
        detected_target_values: list[
            tuple[float, float, float, float, float, float, float]
        ] = []
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
                detected_target_values.append((tx, ty, tz, hx, hy, hz, target_yaw))
        pose_yaw = t3_bridge._yaw(pose[3:])
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
            semantic_now = t3_bridge._semantic_now(self)
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

    def _publish_pointcloud(
        self, points: list[tuple[float, float, float]]
    ) -> None:
        """Publish a point cloud stamped identically to its pose/depth frame."""
        message = PointCloud2()
        message.header.stamp = getattr(
            self, "_t4_frame_stamp", self.get_clock().now().to_msg()
        )
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

    def _summary(self, status: str) -> dict[str, Any]:
        payload = super()._summary(status)
        if not self._t4_initialized:
            return payload
        payload.update(
            {
                "t4_schema_version": 1,
                "map_source": self.map_source,
                "pose_source": self.pose_source,
                "ground_truth_pose_used_for_nav": self.pose_source == "ground_truth",
                "static_map_publish_count": (
                    0 if self.map_source == "nvblox_online" else self.static_map_publish_count
                ),
                "static_map_selection_count": (
                    0
                    if self.map_source == "nvblox_online"
                    else len(self.static_map_selections)
                ),
                "static_map_generation": (
                    -1 if self.map_source == "nvblox_online" else self.static_map_generation
                ),
                "suppressed_static_map_calls": self.suppressed_static_map_calls,
                "last_reset_generation": self.last_reset_generation,
                "bootstrap_tf_publish_count": self.bootstrap_tf_publish_count,
                "real_pose_published": self._real_pose_published,
                "external_odometry_topic": self.external_odometry_topic,
                "external_odometry_timeout_sec": self.external_odometry_timeout,
                "external_odometry_received_count": self.external_odometry_received_count,
                "external_odometry_nav_publish_count": self.external_odometry_nav_publish_count,
                "external_odometry_stale_stop_count": self.external_odometry_stale_stop_count,
                "map_ready_generation": self.map_ready_generation,
                "map_not_ready_stop_count": self.map_not_ready_stop_count,
                "identity_safe_stop_count": self.identity_safe_stop_count,
                "stale_motion_execution_count": self.stale_motion_execution_count,
                "metric_depth_frame_count": self.metric_depth_frame_count,
                "metric_depth_valid_count": self.metric_depth_valid_count,
                "metric_depth_pose_sync_max_error_ms": (
                    0.0 if self.metric_depth_frame_count else None
                ),
                "metric_depth_transport_latency_max_ms": (
                    max(self.metric_depth_transport_latency_ms)
                    if self.metric_depth_transport_latency_ms
                    else None
                ),
                "metric_depth_transport_latency_mean_ms": (
                    sum(self.metric_depth_transport_latency_ms)
                    / len(self.metric_depth_transport_latency_ms)
                    if self.metric_depth_transport_latency_ms
                    else None
                ),
                "costmap_class_sample_count": self.costmap_class_sample_count,
                "costmap_class_sample_count_by_source": dict(
                    self.costmap_class_sample_count_by_source
                ),
                "costmap_class_generations": {
                    name: sorted(generations)
                    for name, generations in self.costmap_class_generations.items()
                },
                "stereo_feed_enabled": self.enable_stereo_feed,
                "stereo_frame_count": self.stereo_frame_count,
                "r3_lidar_frame_count": self.r3_lidar_frame_count,
                "r3_front_rgb_frame_count": self.r3_front_rgb_frame_count,
                "r3_d435i_rgb_frame_count": self.r3_d435i_rgb_frame_count,
                "r3_imu_frame_count": self.r3_imu_frame_count,
                "stereo_pair_max_sync_error_ms": (
                    0.0 if self.stereo_frame_count else None
                ),
                "stereo_baseline_m": self.stereo_baseline,
                "stereo_height_above_support_m": self.stereo_height,
                "stereo_pitch_down_deg": self.stereo_pitch_down,
                "stereo_hfov_deg": self.stereo_hfov,
                "camera_frame": self.camera_frame,
                "camera_height_above_support_m": self.camera_height,
                "camera_translation_from_base_m": list(self.camera_translation),
                "camera_pitch_down_deg": self.pitch_down,
                "camera_hfov_deg": self.hfov,
                "camera_vfov_deg": self.vfov,
                "depth_camera_model": "Intel RealSense D435i depth imager",
                "depth_camera_frame": self.camera_frame,
                "depth_height_above_support_m": self.depth_height,
                "depth_translation_from_base_m": list(self.depth_translation),
                "depth_pitch_down_deg": self.depth_pitch_down,
                "depth_hfov_deg": self.depth_hfov,
                "depth_vfov_deg": self.depth_vfov,
                "depth_minimum_m": self.depth_minimum,
            }
        )
        return payload


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = T4SensorBridge()
    executor = MultiThreadedExecutor(num_threads=8)
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
