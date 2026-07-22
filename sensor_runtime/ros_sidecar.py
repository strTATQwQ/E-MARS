#!/usr/bin/env python3
"""ROS 2 publisher for guarded model-free batches from the Isaac process."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .atomic import atomic_create_json, atomic_write_json
from .contract import (
    CAUSAL_ORDER,
    DIAGNOSTIC_GEOMETRY,
    DIAGNOSTIC_LIGHT,
    FRAMES,
)
from .core import SensorBatch, SensorFault
from .graph_handshake import (
    SIDECAR_GRAPH_REQUIREMENTS,
    make_steady_control_clock,
    observe_publishers,
    qos_snapshot,
    validate_graph_observation,
)
from .image_contract import downsample_rgb_2x2, require_rgb_content
from .wire import LatestBatchServer
from .pointcloud import (
    depth_points_from_calibration,
    frozen_depth_rejection_masks,
    nearest_valid_depth_tiles,
    valid_depth_evidence,
)
from .runtime_policy import RuntimePolicy, require_policy

try:  # Keep the module importable for offline contract tests.
    import rclpy
    from builtin_interfaces.msg import Time as RosTime
    from geometry_msgs.msg import Pose, PoseArray, TransformStamped, Vector3Stamped
    from nav_msgs.msg import Odometry
    from rclpy.clock import Clock as RclpyClock, ClockType
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2, PointField
    from std_msgs.msg import Bool, Header, String, UInt64
    from tf2_msgs.msg import TFMessage
except ImportError:  # pragma: no cover - ROS exists only in the online container
    rclpy = None
    Node = object  # type: ignore[assignment,misc]


def _rotation_matrix_wxyz(rotation: Any) -> np.ndarray:
    w, x, y, z = [float(value) for value in rotation]
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def project_depth_with_self_filter(
    depth_payload: dict[str, Any], pose_payload: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project D435i depth and retain frozen body + dynamic-link filtering."""

    depth = np.asarray(depth_payload["depth_m"], dtype=np.float32)
    if depth.shape != (480, 640):
        raise ValueError("D435i depth must be 640x480")
    valid_depth_count, valid_depth_ratio, valid_depth_ready = valid_depth_evidence(depth)
    if (
        not valid_depth_ready
        or int(depth_payload.get("valid_in_range_count", -1)) != valid_depth_count
        or not math.isclose(
            float(depth_payload.get("valid_in_range_ratio", -1.0)),
            valid_depth_ratio,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("D435i depth lacks the frozen current-render 10% valid-content evidence")
    links = list(depth_payload.get("link_centers_base", []))
    expected_geometry = [item["name"] for item in DIAGNOSTIC_GEOMETRY]
    if depth_payload.get("diagnostic_geometry_ids") != expected_geometry:
        raise ValueError("depth capture does not identify the frozen diagnostic geometry")
    if depth_payload.get("diagnostic_light_evidence") != {
        **DIAGNOSTIC_LIGHT,
        "temperature_enabled": True,
    }:
        raise ValueError("depth capture lacks observed frozen DomeLight attributes")
    timing = depth_payload.get("capture_timing")
    if not isinstance(timing, dict) or timing.get("clock") != "monotonic_perf_counter":
        raise ValueError("capture lacks monotonic stage timing evidence")
    timing_values = [
        timing.get("camera_read_content_sec"),
        timing.get("lidar_query_sec"),
        timing.get("total_capture_sec"),
    ]
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in timing_values
        )
        or int(timing.get("lidar_raycast_count", -1)) != 8 * 180
        or float(timing_values[2]) < max(float(timing_values[0]), float(timing_values[1]))
    ):
        raise ValueError("capture stage timing values are invalid")
    expected_links = {"base"} | {
        f"{leg}_{part}"
        for leg in ("FL", "FR", "RL", "RR")
        for part in ("thigh", "calf", "foot")
    }
    if len(links) != 13 or {str(item.get("name")) for item in links} != expected_links:
        raise ValueError("self-filter requires base plus all 12 thigh/calf/foot centers")
    if any(
        np.asarray(item.get("center_base"), dtype=np.float32).shape != (3,)
        or not np.isfinite(np.asarray(item.get("center_base"), dtype=np.float32)).all()
        for item in links
    ):
        raise ValueError("self-filter centers must all be finite base-frame xyz values")

    points = depth_points_from_calibration(
        depth,
        fx=float(depth_payload["fx"]),
        fy=float(depth_payload["fy"]),
        cx=float(depth_payload["cx"]),
        cy=float(depth_payload["cy"]),
        pitch_down_deg=float(depth_payload["pitch_down_deg"]),
        translation_from_base_m=depth_payload["translation_from_base_m"],
    )

    base_center = np.asarray(
        next(item["center_base"] for item in links if item["name"] == "base"),
        dtype=np.float32,
    )
    rotation = _rotation_matrix_wxyz(pose_payload["rotation_wxyz"]).astype(np.float32)
    position = np.asarray(pose_payload["position"], dtype=np.float32)
    world_z = position[2] + np.einsum("j,hwj->hw", rotation[2], points, dtype=np.float32)
    masks = frozen_depth_rejection_masks(
        depth,
        points,
        base_center,
        [(str(item["name"]), item["center_base"]) for item in links],
        world_z,
        float(depth_payload["support_plane_world_z"]),
    )
    range_mask = masks["range"]
    body_mask = masks["body"]
    dynamic_mask = masks["dynamic"]
    ground_mask = masks["ground"]
    forward_mask = masks["forward"]
    rejected = masks["rejected"]
    transport_points = nearest_valid_depth_tiles(points, depth, rejected)
    transport_finite = int(
        np.count_nonzero(np.isfinite(transport_points).all(axis=2))
    )
    counts = {
        "self_filter_input_points": int(depth.size),
        "self_filter_output_points": int(np.count_nonzero(~rejected)),
        "range_rejected_points": int(np.count_nonzero(range_mask)),
        "body_rejected_points": int(np.count_nonzero(body_mask & ~range_mask)),
        "dynamic_link_rejected_points": int(np.count_nonzero(dynamic_mask & ~range_mask)),
        "ground_rejected_points": int(np.count_nonzero(ground_mask & ~range_mask)),
        "forward_rejected_points": int(np.count_nonzero(forward_mask & ~range_mask)),
        "dynamic_link_count": len(links),
        "depth_valid_in_range_input_points": valid_depth_count,
        "depth_valid_in_range_ratio": valid_depth_ratio,
        "depth_cloud_reduction": "4x4_nearest_valid_after_full_resolution_filter",
        "depth_cloud_tile_stride": 4,
        "depth_cloud_width": int(transport_points.shape[1]),
        "depth_cloud_height": int(transport_points.shape[0]),
        "depth_cloud_point_count": int(transport_points.shape[0] * transport_points.shape[1]),
        "depth_cloud_output_points": transport_finite,
    }
    return transport_points, counts


class EvidenceWriter:
    """Move JSONL filesystem latency away from ROS publication callbacks."""

    def __init__(self, paths: dict[str, Path]) -> None:
        self.paths = paths
        self.queue: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue(maxsize=4096)
        self.drop_count = 0
        self.fault = ""
        self.written_counts = {name: 0 for name in paths}
        self._closing = threading.Event()
        self._flush_request = threading.Event()
        self._flush_ack = threading.Event()
        self.thread = threading.Thread(target=self._run, name="sensor-evidence-writer", daemon=True)
        self.thread.start()

    def offer(self, name: str, payload: dict[str, Any]) -> None:
        if self.fault:
            raise RuntimeError(self.fault)
        if not self.thread.is_alive():
            raise RuntimeError("evidence writer died")
        try:
            self.queue.put_nowait((name, payload))
        except queue.Full:
            self.drop_count += 1
            self.fault = "evidence_queue_overflow"
            raise RuntimeError(self.fault)

    def _run(self) -> None:
        streams: dict[str, Any] = {}
        try:
            streams = {
                name: path.open("x", encoding="utf-8", newline="\n")
                for name, path in self.paths.items()
            }
            while True:
                try:
                    item = self.queue.get(timeout=0.05)
                except queue.Empty:
                    if self._flush_request.is_set() and self.queue.empty():
                        for stream in streams.values():
                            stream.flush()
                        self._flush_request.clear()
                        self._flush_ack.set()
                    if self._closing.is_set() and self.queue.empty():
                        return
                    continue
                if item is None:
                    return
                name, payload = item
                streams[name].write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
                streams[name].flush()
                self.written_counts[name] += 1
        except BaseException as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
        finally:
            for stream in streams.values():
                stream.close()

    def flush(self, timeout_sec: float = 2.0) -> dict[str, int]:
        if self.fault or not self.thread.is_alive():
            raise RuntimeError(self.fault or "evidence writer died")
        self._flush_ack.clear()
        self._flush_request.set()
        if not self._flush_ack.wait(timeout_sec):
            raise TimeoutError("evidence writer flush acknowledgement timed out")
        if self.fault:
            raise RuntimeError(self.fault)
        return dict(self.written_counts)

    def close(self) -> None:
        self._closing.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(3.0)
        if self.thread.is_alive():
            raise RuntimeError("evidence writer did not stop within 3 seconds")
        if self.fault:
            raise RuntimeError(self.fault)


class ModelFreeRosSidecar(Node):  # type: ignore[misc]
    def __init__(self, socket_path: Path, result_dir: Path) -> None:
        super().__init__("internnav_model_free_sensor_producer")
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", False)
        if self.get_parameter("use_sim_time").value is not True:
            raise RuntimeError("sidecar requires use_sim_time=true")
        self.result_dir = result_dir.resolve()
        self.policy: RuntimePolicy = require_policy(
            os.environ.get("INTERNNAV_RUNTIME_POLICY", "strict_evidence")
        )
        self.started_monotonic_ns = time.monotonic_ns()
        self.last_batch_monotonic_ns: int | None = None
        self.static_generation: int | None = None
        self.last_identity: tuple[int, int] | None = None
        self.fatal_reason = ""
        self.failure_written = False
        self.gap_warning_active = False
        self._graph_ready_written = False
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=4,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        # tf2 listeners request RELIABLE /tf in Nav2 Jazzy.  A RELIABLE offer
        # remains compatible with the existing BEST_EFFORT recorder and bridge
        # subscriptions, while the reverse offer used before F1 did not match
        # Nav2 and left both costmaps without transforms.
        tf_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        sim_estop_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.clock_pub = self.create_publisher(Clock, "/clock", qos)
        self.generation_pub = self.create_publisher(UInt64, "/internnav/sensor_generation", qos)
        self.identity_pub = self.create_publisher(
            Vector3Stamped, "/internnav/sensor_frame_identity", qos
        )
        self.tf_pub = self.create_publisher(TFMessage, "/tf", tf_qos)
        self.tf_static_pub = self.create_publisher(TFMessage, "/tf_static", static_qos)
        self.rgb_pub = self.create_publisher(Image, "/go2/d435i/color/image_raw", qos)
        self.rgb_info_pub = self.create_publisher(CameraInfo, "/go2/d435i/color/camera_info", qos)
        self.depth_pub = self.create_publisher(Image, "/go2/d435i/depth/image_rect", qos)
        self.depth_info_pub = self.create_publisher(CameraInfo, "/go2/d435i/depth/camera_info", qos)
        self.depth_points_pub = self.create_publisher(PointCloud2, "/go2/depth/points", qos)
        self.lidar_raw_pub = self.create_publisher(PointCloud2, "/internvla_t4/go2/lidar_raw", qos)
        self.front_raw_pub = self.create_publisher(Image, "/internvla_t4/go2/front_rgb_raw", qos)
        self.front_info_raw_pub = self.create_publisher(
            CameraInfo, "/internvla_t4/go2/front_rgb_camera_info_raw", qos
        )
        self.imu_pub = self.create_publisher(Imu, "/go2/imu/data", qos)
        self.centers_pub = self.create_publisher(
            PoseArray, "/internvla_t4/go2/self_filter_link_centers", qos
        )
        self.odom_pub = self.create_publisher(Odometry, "/odom", qos)
        self.metadata_pub = self.create_publisher(
            String, "/internnav/sensor_producer/metadata", qos
        )
        self.sim_estop_pub = (
            self.create_publisher(Bool, "/internvla/stop", sim_estop_qos)
            if os.environ.get("INTERNNAV_SESSION_PROFILE") == "completion_sim_map"
            else None
        )
        self._graph_publishers = {
            "/clock": self.clock_pub,
            "/internnav/sensor_generation": self.generation_pub,
            "/internnav/sensor_frame_identity": self.identity_pub,
            "/tf": self.tf_pub,
            "/tf_static": self.tf_static_pub,
            "/go2/d435i/color/image_raw": self.rgb_pub,
            "/go2/d435i/color/camera_info": self.rgb_info_pub,
            "/go2/d435i/depth/image_rect": self.depth_pub,
            "/go2/d435i/depth/camera_info": self.depth_info_pub,
            "/go2/depth/points": self.depth_points_pub,
            "/internvla_t4/go2/lidar_raw": self.lidar_raw_pub,
            "/internvla_t4/go2/front_rgb_raw": self.front_raw_pub,
            "/internvla_t4/go2/front_rgb_camera_info_raw": self.front_info_raw_pub,
            "/go2/imu/data": self.imu_pub,
            "/internvla_t4/go2/self_filter_link_centers": self.centers_pub,
            "/odom": self.odom_pub,
        }
        self.evidence = EvidenceWriter(
            {
                "sensor": self.result_dir / "sensor_producer_audit.jsonl",
                "controller": self.result_dir / "controller_stop_audit.jsonl",
                "reset": self.result_dir / "reset_audit.jsonl",
                "warning": self.result_dir / "sidecar_warnings.jsonl",
            }
        )
        self.server = LatestBatchServer(socket_path)
        try:
            self.server.start()
        except BaseException:
            self.evidence.close()
            raise
        self._control_clock = make_steady_control_clock(RclpyClock, ClockType)
        self.create_timer(0.002, self._drain, clock=self._control_clock)
        self.create_timer(0.02, self._watchdog, clock=self._control_clock)
        self.create_timer(0.02, self._snapshot, clock=self._control_clock)
        self.create_timer(0.02, self._graph_ready, clock=self._control_clock)
        atomic_create_json(
            self.result_dir / "sidecar_role_ready.json",
            {
                "schema_version": 2,
                "status": "ROLE_READY",
                "role": "sensor_ros_sidecar",
                "runtime_policy": self.policy.name,
                "pid": os.getpid(),
                "control_timer_clock_type": "STEADY_TIME",
                "message_stamp_clock_type": "ROS_TIME",
                "owned_qos": {
                    "dynamic": qos_snapshot(qos),
                    "tf_dynamic": qos_snapshot(tf_qos),
                    "static": qos_snapshot(static_qos),
                },
            },
        )

    def _graph_ready(self) -> None:
        if self._graph_ready_written or self.fatal_reason:
            return
        try:
            verdict = validate_graph_observation(
                SIDECAR_GRAPH_REQUIREMENTS,
                observe_publishers(self, self._graph_publishers),
            )
            if verdict["ready"]:
                verdict["role"] = "sensor_ros_sidecar"
                atomic_create_json(self.result_dir / "sidecar_graph_ready.json", verdict)
                self._graph_ready_written = True
        except BaseException as exc:
            self._fail(f"graph_ready: {type(exc).__name__}: {exc}")

    @staticmethod
    def _time(stamp_ns: int) -> Any:
        return RosTime(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)

    @classmethod
    def _header(cls, stamp_ns: int, frame_id: str) -> Any:
        return Header(stamp=cls._time(stamp_ns), frame_id=frame_id)

    @staticmethod
    def _set_quaternion(target: Any, values: Any) -> None:
        w, x, y, z = [float(value) for value in values]
        target.w, target.x, target.y, target.z = w, x, y, z

    @classmethod
    def _image(cls, array: Any, stamp_ns: int, frame: str, encoding: str) -> Any:
        value = np.ascontiguousarray(array)
        message = Image()
        message.header = cls._header(stamp_ns, frame)
        message.height, message.width = value.shape[:2]
        message.encoding = encoding
        message.is_bigendian = False
        message.step = int(value.strides[0])
        message.data = value.tobytes()
        return message

    @classmethod
    def _info(cls, stamp_ns: int, frame: str, width: int, height: int, fx: float, fy: float) -> Any:
        message = CameraInfo()
        message.header = cls._header(stamp_ns, frame)
        message.width, message.height = width, height
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        message.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        message.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        message.distortion_model = "plumb_bob"
        message.d = [0.0] * 5
        message.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        return message

    @classmethod
    def _cloud(cls, points: Any, stamp_ns: int, frame: str) -> Any:
        array = np.ascontiguousarray(points, dtype=np.float32)
        height = int(array.shape[0]) if array.ndim == 3 else 1
        width = int(array.shape[1]) if array.ndim == 3 else int(array.shape[0])
        message = PointCloud2()
        message.header = cls._header(stamp_ns, frame)
        message.height, message.width = height, width
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = 12
        message.row_step = width * 12
        message.is_dense = bool(np.isfinite(array).all())
        message.data = array.reshape(-1, 3).tobytes()
        return message

    @classmethod
    def _transform(cls, stamp_ns: int, parent: str, child: str, translation: Any, rotation: Any) -> Any:
        message = TransformStamped()
        message.header = cls._header(stamp_ns, parent)
        message.child_frame_id = child
        message.transform.translation.x = float(translation[0])
        message.transform.translation.y = float(translation[1])
        message.transform.translation.z = float(translation[2])
        cls._set_quaternion(message.transform.rotation, rotation)
        return message

    def _publish_prefix(self, batch: SensorBatch) -> None:
        stamp = batch.stamp_ns
        tf_payload = batch.payloads["tf"]
        self.clock_pub.publish(Clock(clock=self._time(stamp)))
        self.generation_pub.publish(UInt64(data=batch.generation))
        if self.static_generation != batch.generation:
            self.tf_static_pub.publish(
                TFMessage(
                    transforms=[
                        self._transform(
                            stamp,
                            item["parent"],
                            item["child"],
                            item["translation"],
                            item["rotation_wxyz"],
                        )
                        for item in tf_payload["fixed"]
                    ]
                )
            )
            self.static_generation = batch.generation
        self.tf_pub.publish(
            TFMessage(
                transforms=[
                    self._transform(
                        stamp,
                        FRAMES["odom"],
                        FRAMES["base"],
                        tf_payload["base_translation"],
                        tf_payload["base_rotation_wxyz"],
                    )
                ]
            )
        )
        identity = Vector3Stamped()
        identity.header = self._header(stamp, FRAMES["base"])
        identity.vector.x = float(batch.generation)
        identity.vector.y = float(batch.sequence)
        identity.vector.z = float(batch.safe_stop["render_id"])
        self.identity_pub.publish(identity)

    def _publish(self, batch: SensorBatch) -> None:
        stamp = batch.stamp_ns
        rgb = batch.payloads["d435i_rgb"]
        depth = batch.payloads["d435i_depth"]
        lidar = batch.payloads["lidar"]
        pose = batch.payloads["pose"]
        d435i_content = require_rgb_content(
            rgb["rgb8"], rgb.get("d435i_content_evidence"), "D435i RGB"
        )
        self._publish_prefix(batch)
        self.rgb_pub.publish(self._image(rgb["rgb8"], stamp, FRAMES["d435i_color"], "rgb8"))
        self.rgb_info_pub.publish(
            self._info(stamp, FRAMES["d435i_color"], 640, 480, rgb["d435i_fx"], rgb["d435i_fy"])
        )
        self.depth_pub.publish(self._image(depth["depth_m"], stamp, FRAMES["d435i_depth"], "32FC1"))
        self.depth_info_pub.publish(
            self._info(stamp, FRAMES["d435i_depth"], 640, 480, depth["fx"], depth["fy"])
        )
        depth_points, filter_counts = project_depth_with_self_filter(depth, pose)
        self.depth_points_pub.publish(self._cloud(depth_points, stamp, FRAMES["base"]))
        front_publish = downsample_rgb_2x2(rgb["front_rgb8"])
        if tuple(front_publish.shape) != (120, 160, 3):
            raise RuntimeError("front RGB render cannot produce frozen 160x120 publish image")
        front_content = require_rgb_content(
            front_publish,
            rgb.get("front_publish_content_evidence"),
            "front publish RGB",
        )
        self.front_raw_pub.publish(
            self._image(front_publish, stamp, FRAMES["front_rgb"], "rgb8")
        )
        self.front_info_raw_pub.publish(
            self._info(
                stamp,
                FRAMES["front_rgb"],
                160,
                120,
                float(rgb["front_fx"]) / 2.0,
                float(rgb["front_fy"]) / 2.0,
            )
        )
        centers = PoseArray(header=self._header(stamp, FRAMES["base"]))
        for item in depth["link_centers_base"]:
            center = item["center_base"]
            message = Pose()
            message.position.x, message.position.y, message.position.z = map(float, center)
            message.orientation.w = 1.0
            centers.poses.append(message)
        self.centers_pub.publish(centers)
        self.lidar_raw_pub.publish(self._cloud(lidar["points_lidar"], stamp, FRAMES["lidar"]))
        odom = Odometry()
        odom.header = self._header(stamp, FRAMES["odom"])
        odom.child_frame_id = FRAMES["base"]
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(
            float, pose["position"]
        )
        self._set_quaternion(odom.pose.pose.orientation, pose["rotation_wxyz"])
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = map(
            float, pose["linear_velocity"]
        )
        odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = map(
            float, pose["angular_velocity"]
        )
        self.odom_pub.publish(odom)
        imu = Imu()
        imu.header = self._header(stamp, FRAMES["imu"])
        self._set_quaternion(imu.orientation, pose["rotation_wxyz"])
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = map(
            float, pose["angular_velocity"]
        )
        imu.linear_acceleration_covariance[0] = -1.0
        self.imu_pub.publish(imu)

        wall_ns = time.monotonic_ns()
        sensor_record = {
            "schema_version": 2,
            "wall_time_unix": time.time(),
            "wall_monotonic_ns": wall_ns,
            "stamp_ns": stamp,
            "generation": batch.generation,
            "sequence": batch.sequence,
            "stream_stamps_ns": dict(batch.stream_stamps_ns),
            "sidecar_publish_order": list(CAUSAL_ORDER),
            "imu_same_stamp_published": True,
            "render_id": int(batch.safe_stop["render_id"]),
            "render_generation": int(batch.safe_stop["render_generation"]),
            "receive_overwrite_count": self.server.slot.overwrite_count,
            "receive_reset_clear_count": self.server.slot.reset_clear_count,
            "receive_barrier_drop_count": self.server.slot.barrier_drop_count,
            "diagnostic_geometry_ids": list(depth["diagnostic_geometry_ids"]),
            "diagnostic_light_evidence": dict(depth["diagnostic_light_evidence"]),
            "d435i_rgb_content": d435i_content,
            "front_rgb_content": front_content,
            "capture_timing": dict(depth["capture_timing"]),
            "lidar_finite_input_points": int(
                np.count_nonzero(np.isfinite(lidar["points_lidar"]).all(axis=2))
            ),
            **filter_counts,
        }
        controller_record = {
            "schema_version": 2,
            "wall_monotonic_ns": wall_ns,
            "generation": batch.generation,
            "sequence": batch.sequence,
            **dict(batch.safe_stop),
        }
        if batch.reset_reason is not None:
            self.evidence.offer(
                "reset",
                {
                    "schema_version": 2,
                    "wall_monotonic_ns": wall_ns,
                    "generation": batch.generation,
                    "sequence": batch.sequence,
                    "reason": batch.reset_reason,
                    "reset_kind": batch.safe_stop["reset_kind"],
                    "atomic_latest_clear": True,
                },
            )
        self.evidence.offer("controller", controller_record)
        self.evidence.offer("sensor", sensor_record)
        self.metadata_pub.publish(String(data=json.dumps(sensor_record, sort_keys=True)))
        if self.sim_estop_pub is not None:
            self.sim_estop_pub.publish(
                Bool(data=bool(batch.safe_stop["emergency_stop"]))
            )
        self.last_batch_monotonic_ns = wall_ns
        self.last_identity = (batch.generation, batch.sequence)
        self.gap_warning_active = False

    def _drain(self) -> None:
        try:
            batch = self.server.slot.take(0.0)
            if batch is not None:
                self._publish(batch)
        except BaseException as exc:
            self._fail(f"{type(exc).__name__}: {exc}")

    def _fail(self, reason: str) -> None:
        if not self.fatal_reason:
            self.fatal_reason = reason
        try:
            if not self.failure_written:
                atomic_write_json(
                    self.result_dir / "sidecar_first_failure.json",
                    {
                        "schema_version": 2,
                        "status": "FAIL",
                        "reason": self.fatal_reason,
                        "wall_time_unix": time.time(),
                    },
                )
                self.failure_written = True
        except BaseException:
            # Evidence failure must never suppress required-process shutdown.
            pass
        finally:
            if rclpy.ok():
                rclpy.shutdown()

    def _watchdog(self) -> None:
        now = time.monotonic_ns()
        reason = ""
        if self.server.fault:
            reason = self.server.fault
        elif self.evidence.fault or not self.evidence.thread.is_alive():
            reason = self.evidence.fault or "evidence_writer_died"
        elif self.last_batch_monotonic_ns is None:
            if (now - self.started_monotonic_ns) / 1e9 > 20.0:
                reason = "first_sensor_batch_missed_ready_deadline"
        elif (
            not (self.result_dir / "producer_snapshot_ack.json").is_file()
            and (now - self.last_batch_monotonic_ns) / 1e9
            > self.policy.source_timeout_sec
        ):
            reason = (
                "sensor_stream_gap_exceeded_0.35_sec"
                if self.policy.name == "strict_evidence"
                else "sensor_stream_gap_exceeded_5.0_sec"
            )
        if reason and rclpy.ok():
            if self.policy.name == "completion_sim" and reason.startswith(
                "sensor_stream_gap_exceeded_"
            ):
                if not self.gap_warning_active:
                    self.get_logger().warning(reason)
                    self.evidence.offer(
                        "warning",
                        {
                            "schema_version": 1,
                            "level": "WARN",
                            "reason": reason,
                            "runtime_policy": self.policy.name,
                            "wall_monotonic_ns": now,
                            "last_identity": self.last_identity,
                            "action": "continue_without_global_shutdown",
                        },
                    )
                    self.gap_warning_active = True
                return
            self.get_logger().fatal(reason)
            self._fail(reason)

    def _snapshot(self) -> None:
        request_path = self.result_dir / "snapshot_request.json"
        ack_path = self.result_dir / "sidecar_snapshot_ack.json"
        if not request_path.is_file() or ack_path.exists() or self.fatal_reason:
            return
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            target = (int(request["generation"]), int(request["sequence"]))
            self._drain()
            if self.last_identity != target:
                return
            counts = self.evidence.flush()
            atomic_write_json(
                ack_path,
                {
                    "schema_version": 2,
                    "status": "PASS",
                    "snapshot_id": request["snapshot_id"],
                    "generation": target[0],
                    "sequence": target[1],
                    "writer_counts": counts,
                    "server_received_count": self.server.received_count,
                    "receive_overwrite_count": self.server.slot.overwrite_count,
                    "receive_reset_clear_count": self.server.slot.reset_clear_count,
                    "receive_barrier_drop_count": self.server.slot.barrier_drop_count,
                    "writer_thread_alive": self.evidence.thread.is_alive(),
                    "writer_fault": self.evidence.fault or None,
                },
            )
        except BaseException as exc:
            self._fail(f"snapshot_failed: {type(exc).__name__}: {exc}")

    def close(self) -> None:
        errors = []
        for name, action in (("server", self.server.close), ("evidence", self.evidence.close)):
            try:
                action()
            except BaseException as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))


def main(argv: list[str] | None = None) -> int:
    if rclpy is None:
        raise RuntimeError("ROS 2 Python packages are required")
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args, ros_args = parser.parse_known_args(argv)
    rclpy.init(args=ros_args)
    node: ModelFreeRosSidecar | None = None
    errors: list[str] = []

    def coordinated_completion_shutdown() -> bool:
        return bool(
            node is not None
            and node.policy.name == "completion_sim"
            and (node.result_dir / "inner_stop.request").is_file()
            and not rclpy.ok()
        )

    try:
        node = ModelFreeRosSidecar(args.socket, args.result_dir)
        rclpy.spin(node)
    except BaseException as exc:
        if not coordinated_completion_shutdown():
            errors.append(f"spin: {type(exc).__name__}: {exc}")
            if node is not None:
                node._fail(errors[-1])
    finally:
        for name, action in (
            ("sidecar_close", None if node is None else node.close),
            ("destroy_node", None if node is None else node.destroy_node),
            ("rclpy_shutdown", (lambda: rclpy.shutdown()) if rclpy.ok() else None),
        ):
            if action is None:
                continue
            try:
                action()
            except BaseException as exc:
                if not coordinated_completion_shutdown():
                    errors.append(f"{name}: {type(exc).__name__}: {exc}")
    for error in errors:
        print(error, file=sys.stderr, flush=True)
    return 2 if errors or (node is not None and node.fatal_reason) else 0


if __name__ == "__main__":
    raise SystemExit(main())
