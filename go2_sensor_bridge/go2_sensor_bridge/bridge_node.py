"""Generation-aware standard-message bridge for frozen Go2 sensor topics."""

from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import PoseArray, Vector3Stamped
from rclpy.clock import Clock as RclpyClock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tf2_msgs.msg import TFMessage

from sensor_runtime.atomic import atomic_create_json, atomic_write_json
from sensor_runtime.camera_contract import matrices_from_fov, require_camera_matrices
from sensor_runtime.graph_handshake import (
    BRIDGE_GRAPH_REQUIREMENTS,
    make_steady_control_clock,
    observe_publishers,
    qos_snapshot,
    validate_graph_observation,
)
from sensor_runtime.pointcloud import (
    filter_lidar_xyz,
    finite_xyz_rows,
    packed_xyz,
    xyz_array_from_message,
)
from sensor_runtime.runtime_policy import require_policy


FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
]
REQUIRED_PARTS = frozenset(
    {"clock", "tf", "identity", "centers", "lidar", "depth", "front", "front_info"}
)


def _is_t5_preclock_zero_tf_warn_drop(
    *,
    runtime_policy: str,
    opt_in: bool,
    lane_identity_prefix: str,
    part: str,
    stamp: int,
    positive_stamp_seen: bool,
) -> bool:
    """Allow only the expected T5 bootstrap TF emitted before x86 /clock."""

    return (
        runtime_policy == "completion_sim"
        and opt_in
        and lane_identity_prefix in {"a::", "b::"}
        and part == "tf"
        and stamp == 0
        and not positive_stamp_seen
    )


def _stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _clock_ns(message: Clock) -> int:
    return int(message.clock.sec) * 1_000_000_000 + int(message.clock.nanosec)


def _cloud(header: Any, points: Any) -> PointCloud2:
    _array, width, height, data, dense = packed_xyz(points)
    message = PointCloud2()
    message.header = header
    message.width, message.height = width, height
    message.fields = FIELDS
    message.is_bigendian = False
    message.point_step = 12
    message.row_step = width * 12
    message.is_dense = dense
    message.data = data
    return message


class FrameWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=4096)
        self.fault = ""
        self.written_count = 0
        self._flush_request = threading.Event()
        self._flush_ack = threading.Event()
        self._closing = threading.Event()
        self.thread = threading.Thread(target=self._run, name="bridge-frame-writer", daemon=True)
        self.thread.start()

    def offer(self, record: dict[str, Any]) -> None:
        if self.fault or not self.thread.is_alive():
            raise RuntimeError(self.fault or "bridge writer died")
        try:
            self.queue.put_nowait(record)
        except queue.Full as exc:
            self.fault = "bridge_writer_queue_overflow"
            raise RuntimeError(self.fault) from exc

    def _run(self) -> None:
        stream = None
        try:
            stream = self.path.open("x", encoding="utf-8", newline="\n")
            while True:
                try:
                    record = self.queue.get(timeout=0.05)
                except queue.Empty:
                    if self._flush_request.is_set() and self.queue.empty():
                        stream.flush()
                        self._flush_request.clear()
                        self._flush_ack.set()
                    if self._closing.is_set() and self.queue.empty():
                        return
                    continue
                if record is None:
                    return
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                self.written_count += 1
        except BaseException as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
        finally:
            if stream is not None:
                stream.close()

    def flush(self) -> int:
        if self.fault or not self.thread.is_alive():
            raise RuntimeError(self.fault or "bridge writer died")
        self._flush_ack.clear()
        self._flush_request.set()
        if not self._flush_ack.wait(2.0):
            raise TimeoutError("bridge writer flush timed out")
        return self.written_count

    def close(self) -> None:
        self._closing.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(3.0)
        if self.thread.is_alive() or self.fault:
            raise RuntimeError(self.fault or "bridge writer did not stop")


class Go2SensorBridge(Node):
    """Pair clock/TF/identity/centers/depth/LiDAR before canonical output."""

    def __init__(self) -> None:
        super().__init__("go2_sensor_bridge")
        self.declare_parameter("result_dir", "")
        self.declare_parameter("sensor_timeout_sec", 0.35)
        self.declare_parameter("runtime_policy", "strict_evidence")
        self.declare_parameter("enable_d435i", True)
        self.declare_parameter("enable_lidar", True)
        self.declare_parameter("enable_rgb", True)
        self.declare_parameter("allow_preclock_zero_tf_warn_drop", False)
        self.declare_parameter("minimum_range_m", 0.10)
        self.declare_parameter("maximum_range_m", 12.0)
        self.declare_parameter("minimum_height_m", -0.55)
        self.declare_parameter("maximum_height_m", 2.0)
        result_text = str(self.get_parameter("result_dir").value)
        if not result_text:
            raise RuntimeError("result_dir is required")
        self.result_dir = Path(result_text).resolve()
        self.result_dir.mkdir(parents=False, exist_ok=False)
        self.timeout = float(self.get_parameter("sensor_timeout_sec").value)
        self.runtime_policy = require_policy(
            str(self.get_parameter("runtime_policy").value)
        )
        self.enable_d435i = bool(self.get_parameter("enable_d435i").value)
        self.enable_lidar = bool(self.get_parameter("enable_lidar").value)
        self.enable_rgb = bool(self.get_parameter("enable_rgb").value)
        self.allow_preclock_zero_tf_warn_drop = bool(
            self.get_parameter("allow_preclock_zero_tf_warn_drop").value
        )
        self.lane_identity_prefix = os.environ.get("INTERNNAV_T5_ID_PREFIX", "")
        if self.allow_preclock_zero_tf_warn_drop and (
            self.runtime_policy.name != "completion_sim"
            or self.lane_identity_prefix not in {"a::", "b::"}
        ):
            raise RuntimeError(
                "preclock zero-TF relaxation requires a T5 completion_sim lane"
            )
        if self.runtime_policy.name != "completion_sim" and not all(
            (self.enable_d435i, self.enable_lidar, self.enable_rgb)
        ):
            raise RuntimeError("strict_evidence requires every frozen sensor group")
        self.minimum_range = float(self.get_parameter("minimum_range_m").value)
        self.maximum_range = float(self.get_parameter("maximum_range_m").value)
        self.minimum_height = float(self.get_parameter("minimum_height_m").value)
        self.maximum_height = float(self.get_parameter("maximum_height_m").value)
        observed = (
            self.timeout,
            self.minimum_range,
            self.maximum_range,
            self.minimum_height,
            self.maximum_height,
        )
        if observed != (
            self.runtime_policy.bridge_timeout_sec,
            0.10,
            12.0,
            -0.55,
            2.0,
        ):
            raise RuntimeError("frozen LiDAR/freshness filters cannot be overridden")
        self._lock = threading.RLock()
        self._pending: dict[int, dict[str, Any]] = {}
        self._completion_latest: dict[str, tuple[int, Any]] = {}
        self._completion_tolerance_ns = 2_500_000_000
        self._completion_lidar_stamp = 0
        self._completion_depth_stamp = 0
        self._completion_front_stamp = 0
        self._identity_generation = -1
        self._identity_sequence = -1
        self._identity_stamp = 0
        self._last_processed: tuple[int, int] | None = None
        self._last_processed_stamp = 0
        self._discarded_through_stamp = 0
        self._positive_tf_stamp_seen = False
        self._warnings: list[dict[str, Any]] = []
        self._fatal_reason = ""
        self._failure_written = False
        self._graph_ready_written = False
        self.counts = {
            "frames": 0,
            "lidar_input_points": 0,
            "lidar_output_points": 0,
            "lidar_self_filtered_points": 0,
            "lidar_range_filtered_points": 0,
            "lidar_height_filtered_points": 0,
            "safety_frames": 0,
            "partial_batches_dropped": 0,
            "late_parts_dropped": 0,
            "completion_lidar_groups": 0,
            "completion_depth_groups": 0,
            "completion_front_groups": 0,
            "completion_nearest_tf_matches": 0,
            "completion_center_fallbacks": 0,
            "preclock_zero_tf_dropped": 0,
        }
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=4,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.lidar_pub = self.create_publisher(PointCloud2, "/go2/lidar/points", qos)
        self.lidar_base_pub = self.create_publisher(PointCloud2, "/go2/lidar/points_base", qos)
        self.safety_pub = self.create_publisher(PointCloud2, "/go2/safety/points", qos)
        self.front_pub = self.create_publisher(Image, "/go2/front_rgb/image_raw", qos)
        self.front_info_pub = self.create_publisher(CameraInfo, "/go2/front_rgb/camera_info", qos)
        self._graph_publishers = {
            "/go2/lidar/points": self.lidar_pub,
            "/go2/lidar/points_base": self.lidar_base_pub,
            "/go2/safety/points": self.safety_pub,
            "/go2/front_rgb/image_raw": self.front_pub,
            "/go2/front_rgb/camera_info": self.front_info_pub,
        }
        self.create_subscription(Clock, "/clock", self._on_clock, qos)
        self.create_subscription(TFMessage, "/tf", self._on_tf, qos)
        self.create_subscription(
            Vector3Stamped, "/internnav/sensor_frame_identity", self._on_identity, qos
        )
        self.create_subscription(
            PoseArray, "/internvla_t4/go2/self_filter_link_centers", self._on_centers, qos
        )
        self.create_subscription(
            PointCloud2, "/internvla_t4/go2/lidar_raw", self._on_lidar, qos
        )
        self.create_subscription(PointCloud2, "/go2/depth/points", self._on_depth, qos)
        self.create_subscription(Image, "/internvla_t4/go2/front_rgb_raw", self._on_front, qos)
        self.create_subscription(
            CameraInfo,
            "/internvla_t4/go2/front_rgb_camera_info_raw",
            self._on_front_info,
            qos,
        )
        self._control_clock = make_steady_control_clock(RclpyClock, ClockType)
        self.create_timer(0.02, self._watchdog, clock=self._control_clock)
        self.create_timer(0.02, self._snapshot, clock=self._control_clock)
        self.create_timer(1.0, self._summary, clock=self._control_clock)
        self.create_timer(0.02, self._graph_ready, clock=self._control_clock)
        # Start the only non-ROS resource last, so a partially constructed node
        # cannot leak a background writer thread.
        self.frame_writer = FrameWriter(self.result_dir / "go2_sensor_bridge_frames.jsonl")
        atomic_create_json(
            self.result_dir.parent / "bridge_role_ready.json",
            {
                "schema_version": 2,
                "status": "ROLE_READY",
                "role": "go2_sensor_bridge",
                "runtime_policy": self.runtime_policy.as_dict(),
                "pid": os.getpid(),
                "control_timer_clock_type": "STEADY_TIME",
                "message_stamp_clock_type": "ROS_TIME",
                "allow_preclock_zero_tf_warn_drop": (
                    self.allow_preclock_zero_tf_warn_drop
                ),
                "owned_qos": {"dynamic": qos_snapshot(qos)},
            },
        )

    def _graph_ready(self) -> None:
        with self._lock:
            if self._graph_ready_written or self._fatal_reason:
                return
        try:
            verdict = validate_graph_observation(
                BRIDGE_GRAPH_REQUIREMENTS,
                observe_publishers(self, self._graph_publishers),
            )
            if verdict["ready"]:
                verdict["role"] = "go2_sensor_bridge"
                atomic_create_json(
                    self.result_dir.parent / "bridge_graph_ready.json", verdict
                )
                with self._lock:
                    self._graph_ready_written = True
        except BaseException as exc:
            self._fail(f"graph_ready: {type(exc).__name__}: {exc}")

    def _fail(self, reason: str) -> None:
        with self._lock:
            if not self._fatal_reason:
                self._fatal_reason = reason
            should_write = not self._failure_written
        if should_write:
            try:
                atomic_write_json(
                    self.result_dir / "bridge_first_failure.json",
                    {"schema_version": 2, "status": "FAIL", "reason": self._fatal_reason, "wall_time_unix": time.time()},
                )
                with self._lock:
                    self._failure_written = True
            except BaseException:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except BaseException:
            pass

    def _entry(self, stamp: int) -> dict[str, Any] | None:
        if stamp <= 0:
            raise ValueError("bridge input has non-positive sim stamp")
        closed_boundary = max(self._last_processed_stamp, self._discarded_through_stamp)
        if stamp <= closed_boundary and self.runtime_policy.name == "completion_sim":
            self.counts["late_parts_dropped"] += 1
            return None
        if stamp <= closed_boundary:
            raise RuntimeError("late/replayed bridge input crossed a published boundary")
        entry = self._pending.setdefault(
            stamp, {"created_monotonic": time.monotonic(), "parts": {}}
        )
        return entry

    def _ingest(self, part: str, stamp: int, value: Any) -> None:
        try:
            with self._lock:
                if _is_t5_preclock_zero_tf_warn_drop(
                    runtime_policy=self.runtime_policy.name,
                    opt_in=self.allow_preclock_zero_tf_warn_drop,
                    lane_identity_prefix=self.lane_identity_prefix,
                    part=part,
                    stamp=stamp,
                    positive_stamp_seen=self._positive_tf_stamp_seen,
                ):
                    self.counts["preclock_zero_tf_dropped"] += 1
                    if self.counts["preclock_zero_tf_dropped"] == 1:
                        self._warnings.append(
                            {
                                "schema_version": 1,
                                "level": "WARN",
                                "runtime_policy": self.runtime_policy.name,
                                "reason": "t5_preclock_zero_tf",
                                "stamp_ns": stamp,
                                "observed_parts": [part],
                                "missing_parts": [],
                                "action": "record_and_drop_preclock_tf_then_continue",
                            }
                        )
                        atomic_write_json(
                            self.result_dir / "bridge_warnings.json",
                            {
                                "schema_version": 1,
                                "status": "RECORDED",
                                "runtime_policy": self.runtime_policy.name,
                                "items": list(self._warnings),
                            },
                        )
                    return
                # Bootstrap and real pose TF are emitted by the same sensor
                # bridge publisher.  Seal the exception on its first positive
                # TF, not on a cross-topic callback whose delivery may race an
                # already queued zero-stamp bootstrap transform.
                if part == "tf" and stamp > 0:
                    self._positive_tf_stamp_seen = True
                if self.runtime_policy.name == "completion_sim":
                    self._ingest_completion_locked(part, stamp, value)
                    return
                entry = self._entry(stamp)
                if entry is None:
                    return
                if part in entry["parts"]:
                    raise RuntimeError(f"duplicate bridge part: {part}")
                entry["parts"][part] = value
                self._drain_ready()
        except BaseException as exc:
            self._fail(f"{type(exc).__name__}: {exc}")

    def _completion_nearest_locked(
        self, part: str, target_stamp: int
    ) -> tuple[int, Any] | None:
        candidate = self._completion_latest.get(part)
        if candidate is None:
            return None
        if abs(candidate[0] - target_stamp) > self._completion_tolerance_ns:
            return None
        return candidate

    def _ingest_completion_locked(self, part: str, stamp: int, value: Any) -> None:
        """Dispatch completion_sim data by consumer instead of atomic frame stamp."""

        if stamp <= 0:
            raise ValueError("bridge input has non-positive sim stamp")
        previous = self._completion_latest.get(part)
        if previous is not None and stamp < previous[0]:
            self.counts["late_parts_dropped"] += 1
            return
        self._completion_latest[part] = (stamp, value)
        if self.enable_lidar and part in {"lidar", "centers", "tf"}:
            self._publish_completion_lidar_locked()
        if self.enable_d435i and part == "depth":
            self._publish_completion_depth_locked()
        if self.enable_rgb and part in {"front", "front_info"}:
            self._publish_completion_front_locked()

    def _publish_completion_lidar_locked(self) -> None:
        lidar_entry = self._completion_latest.get("lidar")
        if lidar_entry is None or lidar_entry[0] <= self._completion_lidar_stamp:
            return
        stamp, raw_lidar = lidar_entry
        centers_entry = self._completion_nearest_locked("centers", stamp)
        tf_entry = self._completion_nearest_locked("tf", stamp)
        if tf_entry is None:
            return
        if centers_entry is None and self.enable_d435i:
            return
        if centers_entry is None:
            # completion_sim may intentionally disable the duplicate D435 path
            # that normally supplies dynamic link centers. Keep the frozen
            # chassis box and a conservative base-centered sphere; strict
            # evidence still requires all 13 measured centers at the same stamp.
            centers = [(0.0, 0.0, 0.0)] * 13
            centers_stamp = None
            self.counts["completion_center_fallbacks"] += 1
        else:
            centers = centers_entry[1]
            centers_stamp = centers_entry[0]
        lidar, lidar_base, counts = self._filter_lidar(raw_lidar, centers)
        self.lidar_pub.publish(lidar)
        self.lidar_base_pub.publish(lidar_base)
        lidar_base_array = xyz_array_from_message(lidar_base)
        safety = _cloud(lidar_base.header, finite_xyz_rows(lidar_base_array))
        self.safety_pub.publish(safety)
        identity_entry = self._completion_nearest_locked("identity", stamp)
        if identity_entry is None:
            generation, sequence, render_id = -1, -1, -1
        else:
            generation, sequence, render_id = identity_entry[1]
        self._completion_lidar_stamp = stamp
        self._last_processed = (generation, sequence)
        self._last_processed_stamp = stamp
        self.counts["frames"] += 1
        self.counts["completion_lidar_groups"] += 1
        self.counts["completion_nearest_tf_matches"] += 1
        self.counts["lidar_input_points"] += counts["input"]
        self.counts.setdefault("lidar_finite_input_points", 0)
        self.counts["lidar_finite_input_points"] += counts["finite_input"]
        self.counts["lidar_output_points"] += counts["output"]
        self.counts["lidar_self_filtered_points"] += counts["self"]
        self.counts["lidar_range_filtered_points"] += counts["range"]
        self.counts["lidar_height_filtered_points"] += counts["height"]
        self.counts["safety_frames"] += 1
        self.frame_writer.offer(
            {
                "schema_version": 2,
                "runtime_policy": "completion_sim",
                "consumer_group": "nav2_lidar_nearest_tf",
                "wall_monotonic_ns": time.monotonic_ns(),
                "stamp_ns": stamp,
                "generation": generation,
                "sequence": sequence,
                "render_id": render_id,
                "nearest_tf_stamp_ns": tf_entry[0],
                "nearest_tf_gap_sec": abs(tf_entry[0] - stamp) / 1_000_000_000.0,
                "nearest_centers_stamp_ns": centers_stamp,
                "center_policy": (
                    "nearest_dynamic_links"
                    if centers_stamp is not None
                    else "completion_chassis_box_base_sphere"
                ),
                "lidar_input_points": counts["input"],
                "lidar_finite_input_points": counts["finite_input"],
                "lidar_output_points": counts["output"],
                "published_topics": [
                    "/go2/lidar/points",
                    "/go2/lidar/points_base",
                    "/go2/safety/points",
                ],
            }
        )

    def _publish_completion_depth_locked(self) -> None:
        depth_entry = self._completion_latest.get("depth")
        if depth_entry is None or depth_entry[0] <= self._completion_depth_stamp:
            return
        stamp, depth = depth_entry
        depth_array = xyz_array_from_message(depth)
        safety = _cloud(depth.header, finite_xyz_rows(depth_array))
        self.safety_pub.publish(safety)
        self._completion_depth_stamp = stamp
        self.counts["completion_depth_groups"] += 1
        self.counts["safety_frames"] += 1

    def _publish_completion_front_locked(self) -> None:
        front_entry = self._completion_latest.get("front")
        if front_entry is None or front_entry[0] <= self._completion_front_stamp:
            return
        stamp, front = front_entry
        info_entry = self._completion_nearest_locked("front_info", stamp)
        if info_entry is None:
            return
        self.front_pub.publish(front)
        self.front_info_pub.publish(info_entry[1])
        self._completion_front_stamp = stamp
        self.counts["completion_front_groups"] += 1

    def _on_clock(self, message: Clock) -> None:
        stamp = _clock_ns(message)
        self._ingest("clock", stamp, message)

    def _on_tf(self, message: TFMessage) -> None:
        matches = [
            transform
            for transform in message.transforms
            if transform.header.frame_id == "odom" and transform.child_frame_id == "base_link"
        ]
        if len(matches) != 1:
            self._fail("dynamic TF batch lacks exactly one odom->base_link transform")
            return
        self._ingest("tf", _stamp_ns(matches[0]), matches[0])

    def _on_identity(self, message: Vector3Stamped) -> None:
        try:
            generation = int(message.vector.x)
            sequence = int(message.vector.y)
            render_id = int(message.vector.z)
            if (
                float(generation) != message.vector.x
                or float(sequence) != message.vector.y
                or float(render_id) != message.vector.z
                or generation < 0
                or sequence < 0
                or render_id <= 0
                or message.header.frame_id != "base_link"
            ):
                raise ValueError("invalid bridge identity message")
            stamp = _stamp_ns(message)
            if self.runtime_policy.name == "completion_sim":
                self._ingest("identity", stamp, (generation, sequence, render_id))
                return
            with self._lock:
                if (
                    self.runtime_policy.name == "completion_sim"
                    and stamp <= max(
                        self._last_processed_stamp, self._discarded_through_stamp
                    )
                ):
                    self.counts["late_parts_dropped"] += 1
                    return
                if self._identity_generation < 0:
                    if (generation, sequence) != (0, 0):
                        raise RuntimeError("bridge stream did not begin at generation 0 sequence 0")
                elif generation == self._identity_generation:
                    if sequence <= self._identity_sequence:
                        raise RuntimeError("duplicate/replayed bridge identity")
                elif generation == self._identity_generation + 1:
                    if sequence != 0:
                        raise RuntimeError("new bridge generation did not begin at sequence 0")
                else:
                    raise RuntimeError("bridge generation contamination or jump")
                if stamp <= self._identity_stamp:
                    raise RuntimeError("bridge identity sim time did not globally increase")
                entry = self._entry(stamp)
                if entry is None:
                    return
                if "identity" in entry["parts"]:
                    raise RuntimeError("duplicate bridge identity")
                # Commit accepted identity state and batch membership together.
                # Earlier partial batches remain bounded by the watchdog and
                # are always drained before this one can publish.
                entry["parts"]["identity"] = (generation, sequence, render_id)
                self._identity_generation = generation
                self._identity_sequence = sequence
                self._identity_stamp = stamp
                self._drain_ready()
        except BaseException as exc:
            self._fail(f"{type(exc).__name__}: {exc}")

    def _on_centers(self, message: PoseArray) -> None:
        if message.header.frame_id != "base_link" or len(message.poses) != 13:
            self._fail("bridge requires 13 same-stamp base/link centers")
            return
        centers = [
            (float(pose.position.x), float(pose.position.y), float(pose.position.z))
            for pose in message.poses
        ]
        if not all(all(math.isfinite(value) for value in center) for center in centers):
            self._fail("bridge self-filter centers must all be finite")
            return
        self._ingest("centers", _stamp_ns(message), centers)

    def _on_lidar(self, message: PointCloud2) -> None:
        if message.header.frame_id != "go2_l1_lidar":
            self._fail("raw LiDAR frame mismatch")
            return
        self._ingest("lidar", _stamp_ns(message), message)

    def _on_depth(self, message: PointCloud2) -> None:
        if message.header.frame_id != "base_link":
            self._fail("depth geometry frame mismatch")
            return
        self._ingest("depth", _stamp_ns(message), message)

    def _on_front(self, message: Image) -> None:
        if (
            message.header.frame_id != "go2_front_rgb_optical_frame"
            or int(message.width) != 160
            or int(message.height) != 120
            or message.encoding != "rgb8"
        ):
            self._fail("front image violates frozen contract")
            return
        self._ingest("front", _stamp_ns(message), message)

    def _on_front_info(self, message: CameraInfo) -> None:
        try:
            if (
                message.header.frame_id != "go2_front_rgb_optical_frame"
                or int(message.width) != 160
                or int(message.height) != 120
                or message.distortion_model != "plumb_bob"
            ):
                raise ValueError("front CameraInfo header violates frozen contract")
            require_camera_matrices(
                {
                    "d": list(message.d),
                    "k": list(message.k),
                    "r": list(message.r),
                    "p": list(message.p),
                },
                matrices_from_fov(160, 120, 120.0, 75.0),
            )
        except (TypeError, ValueError):
            self._fail("front CameraInfo violates frozen contract")
            return
        self._ingest("front_info", _stamp_ns(message), message)

    def _self_return(self, point: tuple[float, float, float], centers: list[tuple[float, float, float]]) -> bool:
        x, y, z = point
        if abs(x) <= 0.36 and abs(y) <= 0.20 and abs(z) <= 0.16:
            return True
        return any((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= 0.12**2 for cx, cy, cz in centers)

    def _filter_lidar(
        self, message: PointCloud2, centers: list[tuple[float, float, float]]
    ) -> tuple[PointCloud2, PointCloud2, dict[str, int]]:
        points = xyz_array_from_message(message)
        sensor_output, base_output, counts = filter_lidar_xyz(
            points,
            centers,
            minimum_range=self.minimum_range,
            maximum_range=self.maximum_range,
            minimum_height=self.minimum_height,
            maximum_height=self.maximum_height,
        )
        sensor_header = message.header
        base_header = type(message.header)()
        base_header.stamp = message.header.stamp
        base_header.frame_id = "base_link"
        return (
            _cloud(sensor_header, sensor_output),
            _cloud(base_header, base_output),
            counts,
        )

    def _drain_ready(self) -> None:
        """Publish only the earliest pending stamp, never a later completed batch."""

        while self._pending:
            stamp = min(self._pending)
            entry = self._pending[stamp]
            if set(entry["parts"]) != REQUIRED_PARTS:
                return
            self._publish_complete(stamp, entry)

    def _publish_complete(self, stamp: int, entry: dict[str, Any]) -> None:
        parts = entry["parts"]
        generation, sequence, render_id = parts["identity"]
        lidar, lidar_base, counts = self._filter_lidar(parts["lidar"], parts["centers"])
        depth_array = xyz_array_from_message(parts["depth"])
        lidar_base_array = xyz_array_from_message(lidar_base)
        safety_points = finite_xyz_rows(depth_array, lidar_base_array)
        safety = _cloud(lidar_base.header, safety_points)
        self.lidar_pub.publish(lidar)
        self.lidar_base_pub.publish(lidar_base)
        self.safety_pub.publish(safety)
        self.front_pub.publish(parts["front"])
        self.front_info_pub.publish(parts["front_info"])
        self._last_processed = (generation, sequence)
        self._last_processed_stamp = stamp
        self.counts["frames"] += 1
        self.counts["lidar_input_points"] += counts["input"]
        self.counts.setdefault("lidar_finite_input_points", 0)
        self.counts["lidar_finite_input_points"] += counts["finite_input"]
        self.counts["lidar_output_points"] += counts["output"]
        self.counts["lidar_self_filtered_points"] += counts["self"]
        self.counts["lidar_range_filtered_points"] += counts["range"]
        self.counts["lidar_height_filtered_points"] += counts["height"]
        self.counts["safety_frames"] += 1
        self.frame_writer.offer(
            {
                "schema_version": 2,
                "wall_monotonic_ns": time.monotonic_ns(),
                "stamp_ns": stamp,
                "generation": generation,
                "sequence": sequence,
                "render_id": render_id,
                "matched_parts": sorted(REQUIRED_PARTS),
                "clock_same_stamp_observed": True,
                "dynamic_tf_same_stamp_observed": True,
                "link_center_count": len(parts["centers"]),
                "lidar_input_points": counts["input"],
                "lidar_finite_input_points": counts["finite_input"],
                "lidar_output_points": counts["output"],
                "safety_point_count": int(safety_points.shape[0]),
                "published_topics": [
                    "/go2/lidar/points",
                    "/go2/lidar/points_base",
                    "/go2/safety/points",
                    "/go2/front_rgb/image_raw",
                    "/go2/front_rgb/camera_info",
                ],
            }
        )
        del self._pending[stamp]

    def _discard_partial_locked(
        self, stamp: int, entry: dict[str, Any], reason: str
    ) -> None:
        parts = sorted(entry["parts"])
        warning = {
            "schema_version": 1,
            "level": "WARN",
            "runtime_policy": self.runtime_policy.name,
            "reason": reason,
            "stamp_ns": stamp,
            "age_sec": max(0.0, time.monotonic() - float(entry["created_monotonic"])),
            "observed_parts": parts,
            "missing_parts": sorted(REQUIRED_PARTS - set(parts)),
            "action": "record_and_drop_partial_then_continue",
        }
        self._warnings.append(warning)
        atomic_write_json(
            self.result_dir / "bridge_warnings.json",
            {
                "schema_version": 1,
                "status": "RECORDED",
                "runtime_policy": self.runtime_policy.name,
                "items": list(self._warnings),
            },
        )
        del self._pending[stamp]
        self._discarded_through_stamp = max(self._discarded_through_stamp, stamp)
        self.counts["partial_batches_dropped"] += 1

    def _drain_or_discard_completion_locked(
        self, *, reason: str, require_expired: bool
    ) -> None:
        while self._pending:
            self._drain_ready()
            if not self._pending:
                return
            stamp = min(self._pending)
            entry = self._pending[stamp]
            if (
                require_expired
                and time.monotonic() - float(entry["created_monotonic"])
                < self.timeout
            ):
                return
            self._discard_partial_locked(stamp, entry, reason)

    def _watchdog(self) -> None:
        try:
            with self._lock:
                if self.frame_writer.fault or not self.frame_writer.thread.is_alive():
                    raise RuntimeError(self.frame_writer.fault or "bridge writer died")
                now = time.monotonic()
                if self.runtime_policy.name == "completion_sim":
                    self._drain_or_discard_completion_locked(
                        reason="completion_partial_timeout", require_expired=True
                    )
                else:
                    expired = [
                        entry
                        for entry in self._pending.values()
                        if now - float(entry["created_monotonic"]) >= self.timeout
                    ]
                if self.runtime_policy.name != "completion_sim" and expired:
                    entry = expired[0]
                    raise RuntimeError(
                        f"bridge batch incomplete after {self.timeout:.2f} seconds: {sorted(entry['parts'])}"
                    )
        except BaseException as exc:
            self._fail(f"{type(exc).__name__}: {exc}")

    def _snapshot(self) -> None:
        request_path = self.result_dir.parent / "snapshot_request.json"
        ack_path = self.result_dir.parent / "bridge_snapshot_ack.json"
        with self._lock:
            if not request_path.is_file() or ack_path.exists() or self._fatal_reason:
                return
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            target = (int(request["generation"]), int(request["sequence"]))
            with self._lock:
                if self.runtime_policy.name == "completion_sim":
                    self._drain_or_discard_completion_locked(
                        reason="completion_snapshot_freeze", require_expired=False
                    )
                    if self._last_processed is None:
                        return
                elif self._last_processed != target or self._pending:
                    return
                count = self.frame_writer.flush()
                writer_alive = self.frame_writer.thread.is_alive()
                writer_fault = self.frame_writer.fault or None
                selected = self._last_processed
                pending_count = len(self._pending)
            atomic_write_json(
                ack_path,
                {
                    "schema_version": 2,
                    "status": "PASS",
                    "runtime_policy": self.runtime_policy.name,
                    "snapshot_id": request["snapshot_id"],
                    "generation": selected[0],
                    "sequence": selected[1],
                    "producer_target": list(target),
                    "target_matched": selected == target,
                    "frame_count": count,
                    "pending_count": pending_count,
                    "partial_batches_dropped": self.counts["partial_batches_dropped"],
                    "warning_count": len(self._warnings),
                    "writer_thread_alive": writer_alive,
                    "writer_fault": writer_fault,
                },
            )
        except BaseException as exc:
            self._fail(f"bridge_snapshot_failed: {type(exc).__name__}: {exc}")

    def _summary(self) -> None:
        try:
            with self._lock:
                summary = {
                    "status": "FAIL" if self._fatal_reason else "RUNNING",
                    "fatal_reason": self._fatal_reason or None,
                    "last_processed": self._last_processed,
                    "discarded_through_stamp": self._discarded_through_stamp,
                    "warning_count": len(self._warnings),
                    "counts": dict(self.counts),
                }
            atomic_write_json(
                self.result_dir / "go2_sensor_bridge_summary.json",
                {
                    "schema_version": 2,
                    **summary,
                },
            )
        except BaseException as exc:
            self._fail(f"summary: {type(exc).__name__}: {exc}")

    def close(self) -> None:
        self.frame_writer.close()


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node: Go2SensorBridge | None = None
    executor: MultiThreadedExecutor | None = None
    errors: list[str] = []
    try:
        node = Go2SensorBridge()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except BaseException as exc:
        errors.append(f"spin: {type(exc).__name__}: {exc}")
        if node is not None:
            node._fail(errors[-1])
    finally:
        for name, action in (
            ("executor_shutdown", None if executor is None else executor.shutdown),
            ("writer_close", None if node is None else node.close),
            ("destroy_node", None if node is None else node.destroy_node),
            ("rclpy_shutdown", (lambda: rclpy.shutdown()) if rclpy.ok() else None),
        ):
            if action is None:
                continue
            try:
                action()
            except BaseException as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return 2 if errors or (node is not None and node._fatal_reason) else 0


if __name__ == "__main__":
    raise SystemExit(main())
