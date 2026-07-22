#!/usr/bin/env python3
"""Independent recorder for actual ROS outputs of the frozen sensor chain."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import numpy as np
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import rclpy
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.clock import Clock as RclpyClock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2
from std_msgs.msg import UInt64
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer

from .atomic import atomic_create_json, atomic_write_json
from .contract import FRAMES
from .core import SensorFault, assert_unpaired_generation_event_fresh
from .graph_handshake import (
    ingest_observed_transforms,
    make_exact_ros_time,
    make_steady_control_clock,
    qos_snapshot,
    summarize_pending_batches,
)
from .runtime_policy import RuntimePolicy, require_policy


def _stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _camera_contract(
    message: CameraInfo,
    *,
    width: int,
    height: int,
    frame: str,
    hfov: float,
    vfov: float,
) -> dict[str, Any]:
    fx = width / (2.0 * math.tan(math.radians(hfov) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(vfov) / 2.0))
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    expected_k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    expected_d = [0.0] * 5
    expected_r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    expected_p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    observed = {
        "width": int(message.width),
        "height": int(message.height),
        "frame_id": message.header.frame_id,
        "distortion_model": message.distortion_model,
        "d": list(message.d),
        "k": list(message.k),
        "r": list(message.r),
        "p": list(message.p),
    }
    if observed["width"] != width or observed["height"] != height:
        raise ValueError("CameraInfo dimensions violate frozen contract")
    if observed["frame_id"] != frame or observed["distortion_model"] != "plumb_bob":
        raise ValueError("CameraInfo frame/distortion contract mismatch")
    for name, actual, expected in (
        ("D", observed["d"], expected_d),
        ("K", observed["k"], expected_k),
        ("R", observed["r"], expected_r),
        ("P", observed["p"], expected_p),
    ):
        if len(actual) != len(expected) or not all(
            math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-5)
            for a, b in zip(actual, expected)
        ):
            raise ValueError(f"CameraInfo {name} violates frozen contract")
    return observed


class DownstreamRecorder(Node):
    REQUIRED = {
        "clock",
        "tf",
        "identity",
        "generation_topic",
        "rgb",
        "rgb_info",
        "depth",
        "depth_info",
        "depth_points",
        "lidar",
        "lidar_base",
        "front",
        "front_info",
        "imu",
        "safety",
        "odom",
    }
    STATIC_CHILDREN = {
        FRAMES["lidar"],
        FRAMES["imu"],
        FRAMES["d435i_depth"],
        FRAMES["d435i_color"],
        FRAMES["front_rgb"],
    }
    TF_BUFFER_AUTHORITY = "internnav_downstream_recorder_observed_tf"
    CONSUMER_GROUPS = {
        "nav2": (("lidar",), FRAMES["lidar"]),
        "nvblox_depth": (("depth", "depth_info"), FRAMES["d435i_depth"]),
        "nvblox_lidar": (("lidar",), FRAMES["lidar"]),
        "internvla": (("rgb", "odom"), FRAMES["d435i_color"]),
    }
    FUNCTIONAL_GROUPS = frozenset({"nav2", "internvla"})

    def __init__(self, result_dir: Path, runtime_policy: str = "strict_evidence") -> None:
        super().__init__("internav_downstream_recorder")
        self.policy: RuntimePolicy = require_policy(runtime_policy)
        self.completion_shadow = self.policy.name == "completion_sim"
        self.result_dir = result_dir.resolve()
        self.result_dir.mkdir(parents=False, exist_ok=False)
        self.frames_path = self.result_dir / "downstream_frames.jsonl"
        self._stream: Any = None
        self._warning_stream: Any = None
        self._lock = threading.RLock()
        self._pending: dict[int, dict[str, Any]] = {}
        self._static_by_stamp: dict[int, set[str]] = {}
        self._generation_events: deque[tuple[int, int]] = deque()
        self._unpaired_identities: deque[int] = deque()
        self._last_accepted_identity: tuple[int, int] | None = None
        self._last_completed_identity: tuple[int, int] | None = None
        self._last_stamp = 0
        self._written = 0
        self._fatal_reason = ""
        self._failure_written = False
        self._closed = False
        self._tf_lookup_exception_count = 0
        self._latest: dict[str, dict[str, Any]] = {}
        self._shadow_groups_by_identity: dict[tuple[int, int], set[str]] = {}
        self._shadow_group_counts = {name: 0 for name in self.CONSUMER_GROUPS}
        self._warned_shadow_identities: set[tuple[int, int]] = set()
        self._warning_count = 0
        self.tf_buffer = Buffer()
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=8,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Clock, "/clock", self._on_clock, qos)
        self.create_subscription(TFMessage, "/tf", self._tf, qos)
        self.create_subscription(TFMessage, "/tf_static", self._tf_static, static_qos)
        self.create_subscription(UInt64, "/internnav/sensor_generation", self._generation, qos)
        self.create_subscription(Vector3Stamped, "/internnav/sensor_frame_identity", self._identity, qos)
        specs: tuple[tuple[str, Any, str, Callable[[Any], dict[str, Any] | None]], ...] = (
            ("rgb", Image, "/go2/d435i/color/image_raw", self._rgb),
            ("rgb_info", CameraInfo, "/go2/d435i/color/camera_info", self._rgb_info),
            ("depth", Image, "/go2/d435i/depth/image_rect", self._depth),
            ("depth_info", CameraInfo, "/go2/d435i/depth/camera_info", self._depth_info),
            ("depth_points", PointCloud2, "/go2/depth/points", lambda m: self._cloud(m, FRAMES["base"])),
            ("lidar", PointCloud2, "/go2/lidar/points", lambda m: self._cloud(m, FRAMES["lidar"])),
            ("lidar_base", PointCloud2, "/go2/lidar/points_base", lambda m: self._cloud(m, FRAMES["base"])),
            ("front", Image, "/go2/front_rgb/image_raw", self._front),
            ("front_info", CameraInfo, "/go2/front_rgb/camera_info", self._front_info),
            ("imu", Imu, "/go2/imu/data", self._imu),
            ("safety", PointCloud2, "/go2/safety/points", lambda m: self._cloud(m, FRAMES["base"])),
            ("odom", Odometry, "/odom", self._odom),
        )
        for name, message_type, topic, validator in specs:
            self.create_subscription(
                message_type,
                topic,
                lambda message, n=name, v=validator: self._message(n, message, v),
                qos,
            )
        self._control_clock = make_steady_control_clock(RclpyClock, ClockType)
        self.create_timer(0.02, self._watchdog, clock=self._control_clock)
        self.create_timer(0.02, self._snapshot, clock=self._control_clock)
        # Open the only non-ROS resource last; subscription construction
        # failure therefore cannot leak an evidence stream.
        self._stream = self.frames_path.open("x", encoding="utf-8", newline="\n")
        if self.completion_shadow:
            self._warning_stream = (self.result_dir / "downstream_warnings.jsonl").open(
                "x", encoding="utf-8", newline="\n"
            )
        atomic_create_json(
            self.result_dir.parent / "downstream_role_ready.json",
            {
                "schema_version": 2,
                "status": "ROLE_READY",
                "role": "downstream_recorder",
                "runtime_policy": self.policy.as_dict(),
                "shadow_nonfatal": self.completion_shadow,
                "pid": os.getpid(),
                "control_timer_clock_type": "STEADY_TIME",
                "tf_query_clock_type": "ROS_TIME",
                "tf_buffer_ingestion": {
                    "source": "manual_observed_callbacks",
                    "dynamic": "set_transform",
                    "static": "set_transform_static",
                    "authority": self.TF_BUFFER_AUTHORITY,
                },
                "owned_qos": {
                    "dynamic": qos_snapshot(qos),
                    "static": qos_snapshot(static_qos),
                },
            },
        )

    @staticmethod
    def _header(message: Any, frame: str) -> dict[str, Any]:
        if message.header.frame_id != frame:
            raise ValueError(f"unexpected frame {message.header.frame_id!r}, expected {frame!r}")
        return {"frame_id": message.header.frame_id}

    def _rgb(self, message: Image) -> dict[str, Any]:
        if (message.width, message.height, message.encoding) != (640, 480, "rgb8"):
            raise ValueError("RGB image contract mismatch")
        if int(message.step) != 640 * 3 or len(message.data) != int(message.step) * 480:
            raise ValueError("RGB image step/data length mismatch")
        return {**self._header(message, FRAMES["d435i_color"]), "width": 640, "height": 480, "encoding": "rgb8"}

    def _depth(self, message: Image) -> dict[str, Any]:
        if (message.width, message.height, message.encoding) != (640, 480, "32FC1"):
            raise ValueError("depth image contract mismatch")
        if int(message.step) != 640 * 4 or len(message.data) != int(message.step) * 480:
            raise ValueError("depth image step/data length mismatch")
        return {**self._header(message, FRAMES["d435i_depth"]), "width": 640, "height": 480, "encoding": "32FC1"}

    def _front(self, message: Image) -> dict[str, Any]:
        if (message.width, message.height, message.encoding) != (160, 120, "rgb8"):
            raise ValueError("front image contract mismatch")
        if int(message.step) != 160 * 3 or len(message.data) != int(message.step) * 120:
            raise ValueError("front image step/data length mismatch")
        return {**self._header(message, FRAMES["front_rgb"]), "width": 160, "height": 120, "encoding": "rgb8"}

    def _rgb_info(self, message: CameraInfo) -> dict[str, Any]:
        return _camera_contract(message, width=640, height=480, frame=FRAMES["d435i_color"], hfov=69.4, vfov=42.5)

    def _depth_info(self, message: CameraInfo) -> dict[str, Any]:
        return _camera_contract(message, width=640, height=480, frame=FRAMES["d435i_depth"], hfov=87.0, vfov=58.0)

    def _front_info(self, message: CameraInfo) -> dict[str, Any]:
        return _camera_contract(message, width=160, height=120, frame=FRAMES["front_rgb"], hfov=120.0, vfov=75.0)

    def _odom(self, message: Odometry) -> dict[str, Any]:
        if message.child_frame_id != FRAMES["base"]:
            raise ValueError("odometry child frame mismatch")
        values = (
            message.pose.pose.position.x, message.pose.pose.position.y, message.pose.pose.position.z,
            message.pose.pose.orientation.x, message.pose.pose.orientation.y,
            message.pose.pose.orientation.z, message.pose.pose.orientation.w,
            message.twist.twist.linear.x, message.twist.twist.linear.y, message.twist.twist.linear.z,
            message.twist.twist.angular.x, message.twist.twist.angular.y, message.twist.twist.angular.z,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("odometry contains non-finite values")
        return {**self._header(message, FRAMES["odom"]), "child_frame_id": message.child_frame_id}

    def _imu(self, message: Imu) -> dict[str, Any]:
        values = (
            message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w,
            message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z,
            message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("IMU contains non-finite values")
        return self._header(message, FRAMES["imu"])

    def _cloud(self, message: PointCloud2, frame: str) -> dict[str, Any]:
        fields = [(item.name, int(item.offset), int(item.datatype), int(item.count)) for item in message.fields]
        expected = [("x", 0, 7, 1), ("y", 4, 7, 1), ("z", 8, 7, 1)]
        width, height = int(message.width), int(message.height)
        if fields != expected or message.is_bigendian or int(message.point_step) != 12:
            raise ValueError("PointCloud2 field/endianness/point_step contract mismatch")
        if width <= 0 or height <= 0 or int(message.row_step) != width * 12:
            raise ValueError("PointCloud2 dimensions/row_step are invalid")
        if len(message.data) != int(message.row_step) * height:
            raise ValueError("PointCloud2 data length is truncated or padded")
        points = np.frombuffer(memoryview(message.data), dtype="<f4").reshape(height, width, 3)
        finite_count = int(np.count_nonzero(np.isfinite(points).all(axis=2)))
        if finite_count <= 0:
            raise ValueError("PointCloud2 contains no finite xyz sample")
        return {
            **self._header(message, frame),
            "width": width,
            "height": height,
            "point_count": width * height,
            "finite_point_count": finite_count,
        }

    def _entry(self, stamp: int) -> dict[str, Any]:
        if stamp <= self._last_stamp:
            raise RuntimeError("late downstream message crossed completed boundary")
        return self._pending.setdefault(stamp, {"created": time.monotonic(), "parts": {}, "details": {}})

    def _put(self, name: str, stamp: int, detail: dict[str, Any] | None = None) -> None:
        with self._lock:
            if self.completion_shadow:
                self._latest[name] = {
                    "stamp_ns": stamp,
                    "detail": detail,
                    "received_monotonic": time.monotonic(),
                }
                self._drain_consumer_groups_locked()
                return
            entry = self._entry(stamp)
            if name in entry["parts"]:
                raise RuntimeError(f"duplicate downstream part {name}")
            entry["parts"][name] = True
            if detail is not None:
                entry["details"][name] = detail
            self._drain_ready_locked()

    def _message(self, name: str, message: Any, validator: Callable[[Any], dict[str, Any] | None]) -> None:
        try:
            detail = validator(message)
            self._put(name, _stamp_ns(message), detail)
        except BaseException as exc:
            self._fail(f"{name}: {type(exc).__name__}: {exc}")

    def _on_clock(self, message: Clock) -> None:
        try:
            stamp = int(message.clock.sec) * 1_000_000_000 + int(message.clock.nanosec)
            self._put("clock", stamp)
        except BaseException as exc:
            self._fail(f"clock: {type(exc).__name__}: {exc}")

    def _tf(self, message: TFMessage) -> None:
        try:
            matches = [t for t in message.transforms if t.header.frame_id == FRAMES["odom"] and t.child_frame_id == FRAMES["base"]]
            if len(matches) != 1:
                raise ValueError("requires exactly one observed odom->base_link transform")
            ingest_observed_transforms(
                self.tf_buffer,
                matches,
                authority=self.TF_BUFFER_AUTHORITY,
                static=False,
            )
            self._put("tf", _stamp_ns(matches[0]), {"lookup": f"{FRAMES['odom']}->{FRAMES['base']}"})
        except BaseException as exc:
            self._fail(f"tf: {type(exc).__name__}: {exc}")

    def _tf_static(self, message: TFMessage) -> None:
        try:
            stamps = {_stamp_ns(item) for item in message.transforms}
            children = {item.child_frame_id for item in message.transforms if item.header.frame_id == FRAMES["base"]}
            if len(stamps) != 1 or children != self.STATIC_CHILDREN:
                raise ValueError("static TF batch violates frozen frame set/stamp")
            ingest_observed_transforms(
                self.tf_buffer,
                message.transforms,
                authority=self.TF_BUFFER_AUTHORITY,
                static=True,
            )
            stamp = stamps.pop()
            with self._lock:
                if self.completion_shadow:
                    self._latest["tf_static"] = {
                        "stamp_ns": stamp,
                        "detail": {"children": sorted(children)},
                        "received_monotonic": time.monotonic(),
                    }
                    self._drain_consumer_groups_locked()
                    return
                entry = self._entry(stamp)
                if "tf_static" in entry["parts"]:
                    raise RuntimeError("duplicate static TF batch")
                self._static_by_stamp[stamp] = children
                entry["parts"]["tf_static"] = True
                entry["details"]["tf_static"] = {"children": sorted(children)}
                self._drain_ready_locked()
        except BaseException as exc:
            self._fail(f"tf_static: {type(exc).__name__}: {exc}")

    def _generation(self, message: UInt64) -> None:
        try:
            with self._lock:
                if self.completion_shadow:
                    self._latest["generation_topic"] = {
                        "stamp_ns": 0,
                        "detail": {"observed_generation": int(message.data)},
                        "received_monotonic": time.monotonic(),
                    }
                    return
                self._generation_events.append((int(message.data), time.monotonic_ns()))
                self._pair_generations_locked()
                self._drain_ready_locked()
        except BaseException as exc:
            self._fail(f"generation_topic: {type(exc).__name__}: {exc}")

    def _pair_generations_locked(self) -> None:
        while self._generation_events and self._unpaired_identities:
            observed, wall_ns = self._generation_events.popleft()
            stamp = self._unpaired_identities.popleft()
            entry = self._pending[stamp]
            identity = entry["details"]["identity"]
            if observed != int(identity["generation"]):
                raise RuntimeError("unstamped generation topic disagrees with ordered identity")
            entry["parts"]["generation_topic"] = True
            entry["details"]["generation_topic"] = {
                "observed_generation": observed,
                "ordered_pair_wall_monotonic_ns": wall_ns,
            }

    def _identity(self, message: Vector3Stamped) -> None:
        try:
            generation, sequence, render_id = int(message.vector.x), int(message.vector.y), int(message.vector.z)
            if (float(generation), float(sequence), float(render_id)) != (message.vector.x, message.vector.y, message.vector.z):
                raise ValueError("non-integral downstream identity")
            if generation < 0 or sequence < 0 or render_id <= 0:
                raise ValueError("negative identity or non-positive render identity")
            stamp = _stamp_ns(message)
            with self._lock:
                if self.completion_shadow:
                    identity = {
                        "generation": generation,
                        "sequence": sequence,
                        "render_id": render_id,
                        "sequence_gap_from_previous_observed": 0,
                    }
                    self._latest["identity"] = {
                        "stamp_ns": stamp,
                        "detail": identity,
                        "received_monotonic": time.monotonic(),
                    }
                    self._last_accepted_identity = (generation, sequence)
                    self._drain_consumer_groups_locked()
                    return
                previous = self._last_accepted_identity
                if previous is None and (generation, sequence) != (0, 0):
                    raise RuntimeError("downstream did not begin at generation 0 sequence 0")
                sequence_gap = 0
                if previous is not None:
                    if generation == previous[0]:
                        if sequence <= previous[1]:
                            raise RuntimeError("downstream sequence replay/rollback")
                        sequence_gap = sequence - previous[1] - 1
                    elif generation == previous[0] + 1:
                        if sequence != 0:
                            raise RuntimeError("new downstream generation lacks sequence-zero barrier")
                    else:
                        raise RuntimeError("downstream generation contamination or jump")
                entry = self._entry(stamp)
                if "identity" in entry["parts"]:
                    raise RuntimeError("duplicate downstream identity")
                entry["parts"]["identity"] = True
                entry["details"]["identity"] = {
                    "generation": generation,
                    "sequence": sequence,
                    "render_id": render_id,
                    "sequence_gap_from_previous_observed": sequence_gap,
                }
                self._last_accepted_identity = (generation, sequence)
                self._unpaired_identities.append(stamp)
                self._pair_generations_locked()
                self._drain_ready_locked()
        except BaseException as exc:
            self._fail(f"identity: {type(exc).__name__}: {exc}")

    def _shadow_tf_lookup_locked(
        self, source_frame: str, anchor_stamp: int
    ) -> dict[str, Any] | None:
        query_time = Time()
        try:
            if not self.tf_buffer.can_transform(FRAMES["odom"], source_frame, query_time):
                return None
            observed = self.tf_buffer.lookup_transform(
                FRAMES["odom"], source_frame, query_time
            )
        except BaseException as exc:
            self._tf_lookup_exception_count += 1
            self._warn_locked(
                f"nearest/latest TF unavailable for {source_frame}: "
                f"{type(exc).__name__}: {exc}"
            )
            return None
        tf_stamp = _stamp_ns(observed)
        age_sec = abs(anchor_stamp - tf_stamp) / 1_000_000_000.0
        if age_sec > self.policy.transform_tolerance_sec:
            return None
        return {
            "mode": "nearest_or_latest",
            "target_frame": FRAMES["odom"],
            "source_frame": source_frame,
            "anchor_stamp_ns": anchor_stamp,
            "transform_stamp_ns": tf_stamp,
            "age_sec": age_sec,
            "tolerance_sec": self.policy.transform_tolerance_sec,
            "fallback_source": "isaac_ground_truth_pose",
        }

    def _drain_consumer_groups_locked(self) -> None:
        if not self.completion_shadow:
            return
        identity_row = self._latest.get("identity")
        if identity_row is None:
            return
        identity = identity_row["detail"]
        key = (int(identity["generation"]), int(identity["sequence"]))
        completed = self._shadow_groups_by_identity.setdefault(key, set())
        anchor_stamp = int(identity_row["stamp_ns"])
        for group, (required_parts, source_frame) in self.CONSUMER_GROUPS.items():
            if group in completed:
                continue
            rows = {name: self._latest.get(name) for name in required_parts}
            if any(value is None for value in rows.values()):
                continue
            part_stamps = {
                name: int(value["stamp_ns"])
                for name, value in rows.items()
                if value is not None
            }
            max_skew_sec = max(
                abs(anchor_stamp - stamp) / 1_000_000_000.0
                for stamp in part_stamps.values()
            )
            if max_skew_sec > self.policy.transform_tolerance_sec:
                continue
            lookup = self._shadow_tf_lookup_locked(source_frame, anchor_stamp)
            if lookup is None:
                continue
            optional_parts: dict[str, int] = {}
            if group == "internvla" and "depth" in self._latest:
                optional_depth_stamp = int(self._latest["depth"]["stamp_ns"])
                if (
                    abs(anchor_stamp - optional_depth_stamp) / 1_000_000_000.0
                    <= self.policy.transform_tolerance_sec
                ):
                    optional_parts["depth"] = optional_depth_stamp
            record = {
                "schema_version": 3,
                "runtime_policy": self.policy.name,
                "recorder_mode": self.policy.recorder_mode,
                "wall_monotonic_ns": time.monotonic_ns(),
                "consumer_group": group,
                "anchor_stamp_ns": anchor_stamp,
                **identity,
                "observed_parts": sorted(required_parts),
                "part_stamps_ns": part_stamps,
                "optional_part_stamps_ns": optional_parts,
                "same_stamp_observed": len({anchor_stamp, *part_stamps.values()}) == 1,
                "max_consumer_skew_sec": max_skew_sec,
                "tf_lookup": lookup,
                "pose_history_mode": "nearest_odom_and_identity_history"
                if group == "internvla"
                else None,
            }
            self._stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            self._stream.flush()
            self._written += 1
            self._shadow_group_counts[group] += 1
            completed.add(group)
        if self.FUNCTIONAL_GROUPS.issubset(completed):
            self._last_completed_identity = key
            self._last_stamp = max(self._last_stamp, anchor_stamp)

    def _warn_locked(self, reason: str) -> None:
        self._warning_count += 1
        if self._warning_stream is None:
            return
        self._warning_stream.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "WARN",
                    "runtime_policy": self.policy.name,
                    "reason": reason,
                    "wall_time_unix": time.time(),
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        self._warning_stream.flush()

    def _measured_tf_lookup_locked(self, stamp: int) -> dict[str, Any] | None:
        query_time = make_exact_ros_time(Time, ClockType, stamp)
        frames = [FRAMES["base"], *sorted(self.STATIC_CHILDREN)]
        if not all(self.tf_buffer.can_transform(FRAMES["odom"], frame, query_time) for frame in frames):
            return None
        try:
            for frame in frames:
                self.tf_buffer.lookup_transform(FRAMES["odom"], frame, query_time)
        except BaseException:
            self._tf_lookup_exception_count += 1
            raise
        return {
            "target_frame": FRAMES["odom"],
            "source_frames": frames,
            "stamp_ns": stamp,
            "lookup_success_count": len(frames),
            "lookup_exception_count": self._tf_lookup_exception_count,
        }

    def _drain_ready_locked(self) -> None:
        while self._pending:
            stamp = min(self._pending)
            entry = self._pending[stamp]
            identity = entry["details"].get("identity")
            if identity is None:
                return
            required = self.REQUIRED | ({"tf_static"} if identity["sequence"] == 0 else set())
            if not required.issubset(entry["parts"]):
                return
            if set(entry["parts"]) - required:
                raise RuntimeError("unexpected downstream part at this stamp")
            lookup = self._measured_tf_lookup_locked(stamp)
            if lookup is None:
                return
            self._write_complete_locked(stamp, entry, required, lookup)

    def _write_complete_locked(
        self,
        stamp: int,
        entry: dict[str, Any],
        required: set[str],
        lookup: dict[str, Any],
    ) -> None:
        identity = entry["details"]["identity"]
        record = {
            "schema_version": 2,
            "wall_monotonic_ns": time.monotonic_ns(),
            "stamp_ns": stamp,
            **identity,
            "observed_parts": sorted(required),
            "same_stamp_observed": True,
            "tf_lookup": lookup,
            "static_tf_same_stamp_observed": identity["sequence"] != 0 or stamp in self._static_by_stamp,
            "camera_contracts": {
                "rgb": entry["details"]["rgb_info"],
                "depth": entry["details"]["depth_info"],
                "front": entry["details"]["front_info"],
            },
            "image_contracts": {
                "rgb": entry["details"]["rgb"],
                "depth": entry["details"]["depth"],
                "front": entry["details"]["front"],
            },
            "cloud_contracts": {
                name: entry["details"][name]
                for name in ("depth_points", "lidar", "lidar_base", "safety")
            },
            "generation_topic": entry["details"]["generation_topic"],
        }
        self._stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._stream.flush()
        self._written += 1
        self._last_stamp = stamp
        self._last_completed_identity = (int(identity["generation"]), int(identity["sequence"]))
        del self._pending[stamp]

    def _pending_failure_evidence_locked(self, now_monotonic: float) -> dict[str, Any]:
        rows = summarize_pending_batches(
            self._pending,
            self.REQUIRED,
            now_monotonic=now_monotonic,
        )
        frames = [FRAMES["base"], *sorted(self.STATIC_CHILDREN)]
        for row in rows:
            query_time = make_exact_ros_time(Time, ClockType, int(row["stamp_ns"]))
            availability: dict[str, bool] = {}
            errors: dict[str, str] = {}
            for frame in frames:
                try:
                    availability[frame] = bool(
                        self.tf_buffer.can_transform(FRAMES["odom"], frame, query_time)
                    )
                except BaseException as exc:
                    availability[frame] = False
                    errors[frame] = f"{type(exc).__name__}: {exc}"
            row["tf_can_transform"] = availability
            row["tf_unavailable_frames"] = sorted(
                frame for frame, available in availability.items() if not available
            )
            row["tf_query_errors"] = errors
        return {
            "schema_version": 1,
            "status": "FAIL",
            "watchdog_sec": 0.35,
            "target_frame": FRAMES["odom"],
            "pending": rows,
            "generation_event_queue_count": len(self._generation_events),
            "unpaired_identity_stamps": list(self._unpaired_identities),
            "last_accepted_identity": self._last_accepted_identity,
            "last_completed_identity": self._last_completed_identity,
            "last_completed_stamp_ns": self._last_stamp,
            "tf_lookup_exception_count": self._tf_lookup_exception_count,
        }

    def _watchdog(self) -> None:
        try:
            if self.completion_shadow:
                with self._lock:
                    self._drain_consumer_groups_locked()
                    identity_row = self._latest.get("identity")
                    if identity_row is None:
                        return
                    identity = identity_row["detail"]
                    key = (int(identity["generation"]), int(identity["sequence"]))
                    groups = self._shadow_groups_by_identity.get(key, set())
                    age = time.monotonic() - float(identity_row["received_monotonic"])
                    if (
                        age >= self.policy.downstream_watchdog_sec
                        and not self.FUNCTIONAL_GROUPS.issubset(groups)
                        and key not in self._warned_shadow_identities
                    ):
                        self._warned_shadow_identities.add(key)
                        self._warn_locked(
                            "partial consumer frame retained as shadow: "
                            f"identity={key}, ready_groups={sorted(groups)}"
                        )
                return
            failure_evidence: dict[str, Any] | None = None
            with self._lock:
                self._drain_ready_locked()
                now_monotonic = time.monotonic()
                pending_too_old = any(
                    now_monotonic - float(entry["created"]) >= 0.35
                    for entry in self._pending.values()
                )
                assert_unpaired_generation_event_fresh(
                    self._generation_events, time.monotonic_ns()
                )
                if pending_too_old:
                    failure_evidence = self._pending_failure_evidence_locked(
                        now_monotonic
                    )
            if (self.result_dir.parent / "producer_snapshot_ack.json").is_file():
                return
            if pending_too_old:
                assert failure_evidence is not None
                atomic_write_json(
                    self.result_dir / "downstream_pending_failure.json",
                    failure_evidence,
                )
                oldest = failure_evidence["pending"][0]
                raise TimeoutError(
                    "actual downstream batch incomplete for 0.35 seconds: "
                    f"oldest_stamp={oldest['stamp_ns']}, "
                    f"missing_parts={oldest['missing_parts']}, "
                    f"tf_unavailable={oldest['tf_unavailable_frames']}"
                )
        except BaseException as exc:
            self._fail(f"watchdog: {type(exc).__name__}: {exc}")

    def _snapshot(self) -> None:
        request_path = self.result_dir.parent / "snapshot_request.json"
        ack_path = self.result_dir.parent / "downstream_snapshot_ack.json"
        if not request_path.is_file() or ack_path.exists() or (
            self._fatal_reason and not self.completion_shadow
        ):
            return
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            target = (int(request["generation"]), int(request["sequence"]))
            if self.completion_shadow:
                with self._lock:
                    self._drain_consumer_groups_locked()
                    ready = [
                        (identity, set(groups))
                        for identity, groups in self._shadow_groups_by_identity.items()
                        if self.FUNCTIONAL_GROUPS.issubset(groups)
                    ]
                    if not ready:
                        return
                    selected, groups = max(ready, key=lambda item: item[0])
                    self._stream.flush()
                    os.fsync(self._stream.fileno())
                    if self._warning_stream is not None:
                        self._warning_stream.flush()
                        os.fsync(self._warning_stream.fileno())
                    atomic_write_json(
                        ack_path,
                        {
                            "schema_version": 3,
                            "status": "PASS",
                            "runtime_policy": self.policy.name,
                            "recorder_mode": self.policy.recorder_mode,
                            "snapshot_id": request["snapshot_id"],
                            "generation": selected[0],
                            "sequence": selected[1],
                            "producer_target": list(target),
                            "target_matched": selected == target,
                            "frame_count": self._written,
                            "consumer_group_counts": dict(self._shadow_group_counts),
                            "ready_consumer_groups": sorted(groups),
                            "required_functional_groups": sorted(self.FUNCTIONAL_GROUPS),
                            "pending_count": 0,
                            "generation_event_queue_count": 0,
                            "tf_lookup_exception_count": self._tf_lookup_exception_count,
                            "warning_count": self._warning_count,
                            "recorder_fault": None,
                            "strict_exact_batch": "NOT_EVALUATED",
                        },
                    )
                return
            with self._lock:
                self._drain_ready_locked()
                if (
                    self._last_completed_identity != target
                    or self._pending
                    or self._generation_events
                    or self._unpaired_identities
                ):
                    return
            self._stream.flush()
            os.fsync(self._stream.fileno())
            atomic_write_json(
                ack_path,
                {
                    "schema_version": 2,
                    "status": "PASS",
                    "snapshot_id": request["snapshot_id"],
                    "generation": target[0],
                    "sequence": target[1],
                    "frame_count": self._written,
                    "pending_count": 0,
                    "generation_event_queue_count": 0,
                    "tf_lookup_exception_count": self._tf_lookup_exception_count,
                    "recorder_fault": None,
                },
            )
        except BaseException as exc:
            self._fail(f"snapshot: {type(exc).__name__}: {exc}")

    def _fail(self, reason: str) -> None:
        if self.completion_shadow:
            with self._lock:
                self._warn_locked(reason)
            return
        if not self._fatal_reason:
            self._fatal_reason = reason
        if not self._failure_written:
            try:
                atomic_write_json(
                    self.result_dir / "downstream_first_failure.json",
                    {"schema_version": 2, "status": "FAIL", "reason": self._fatal_reason, "wall_time_unix": time.time()},
                )
                self._failure_written = True
            except BaseException:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except BaseException:
            pass

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._stream is not None:
                self._stream.flush()
                os.fsync(self._stream.fileno())
                self._stream.close()
            if self._warning_stream is not None:
                self._warning_stream.flush()
                os.fsync(self._warning_stream.fileno())
                self._warning_stream.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--runtime-policy",
        choices=("strict_evidence", "completion_sim"),
        default="strict_evidence",
    )
    known, ros_args = parser.parse_known_args(argv)
    rclpy.init(args=ros_args)
    node: DownstreamRecorder | None = None
    executor: MultiThreadedExecutor | None = None
    errors: list[str] = []

    def coordinated_completion_shutdown() -> bool:
        return bool(
            node is not None
            and node.completion_shadow
            and (node.result_dir.parent / "inner_stop.request").is_file()
            and not rclpy.ok()
        )

    try:
        node = DownstreamRecorder(known.result_dir, known.runtime_policy)
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except BaseException as exc:
        if not coordinated_completion_shutdown():
            errors.append(f"spin: {type(exc).__name__}: {exc}")
            if node is not None:
                node._fail(errors[-1])
    finally:
        for name, action in (
            ("executor_shutdown", None if executor is None else executor.shutdown),
            ("recorder_close", None if node is None else node.close),
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
    return 2 if errors or (node is not None and node._fatal_reason) else 0


if __name__ == "__main__":
    raise SystemExit(main())
