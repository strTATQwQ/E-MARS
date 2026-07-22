#!/usr/bin/env python3
"""No-GPU ROS graph probe for the Lane-B live-frontier capture sidecar."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import rclpy
from builtin_interfaces.msg import Time as TimeMessage
from internvla_ros2_msgs.msg import ObservationMetadata
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped


def stamp(value: float) -> TimeMessage:
    message = TimeMessage()
    message.sec = int(value)
    message.nanosec = round((value - int(value)) * 1_000_000_000)
    return message


def wait_for(predicate: Any, *, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class Probe(Node):
    def __init__(self) -> None:
        super().__init__("live_frontier_capture_probe", namespace="/t5/lane_b")
        reliable = QoSProfile(depth=8, reliability=ReliabilityPolicy.RELIABLE)
        self.clock_publisher = self.create_publisher(Clock, "/clock", reliable)
        self.tf_publisher = self.create_publisher(
            TFMessage, "/t5/lane_b/tf", reliable
        )
        self.metadata_publisher = self.create_publisher(
            ObservationMetadata,
            "/internvla/observation/metadata",
            reliable,
        )
        self.odometry_publisher = self.create_publisher(
            Odometry, "/odom", reliable
        )
        self.local_costmap_publisher = self.create_publisher(
            OccupancyGrid, "/t5/lane_b/local_costmap/costmap", reliable
        )
        self.global_costmap_publisher = self.create_publisher(
            OccupancyGrid, "/t5/lane_b/global_costmap/costmap", reliable
        )

    def publish_clock(self, sim_s: float) -> None:
        clock = Clock()
        clock.clock = stamp(sim_s)
        self.clock_publisher.publish(clock)

    def publish_sensors(self, *, sim_s: float) -> None:
        sensor_s = sim_s - 0.1
        self.publish_clock(sim_s)

        transform = TransformStamped()
        transform.header.stamp = stamp(sensor_s)
        transform.header.frame_id = "map"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = 1.5
        transform.transform.translation.y = 1.5
        transform.transform.rotation.w = 1.0
        self.tf_publisher.publish(TFMessage(transforms=[transform]))

        odometry = Odometry()
        odometry.header.stamp = stamp(sensor_s)
        odometry.header.frame_id = "map"
        odometry.child_frame_id = "base_link"
        odometry.pose.pose.position.x = 1.5
        odometry.pose.pose.position.y = 1.5
        odometry.pose.pose.orientation.w = 1.0
        self.odometry_publisher.publish(odometry)

        grid = OccupancyGrid()
        grid.header.stamp = stamp(sensor_s)
        grid.header.frame_id = "map"
        grid.info.width = 7
        grid.info.height = 4
        grid.info.resolution = 1.0
        grid.info.origin.orientation.w = 1.0
        cells = {"#": 100, ".": 0, "?": -1}
        rows = ("#######", "#.....?", "#.....?", "#######")
        grid.data = [cells[cell] for row in rows for cell in row]
        self.local_costmap_publisher.publish(grid)
        self.global_costmap_publisher.publish(grid)

    def publish_metadata(self, *, sim_s: float, reset: int, sequence: int) -> None:
        sensor_s = sim_s - 0.1
        self.publish_clock(sim_s)
        metadata = ObservationMetadata()
        metadata.header.stamp = stamp(sensor_s)
        metadata.protocol_version = 1
        metadata.episode_id = "b::probe-episode"
        metadata.reset_generation = reset
        metadata.sequence_id = sequence
        metadata.request_id = f"b::probe-episode::{reset}::{sequence}"
        metadata.observation_digest = "0" * 64
        metadata.sim_stamp = stamp(sensor_s)
        metadata.deadline = stamp(sim_s + 2.0)
        metadata.valid_until = stamp(sim_s + 2.0)
        metadata.instruction = "probe only"
        metadata.global_rotation[3] = 1.0
        self.metadata_publisher.publish(metadata)

    def publish_set(self, *, sim_s: float, reset: int, sequence: int) -> None:
        self.publish_sensors(sim_s=sim_s)
        self.publish_metadata(sim_s=sim_s, reset=reset, sequence=sequence)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=12.0)
    args = parser.parse_args()

    checks: dict[str, bool] = {}
    observations: dict[str, Any] = {}
    rclpy.init()
    node = Probe()
    try:
        discovery_deadline = time.monotonic() + min(args.timeout_sec, 5.0)
        while time.monotonic() < discovery_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.metadata_publisher.get_subscription_count() > 0:
                break
        checks["capture_subscriber_discovered"] = (
            node.metadata_publisher.get_subscription_count() > 0
        )

        first_deadline = time.monotonic() + args.timeout_sec
        while time.monotonic() < first_deadline and not args.snapshot.is_file():
            node.publish_set(sim_s=10.1, reset=0, sequence=0)
            rclpy.spin_once(node, timeout_sec=0.05)
        first = json.loads(args.snapshot.read_text(encoding="utf-8"))
        observations["first_snapshot_id"] = first.get("snapshot_id")
        observations["first_frontier_count"] = len(
            first.get("candidate_frontiers") or []
        )
        checks["first_v1_snapshot"] = (
            first.get("schema_version") == 1
            and first.get("kind") == "t5_live_nav2_frontier_snapshot"
            and first.get("snapshot_id") == "b::probe-episode::0::0"
            and observations["first_frontier_count"] > 0
        )

        # The reset metadata must invalidate all pre-reset sensor and TF
        # caches.  No new snapshot may exist until post-barrier inputs arrive.
        barrier_deadline = time.monotonic() + args.timeout_sec
        while time.monotonic() < barrier_deadline:
            node.publish_metadata(sim_s=11.1, reset=1, sequence=0)
            rclpy.spin_once(node, timeout_sec=0.05)
            if not args.snapshot.exists() and args.status.is_file():
                barrier = json.loads(args.status.read_text(encoding="utf-8"))
                if barrier.get("status") == "WAITING_FOR_INPUT":
                    break
        barrier = json.loads(args.status.read_text(encoding="utf-8"))
        observations["reset_barrier_status"] = barrier.get("status")
        observations["reset_barrier_blocker"] = barrier.get("blocker_code")
        checks["reset_metadata_cleared_pre_reset_inputs"] = (
            not args.snapshot.exists()
            and barrier.get("status") == "WAITING_FOR_INPUT"
            and barrier.get("blocker_code") == "MISSING_ODOMETRY"
        )

        reset_deadline = time.monotonic() + args.timeout_sec
        while time.monotonic() < reset_deadline:
            node.publish_sensors(sim_s=11.1)
            rclpy.spin_once(node, timeout_sec=0.05)
            if args.snapshot.is_file():
                current = json.loads(args.snapshot.read_text(encoding="utf-8"))
                if current.get("snapshot_id") == "b::probe-episode::1::0":
                    break
        reset_value = json.loads(args.snapshot.read_text(encoding="utf-8"))
        observations["reset_snapshot_id"] = reset_value.get("snapshot_id")
        checks["reset_replaced_old_identity"] = (
            reset_value.get("snapshot_id") == "b::probe-episode::1::0"
        )
        checks["snapshot_ready_count_persisted"] = (
            int(
                json.loads(args.status.read_text(encoding="utf-8")).get(
                    "snapshot_ready_count", 0
                )
            )
            >= 2
        )

        stale_deadline = time.monotonic() + args.timeout_sec
        while time.monotonic() < stale_deadline:
            node.publish_set(sim_s=12.1, reset=0, sequence=1)
            rclpy.spin_once(node, timeout_sec=0.05)
            if not args.snapshot.exists() and args.status.is_file():
                state = json.loads(args.status.read_text(encoding="utf-8"))
                if state.get("blocker_code") == "STALE_CAPTURE_IDENTITY":
                    break
        state = json.loads(args.status.read_text(encoding="utf-8"))
        observations["terminal_status"] = state.get("status")
        observations["terminal_blocker"] = state.get("blocker_code")
        checks["stale_identity_cleared_snapshot"] = not args.snapshot.exists()
        checks["stale_identity_recorded"] = (
            state.get("status") == "BLOCKED"
            and state.get("blocker_code") == "STALE_CAPTURE_IDENTITY"
        )
        checks["zero_authority"] = all(
            state.get(key) == "none"
            for key in (
                "motion_authority",
                "terminal_stop_authority",
                "goal_authority",
                "model_request_authority",
            )
        )
    except (FileNotFoundError, json.JSONDecodeError, RuntimeError) as exc:
        observations["exception"] = f"{type(exc).__name__}: {exc}"[:512]
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    result = {
        "schema_version": 1,
        "kind": "t5_live_frontier_capture_ros_probe",
        "status": "PASS" if checks and all(checks.values()) else "FAIL",
        "checks": checks,
        "observations": observations,
        "publishing_scope": "synthetic_clock_tf_sensor_metadata_only",
        "control_authority": "none",
        "recorded_wall_time_s": time.time(),
    }
    _atomic_json(args.output, result)
    return 0 if result["status"] == "PASS" else 75


if __name__ == "__main__":
    raise SystemExit(main())
