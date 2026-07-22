#!/usr/bin/env python3
"""ROS-only cadence, reset, stale, timeout, fall, and NaN gate for T3.1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from internvla_go2_controller.runtime import ControllerIPCClient
from internvla_ros2_msgs.msg import NavigationCommand
from rclpy.node import Node
from std_msgs.msg import Bool


def publish_identity(node: Node, publisher: object, episode: str, generation: int, sequence: int) -> None:
    message = NavigationCommand()
    message.header.stamp = node.get_clock().now().to_msg()
    message.header.frame_id = "base_link"
    message.protocol_version = 1
    message.episode_id = episode
    message.reset_generation = generation
    message.sequence_id = sequence
    message.request_id = f"fault:{generation}:{sequence}"
    message.discrete_action = 1
    message.action_source = 2
    publisher.publish(message)


def request(episode: str, generation: int, sequence: int, **updates: object) -> dict:
    value = {
        "schema_version": 1,
        "operation": "update",
        "episode_id": episode,
        "reset_generation": generation,
        "sequence_id": sequence,
        "pose_wxyz": [0.0, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0],
        "linear_velocity": [0.0, 0.0, 0.0],
        "angular_velocity": [0.0, 0.0, 0.0],
        "fallen": False,
        "nan_detected": False,
    }
    value.update(updates)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Node("go2_controller_fault_runner")
    navigation = node.create_publisher(NavigationCommand, "/internvla/navigation_command", 10)
    motion = node.create_publisher(Bool, "/internvla/nav2_motion_enabled", 10)
    stop = node.create_publisher(Bool, "/internvla/stop", 10)
    safe = node.create_publisher(Twist, "/cmd_vel_safe", 20)
    raw = node.create_publisher(Twist, "/cmd_vel_nav", 20)
    ipc = ControllerIPCClient(args.socket, timeout_sec=1.0)
    try:
        time.sleep(1.0)
        publish_identity(node, navigation, "fault-episode-0", 0, 0)
        time.sleep(0.08)
        motion.publish(Bool(data=True))
        stop.publish(Bool(data=False))
        moving = Twist()
        moving.linear.x = 0.2
        raw.publish(moving)
        safe.publish(moving)
        time.sleep(0.15)
        nominal = ipc.exchange(request("fault-episode-0", 0, 0))

        stale = ipc.exchange(request("fault-episode-stale", 0, 0))

        publish_identity(node, navigation, "fault-episode-1", 1, 0)
        time.sleep(0.08)
        reset_barrier = ipc.exchange(request("fault-episode-1", 1, 0))
        motion.publish(Bool(data=True))
        stop.publish(Bool(data=False))
        raw.publish(moving)
        safe.publish(moving)
        time.sleep(0.10)
        after_reset = ipc.exchange(request("fault-episode-1", 1, 0))

        time.sleep(0.36)
        timed_out = ipc.exchange(request("fault-episode-1", 1, 0))

        nan_stop = ipc.exchange(
            request("fault-episode-1", 1, 0, nan_detected=True)
        )
        fall_stop = ipc.exchange(request("fault-episode-1", 1, 0, fallen=True))
        collision_stop = ipc.exchange(
            request(
                "fault-episode-1",
                1,
                0,
                physical_collision=True,
                maximum_collision_force=25.0,
            )
        )

        depth = [2.0] * (48 * 64)
        cadence_started = time.monotonic()
        for index in range(80):
            raw.publish(moving)
            safe.publish(moving)
            cadence_request = request("fault-episode-1", 1, 0)
            if index % 4 == 0:
                cadence_request.update(
                    {
                        "depth_height": 48,
                        "depth_width": 64,
                        "depth_row_stride": 10,
                        "depth_column_stride": 10,
                        "depth_values": depth,
                    }
                )
            cadence = ipc.exchange(cadence_request)
            remaining = cadence_started + (index + 1) * 0.025 - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)

        passing = (
            nominal["linear_x"] > 0.0
            and stale["emergency_stop"]
            and stale["linear_x"] == 0.0
            and reset_barrier["linear_x"] == 0.0
            and after_reset["linear_x"] > 0.0
            and timed_out["command_timeout"]
            and timed_out["linear_x"] == 0.0
            and nan_stop["emergency_stop"]
            and fall_stop["emergency_stop"]
            and collision_stop["emergency_stop"]
            and cadence["linear_x"] > 0.0
        )
        payload = {
            "schema_version": 1,
            "status": "PASS" if passing else "FAIL",
            "nominal_nonzero": nominal["linear_x"] > 0.0,
            "stale_safe_stop": stale["emergency_stop"] and stale["linear_x"] == 0.0,
            "reset_barrier_safe_stop": reset_barrier["linear_x"] == 0.0,
            "post_reset_reenabled": after_reset["linear_x"] > 0.0,
            "timeout_safe_stop": timed_out["command_timeout"] and timed_out["linear_x"] == 0.0,
            "nan_safe_stop": nan_stop["emergency_stop"],
            "fall_safe_stop": fall_stop["emergency_stop"],
            "collision_safe_stop": collision_stop["emergency_stop"],
            "cadence_samples": 80,
            "depth_samples_per_frame": len(depth),
            "depth_frame_count": 20,
        }
        (args.result_dir / "fault_validation.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(0 if passing else 1)
    finally:
        ipc.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
