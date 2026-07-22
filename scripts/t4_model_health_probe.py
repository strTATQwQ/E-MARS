#!/usr/bin/env python3
"""Wait for one preloaded, episode-free InternVLA model on the DGX graph."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import rclpy
from internvla_ros2.model_health import validate_uninitialized_model_health
from internvla_ros2_msgs.action import Step
from internvla_ros2_msgs.srv import Health
from rclpy.action import ActionClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be fresh")
    if not 0.1 <= args.timeout_sec <= 30.0:
        parser.error("timeout must be in [0.1, 30] seconds")

    rclpy.init()
    node = rclpy.create_node(f"internvla_t4_model_health_probe_{os.getpid()}")
    health = node.create_client(Health, "/internvla/health")
    step = ActionClient(node, Step, "/internvla/step")
    started = time.monotonic()
    try:
        deadline = started + args.timeout_sec
        while time.monotonic() < deadline:
            if health.wait_for_service(timeout_sec=0.1) and step.wait_for_server(
                timeout_sec=0.1
            ):
                break
        else:
            raise TimeoutError("typed model health/action graph was not ready")
        request = Health.Request()
        request.protocol_version = 1
        future = health.call_async(request)
        rclpy.spin_until_future_complete(
            node, future, timeout_sec=max(0.1, deadline - time.monotonic())
        )
        response = future.result()
        if response is None:
            raise TimeoutError("typed model health response timed out")
        snapshot = validate_uninitialized_model_health(
            {
                "status_code": response.status_code,
                "status_message": response.status_message,
                "initialized": response.initialized,
                "lifecycle_state": response.lifecycle_state,
                "episode_id": response.episode_id,
                "reset_generation": response.reset_generation,
                "last_sequence_id": response.last_sequence_id,
                "model_revision": response.model_revision,
                "checkpoint_revision": response.checkpoint_revision,
            }
        )
        snapshot.update(
            {
                "model_host": "dgx_spark",
                "step_action_ready": True,
                "probe_duration_sec": time.monotonic() - started,
                "recorded_unix": time.time(),
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
