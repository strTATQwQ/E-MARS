#!/usr/bin/env python3
"""Return the daemon-free ROS 2 publisher count after a bounded discovery window."""

from __future__ import annotations

import argparse
import time

import rclpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--discovery-sec", type=float, default=1.5)
    args = parser.parse_args()
    if not args.topic.startswith("/"):
        raise SystemExit("topic must be absolute")
    if args.discovery_sec <= 0.0:
        raise SystemExit("discovery-sec must be positive")

    rclpy.init()
    node = rclpy.create_node(
        "internnav_t5_clock_graph_probe",
        namespace=f"/probe_{time.monotonic_ns()}",
        enable_rosout=False,
        start_parameter_services=False,
    )
    count: int
    try:
        deadline = time.monotonic() + args.discovery_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(
                node,
                timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())),
            )
        count = len(node.get_publishers_info_by_topic(args.topic))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(count, flush=True)


if __name__ == "__main__":
    main()
