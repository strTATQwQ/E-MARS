"""Scene-aware static OccupancyGrid publisher for the fixed F1 smoke scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Mapping

import yaml

from .contract import ContractError, _validate_smoke_map


def load_map_config(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ContractError("cannot load smoke map config: {}".format(exc)) from exc
    if not isinstance(value, Mapping):
        raise ContractError("smoke map config must be a mapping")
    _validate_smoke_map(value)
    return value


def build_grid(config: Mapping[str, Any]) -> List[int]:
    """Rasterize the frozen diagnostic rectangles into a Nav2 occupancy grid."""

    _validate_smoke_map(config)
    width = int(config["width"])
    height = int(config["height"])
    resolution = float(config["resolution_m"])
    origin_x, origin_y = (float(item) for item in config["origin_xy"])
    free = int(config["free_value"])
    occupied = int(config["occupied_value"])
    grid = [free] * (width * height)
    for obstacle in config["obstacles"]:
        center_x, center_y = (float(item) for item in obstacle["center_xy_m"])
        size_x, size_y = (float(item) for item in obstacle["size_xy_m"])
        half_x, half_y = size_x / 2.0, size_y / 2.0
        for row in range(height):
            y = origin_y + (row + 0.5) * resolution
            if not center_y - half_y <= y <= center_y + half_y:
                continue
            for column in range(width):
                x = origin_x + (column + 0.5) * resolution
                if center_x - half_x <= x <= center_x + half_x:
                    grid[row * width + column] = occupied
    return grid


def grid_summary(config: Mapping[str, Any]) -> Mapping[str, Any]:
    grid = build_grid(config)
    occupied = int(config["occupied_value"])
    payload = bytes(grid)
    return {
        "schema_version": 1,
        "status": "PASS",
        "name": config["name"],
        "target": config["target"],
        "width": config["width"],
        "height": config["height"],
        "resolution_m": config["resolution_m"],
        "origin_xy": list(config["origin_xy"]),
        "occupied_cells": sum(value == occupied for value in grid),
        "free_cells": sum(value != occupied for value in grid),
        "grid_sha256": hashlib.sha256(payload).hexdigest(),
        "map_to_odom": "identity",
        "qos": dict(config["qos"]),
    }


def require_simulation_boundary() -> None:
    if os.name != "posix":
        raise RuntimeError("ROS static map execution requires the POSIX simulation host")
    if os.environ.get("INTERNNAV_RUNTIME_POLICY") != "completion_sim":
        raise RuntimeError("static map publisher requires completion_sim")
    if os.environ.get("INTERNNAV_SIMULATION_TARGET") != "isaac":
        raise RuntimeError("static map publisher rejects non-Isaac targets")
    if os.environ.get("INTERNNAV_T4_MAP_COMPANION_ACK") != "1":
        raise RuntimeError("static map publisher requires the managed companion ack")


def run_ros(config_path: Path, result_dir: Path, ros_args: List[str]) -> int:
    require_simulation_boundary()
    config = load_map_config(config_path)
    grid = build_grid(config)
    summary = grid_summary(config)
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / "map_publisher.json"
    if summary_path.exists():
        raise FileExistsError("refusing to append static-map evidence")

    # ROS is deliberately imported only after the hardware/profile guards.
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from tf2_ros import StaticTransformBroadcaster

    class StaticMapNode(Node):
        def __init__(self) -> None:
            super().__init__("internnav_completion_static_map")
            qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.publisher = self.create_publisher(OccupancyGrid, "/map", qos)
            self.tf = StaticTransformBroadcaster(self)
            self.published = 0
            self.timer = self.create_timer(1.0, self.publish_map)
            self.publish_map()

        def publish_map(self) -> None:
            stamp = self.get_clock().now().to_msg()
            message = OccupancyGrid()
            message.header.stamp = stamp
            message.header.frame_id = "map"
            message.info.map_load_time = stamp
            message.info.resolution = float(config["resolution_m"])
            message.info.width = int(config["width"])
            message.info.height = int(config["height"])
            message.info.origin.position.x = float(config["origin_xy"][0])
            message.info.origin.position.y = float(config["origin_xy"][1])
            message.info.origin.orientation.w = 1.0
            message.data = grid
            self.publisher.publish(message)
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = "map"
            transform.child_frame_id = "odom"
            transform.transform.rotation.w = 1.0
            self.tf.sendTransform(transform)
            self.published += 1

    rclpy.init(args=ros_args)
    node = StaticMapNode()
    summary_payload = dict(summary)
    summary_payload["status"] = "RUNNING"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        summary_payload["status"] = "STOPPED"
        summary_payload["publish_count"] = node.published
        with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args, ros_args = parser.parse_known_args(argv)
    try:
        return run_ros(args.config, args.result_dir, ros_args)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
