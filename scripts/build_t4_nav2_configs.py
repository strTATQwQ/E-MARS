#!/usr/bin/env python3
"""Mechanically derive T4.2 Nav2 parameters from the frozen T3 profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LOCAL_OBSTACLE = """      plugins: [obstacle_layer, inflation_layer]
      obstacle_layer:
        plugin: nav2_costmap_2d::ObstacleLayer
        enabled: true
        observation_sources: go2_depth
        go2_depth:
          topic: /go2/depth/points
          data_type: PointCloud2
          clearing: true
          marking: true
          min_obstacle_height: 0.05
          max_obstacle_height: 1.5
          obstacle_min_range: 0.10
          obstacle_max_range: 5.0
          raytrace_min_range: 0.10
          raytrace_max_range: 5.5
          expected_update_rate: 0.50
          observation_persistence: 0.40
"""
LOCAL_NVBLOX = """      plugins: [nvblox_layer, inflation_layer]
      nvblox_layer:
        plugin: nvblox::nav2::NvbloxCostmapLayer
        enabled: true
        nav2_costmap_global_frame: odom
        nvblox_map_slice_topic: /nvblox_node/static_map_slice
        convert_to_binary_costmap: true
"""
GLOBAL_OBSTACLE = """      rolling_window: false
      width: 16
      height: 16
      resolution: 0.05
      robot_radius: 0.30
      track_unknown_space: false
      plugins: [static_layer, obstacle_layer, inflation_layer]
      static_layer:
        plugin: nav2_costmap_2d::StaticLayer
        map_subscribe_transient_local: true
        subscribe_to_updates: false
      obstacle_layer:
        plugin: nav2_costmap_2d::ObstacleLayer
        enabled: true
        observation_sources: go2_depth
        go2_depth:
          topic: /go2/depth/points
          data_type: PointCloud2
          clearing: true
          marking: true
          min_obstacle_height: 0.05
          max_obstacle_height: 1.5
          obstacle_min_range: 0.10
          obstacle_max_range: 5.0
          raytrace_min_range: 0.10
          raytrace_max_range: 5.5
          expected_update_rate: 0.50
          observation_persistence: 0.40
"""
GLOBAL_NVBLOX = """      rolling_window: true
      width: 20
      height: 20
      resolution: 0.05
      robot_radius: 0.30
      track_unknown_space: true
      plugins: [nvblox_layer, inflation_layer]
      nvblox_layer:
        plugin: nvblox::nav2::NvbloxCostmapLayer
        enabled: true
        nav2_costmap_global_frame: map
        nvblox_map_slice_topic: /nvblox_node/static_map_slice
        convert_to_binary_costmap: true
"""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"source profile token count is {text.count(old)}, expected 1")
    return text.replace(old, new)


def jazzy_plugin_names(text: str) -> str:
    """Apply only the configuration migrations required by Nav2 Jazzy."""
    replacements = {
        "nav2_navfn_planner/NavfnPlanner": "nav2_navfn_planner::NavfnPlanner",
        "nav2_behaviors/Spin": "nav2_behaviors::Spin",
        "nav2_behaviors/BackUp": "nav2_behaviors::BackUp",
        "nav2_behaviors/DriveOnHeading": "nav2_behaviors::DriveOnHeading",
        "nav2_behaviors/Wait": "nav2_behaviors::Wait",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise RuntimeError(f"expected one frozen plugin token: {old}")
        text = text.replace(old, new)
    old_points = "      points: [0.38, 0.26, 0.38, -0.26, -0.26, -0.26, -0.26, 0.26]"
    new_points = '      points: "[[0.38, 0.26], [0.38, -0.26], [-0.26, -0.26], [-0.26, 0.26]]"'
    if text.count(old_points) != 1:
        raise RuntimeError("expected one frozen Collision Monitor polygon")
    text = text.replace(old_points, new_points)
    if text.count("      max_points: 3") != 2:
        raise RuntimeError("expected two frozen Collision Monitor point thresholds")
    text = text.replace("      max_points: 3", "      min_points: 3")
    if text.count("    source_timeout: 0.35") != 1:
        raise RuntimeError("expected one frozen Collision Monitor source timeout")
    text = text.replace("    source_timeout: 0.35", "    source_timeout: 1.50")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--static-output", type=Path)
    parser.add_argument("--sensor-stack-output", type=Path)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    static_text = jazzy_plugin_names(raw.decode("utf-8"))
    text = static_text
    text = replace_once(text, LOCAL_OBSTACLE, LOCAL_NVBLOX)
    text = replace_once(text, GLOBAL_OBSTACLE, GLOBAL_NVBLOX)
    local_footprint_token = (
        "      robot_radius: 0.30\n"
        "      track_unknown_space: false"
    )
    global_footprint_token = (
        "      robot_radius: 0.30\n"
        "      track_unknown_space: true"
    )
    if text.count(local_footprint_token) != 1:
        raise RuntimeError("expected one local online Nvblox Go2 footprint token")
    if text.count(global_footprint_token) != 1:
        raise RuntimeError("expected one global online Nvblox Go2 footprint token")
    padded_footprint = (
        "      robot_radius: 0.30\n"
        "      footprint_padding: 0.02\n"
        "      track_unknown_space: true"
    )
    text = text.replace(local_footprint_token, padded_footprint)
    text = text.replace(global_footprint_token, padded_footprint)
    # Online Nvblox starts as a partial map.  The global planner must be able
    # to route through the not-yet-observed portion of that rolling map so the
    # robot can move and extend the observed region.  Local Nvblox geometry,
    # inflation, and Collision Monitor remain the fail-safe collision layers.
    if text.count("      allow_unknown: true") != 1:
        raise RuntimeError("expected one online-map Navfn allow_unknown token")
    output = text.encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    if args.static_output is not None:
        args.static_output.parent.mkdir(parents=True, exist_ok=True)
        args.static_output.write_text(static_text, encoding="utf-8", newline="\n")
    if args.sensor_stack_output is not None:
        args.sensor_stack_output.parent.mkdir(parents=True, exist_ok=True)
        args.sensor_stack_output.write_bytes(output)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "source": args.source.as_posix(),
        "source_sha256": digest(raw),
        "output": args.output.as_posix(),
        "output_sha256": digest(output),
        "odometry_static_map_sha256": (
            digest(static_text.encode("utf-8")) if args.static_output is not None else None
        ),
        "sensor_stack_sha256": (
            digest(output) if args.sensor_stack_output is not None else None
        ),
        "static_layer_count": text.count("plugin: nav2_costmap_2d::StaticLayer"),
        "nvblox_layer_count": text.count("plugin: nvblox::nav2::NvbloxCostmapLayer"),
        "direct_pointcloud_costmap_layer_count": text.count(
            "plugin: nav2_costmap_2d::ObstacleLayer"
        ),
        "collision_monitor_pointcloud_retained": "topic: /go2/depth/points" in text,
        "unknown_space_policy": (
            "nvblox_preserves_unknown_global_planner_may_traverse_"
            "unknown_local_safety_retained"
        ),
        "collision_monitor_source_timeout_sec": 1.5,
        "nav2_plugin_api": "jazzy_double_colon",
        "collision_monitor_api": "jazzy_string_polygon_and_min_points",
    }
    if payload["static_layer_count"] != 0 or payload["nvblox_layer_count"] != 2:
        raise RuntimeError(payload)
    args.manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
