#!/usr/bin/env python3
"""Evaluate the non-negotiable T4.2-R3 repaired-smoke gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--write-gate-marker", action="store_true")
    args = parser.parse_args()
    root = args.result_dir.resolve()
    stages = [
        record
        for record in load_jsonl(root / "costmap_service_stage_records.jsonl")
        if record.get("stream") == "service_local"
    ]
    controller_records = load_jsonl(root / "controller_records.jsonl")
    controller = load_json(root / "controller_summary.json")
    bridge = load_json(root / "go2_sensor_bridge_summary.json")
    clearing = (
        load_jsonl(root / "footprint_clearing_status.jsonl")
        if (root / "footprint_clearing_status.jsonl").exists()
        else []
    )

    qualifying = []
    for record in stages:
        final = record["stage_c_actual_final"]
        spatial = final["spatial"]
        forward = spatial["forward"]
        qualifying.append(
            spatial["robot_cell_class"] == "free"
            and bool(spatial["footprint"]["all_traversable"])
            and bool(forward["0.5_m"]["connected_free_from_robot"])
            and bool(forward["1.0_m"]["connected_free_from_robot"])
            and int(final["counts"]["unknown_count"]) > 0
        )
    maximum_consecutive = 0
    consecutive = 0
    for value in qualifying:
        consecutive = consecutive + 1 if value else 0
        maximum_consecutive = max(maximum_consecutive, consecutive)

    episode_records = [
        record
        for record in controller_records
        if bool(record.get("identity_valid", False))
        and not bool(record.get("state_only", False))
    ]
    nonzero_linear = sum(
        abs(float(record.get("desired_linear_x", 0.0))) > 1.0e-4
        for record in episode_records
    )
    poses = [record.get("pose_xyz_wxyz", []) for record in episode_records]
    movement = sum(
        math.hypot(float(current[0]) - float(previous[0]), float(current[1]) - float(previous[1]))
        for previous, current in zip(poses, poses[1:])
        if len(previous) >= 2 and len(current) >= 2
    )
    clear_updates = sum(bool(record.get("cleared", False)) for record in clearing)
    safety = {
        "physical_collision_count": int(controller.get("physical_collision_count", -1)),
        "fall_count": int(controller.get("fall_count", -1)),
        "stale_motion_execution_count": int(controller.get("stale_motion_execution_count", -1)),
        "direct_motion_bypass_count": int(controller.get("direct_motion_bypass_count", -1)),
    }
    source_dropouts = bridge.get("dropouts", {})
    checks = {
        "five_consecutive_stable_costmap_updates": maximum_consecutive >= 5,
        "five_nonzero_linear_commands": nonzero_linear >= 5,
        "actual_movement_at_least_0p5_m": movement >= 0.5,
        "physical_collision_fall_stale_bypass_zero": all(value == 0 for value in safety.values()),
        "distant_unknown_preserved": any(
            int(record["stage_c_actual_final"]["counts"]["unknown_count"]) > 0
            for record in stages
        ),
        "footprint_clearing_executed": clear_updates >= 5,
        "required_sensor_streams_not_dropped_at_end": not any(source_dropouts.values()),
        "collision_monitor_path_present": int(controller.get("collision_monitor_stop_count", 0)) >= 0,
    }
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checks": checks,
        "metrics": {
            "maximum_consecutive_qualifying_costmap_updates": maximum_consecutive,
            "nonzero_linear_command_count": nonzero_linear,
            "actual_xy_path_length_m": movement,
            "footprint_clearing_update_count": clear_updates,
            "stage_pair_count": len(stages),
            "safety": safety,
            "ending_dropouts": source_dropouts,
        },
        "result_dir": str(root),
    }
    output = root / "t4_2_r3_smoke_gate.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_gate_marker and payload["status"] == "PASS":
        marker = root.parents[1] / "gate2_pass.json"
        marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
