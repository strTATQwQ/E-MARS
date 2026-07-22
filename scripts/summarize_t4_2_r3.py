#!/usr/bin/env python3
"""Summarize frozen T4.2-R3 A/B/C/D smoke evidence without inventing metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ATTEMPTS = {
    "A_d435i_only": "t4_2_r3_d435i_smoke_oracle_attempt_004",
    "B_lidar_only": "t4_2_r3_lidar_smoke_oracle_attempt_015",
    "C_fused": "t4_2_r3_fused_smoke_oracle_attempt_001",
    "D_fused_clearing_default": "t4_2_r3_fused_clearing_smoke_oracle_attempt_002",
    "D_fused_clearing_diag_timeout": (
        "t4_2_r3_fused_clearing_diag_timeout_smoke_oracle_attempt_002"
    ),
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def path_metrics(records: list[dict[str, Any]]) -> tuple[float, int]:
    path_length = 0.0
    previous: tuple[float, float] | None = None
    nonzero = 0
    for record in records:
        pose = record.get("pose_xyz_wxyz")
        if isinstance(pose, list) and len(pose) >= 2:
            current = (float(pose[0]), float(pose[1]))
            if previous is not None:
                path_length += math.dist(previous, current)
            previous = current
        if abs(float(record.get("desired_linear_x", 0.0))) > 1.0e-6:
            nonzero += 1
    return path_length, nonzero


def summarize(directory: Path) -> dict[str, Any]:
    controller = load(directory / "controller_summary.json")
    bridge = load(directory / "go2_sensor_bridge_summary.json")
    costmap = load(directory / "costmap_stage_summary.json")
    raw = jsonl(directory / "nvblox_slice_classes.jsonl")
    stages = jsonl(directory / "costmap_service_stage_records.jsonl")
    clearing = jsonl(directory / "footprint_clearing_status.jsonl")
    records = jsonl(directory / "controller_records.jsonl")
    path_length, nonzero = path_metrics(records)
    raw_best = max(
        raw,
        key=lambda item: int(item.get("free_positive_count", 0))
        + int(item.get("occupied_nonpositive_count", 0)),
        default={},
    )
    stage_summary: dict[str, Any] = {}
    for stage_name in ("stage_b_actual_pre_inflation", "stage_c_actual_final"):
        candidates = [item.get(stage_name, {}) for item in stages]
        candidates = [item for item in candidates if item]
        best = max(
            candidates,
            key=lambda item: int(item.get("counts", {}).get("free_count", 0)),
            default={},
        )
        stage_summary[stage_name] = {
            "counts": best.get("counts"),
            "spatial": best.get("spatial"),
        }
    return {
        "attempt": directory.name,
        "phase_status": load(directory / "phase_status.json"),
        "raw_slice_best": {
            key: raw_best.get(key)
            for key in (
                "width",
                "height",
                "free_positive_count",
                "occupied_nonpositive_count",
                "unknown_count",
                "known_min_distance_m",
                "known_max_distance_m",
            )
        },
        "costmap": {
            "maximum_consecutive_final_free_updates": costmap.get(
                "maximum_consecutive_final_free_updates"
            ),
            **stage_summary,
        },
        "sensors": bridge,
        "controller": {
            key: controller.get(key)
            for key in (
                "metric_depth_frame_count",
                "r3_lidar_frame_count",
                "r3_front_rgb_frame_count",
                "r3_imu_frame_count",
                "collision_monitor_stop_count",
                "collision_monitor_recovery_count",
                "physical_collision_count",
                "fall_count",
                "stale_motion_execution_count",
                "direct_motion_bypass_count",
            )
        },
        "motion": {
            "actual_xy_path_length_m": path_length,
            "nonzero_desired_linear_records": nonzero,
        },
        "clearing": {
            "verified_update_count": sum(bool(item.get("cleared")) for item in clearing),
            "reason_counts": dict(Counter(str(item.get("reason")) for item in clearing)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "status": "BLOCKED",
        "attempts": {
            name: summarize(args.root / relative) for name, relative in ATTEMPTS.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
