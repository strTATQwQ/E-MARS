#!/usr/bin/env python3
"""Aggregate continuous Go2 controller evidence by episode/reset generation."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for line in args.records.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        grouped[int(item["reset_generation"])].append(item)

    episodes = []
    all_lateral: list[float] = []
    all_heading: list[float] = []
    for generation, records in sorted(grouped.items()):
        motion_records = [item for item in records if not bool(item.get("state_only", False))]
        episode_id = str(
            (motion_records[0] if motion_records else records[0])["episode_id"]
        )
        lateral = [abs(float(item["lateral_error_m"])) for item in motion_records]
        heading = [abs(float(item["heading_error_rad"])) for item in motion_records]
        all_lateral.extend(lateral)
        all_heading.extend(heading)
        positions = [
            (float(item["pose_xyz_wxyz"][0]), float(item["pose_xyz_wxyz"][1]))
            for item in motion_records
            if len(item.get("pose_xyz_wxyz", [])) >= 2
        ]
        distance = sum(math.dist(a, b) for a, b in zip(positions, positions[1:]))
        speeds = [
            math.hypot(*[float(value) for value in item["actual_linear_velocity_base"][:2]])
            for item in motion_records
        ]
        emergency = [bool(item["emergency_stop"]) for item in motion_records]
        safety_stop = [
            bool(item["emergency_stop"]) or bool(item.get("collision_monitor_stopping", False))
            for item in motion_records
        ]
        recoveries = sum(first and not second for first, second in zip(safety_stop, safety_stop[1:]))
        scenario_names = sorted(
            {
                str(item.get("obstacle_scenario", ""))
                for item in motion_records
                if item.get("obstacle_scenario")
            }
        )
        episodes.append(
            {
                "episode_id": episode_id,
                "reset_generation": generation,
                "sample_count": len(motion_records),
                "state_only_sample_count": len(records) - len(motion_records),
                "measured_path_length_m": distance,
                "mean_speed_mps": sum(speeds) / len(speeds) if speeds else 0.0,
                "maximum_speed_mps": max(speeds, default=0.0),
                "mean_lateral_error_m": sum(lateral) / len(lateral) if lateral else 0.0,
                "p95_lateral_error_m": percentile(lateral, 0.95),
                "mean_heading_error_rad": sum(heading) / len(heading) if heading else 0.0,
                "p95_heading_error_rad": percentile(heading, 0.95),
                "emergency_stop_samples": sum(emergency),
                "safety_stop_samples": sum(safety_stop),
                "recovery_count": recoveries,
                "obstacle_scenarios": scenario_names,
                "obstacle_expected_target_count": sum(
                    int(item.get("obstacle_expected_target_count", 0))
                    for item in motion_records
                ),
                "obstacle_detected_target_count": sum(
                    int(item.get("obstacle_detected_target_count", 0))
                    for item in motion_records
                ),
                "obstacle_expected_frame_count": sum(
                    int(item.get("obstacle_expected_frame_count", 0))
                    for item in motion_records
                ),
                "obstacle_detected_frame_count": sum(
                    int(item.get("obstacle_detected_frame_count", 0))
                    for item in motion_records
                ),
                "physical_collision_samples": sum(
                    bool(item.get("physical_collision", False))
                    for item in motion_records
                ),
                "fallen_samples": sum(bool(item["fallen"]) for item in motion_records),
            }
        )

    payload = {
        "schema_version": 1,
        "episode_count": len(episodes),
        "sample_count": sum(item["sample_count"] for item in episodes),
        "state_only_sample_count": sum(
            item["state_only_sample_count"] for item in episodes
        ),
        "aggregate": {
            "mean_lateral_error_m": sum(all_lateral) / len(all_lateral) if all_lateral else 0.0,
            "p95_lateral_error_m": percentile(all_lateral, 0.95),
            "mean_heading_error_rad": sum(all_heading) / len(all_heading) if all_heading else 0.0,
            "p95_heading_error_rad": percentile(all_heading, 0.95),
        },
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
