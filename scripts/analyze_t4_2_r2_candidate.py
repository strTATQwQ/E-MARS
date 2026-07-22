#!/usr/bin/env python3
"""Evaluate one inflation candidate against the strengthened smoke gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def range_or_none(values: list[float | int]) -> dict[str, float | int | None]:
    return {"min": min(values) if values else None, "max": max(values) if values else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    service = [
        record
        for record in load_jsonl(
            args.result_dir / "costmap_service_stage_records.jsonl"
        )
        if record.get("event") == "actual_costmap_service_stage_pair"
    ]
    stable: dict[str, list[dict[str, Any]]] = {}
    for stream in ("service_local", "service_global"):
        records = [
            record
            for record in service
            if record.get("stream") == stream
            and (
                int(record["stage_b_actual_pre_inflation"]["counts"]["free_count"])
                + int(record["stage_b_actual_pre_inflation"]["counts"]["occupied_count"])
                > 0
            )
        ]
        stable[stream] = records[-20:]

    stream_evidence: dict[str, Any] = {}
    for stream, records in stable.items():
        finals = [record["stage_c_actual_final"] for record in records]
        stream_evidence[stream] = {
            "stable_record_count": len(records),
            "free_count": range_or_none(
                [int(record["counts"]["free_count"]) for record in finals]
            ),
            "connected_free_cell_count": range_or_none(
                [
                    int(record["spatial"]["connected_free_cell_count"])
                    for record in finals
                ]
            ),
            "footprint_all_traversable_count": sum(
                bool(record["spatial"]["footprint"]["all_traversable"])
                for record in finals
            ),
            "forward_connected_count": {
                distance: sum(
                    bool(
                        record["spatial"]["forward"][distance][
                            "connected_free_from_robot"
                        ]
                    )
                    for record in finals
                )
                for distance in ("0.5_m", "1.0_m", "1.5_m")
            },
            "maximum_consecutive_connected_free": max(
                (
                    int(record["consecutive_updates_with_connected_free"])
                    for record in finals
                ),
                default=0,
            ),
            "robot_cell_classes": sorted(
                {str(record["spatial"]["robot_cell_class"]) for record in finals}
            ),
        }

    active = load_jsonl(args.result_dir / "active_records.jsonl")
    nonzero_cmd_vel_count = sum(
        abs(float(record.get("cmd_vel_linear_x", 0.0))) > 1.0e-3 for record in active
    )
    valid_plan_count = sum(bool(record.get("nav2_plan_valid")) for record in active)
    controller_summary = load_json(args.result_dir / "controller_summary.json")
    per_controller = load_json(args.result_dir / "controller_per_episode.json")
    measured_distance_m = sum(
        float(record.get("measured_path_length_m", 0.0))
        for record in per_controller.get("episodes", [])
    )
    safety = {
        key: int(controller_summary.get(key, 0))
        for key in (
            "physical_collision_count",
            "fall_count",
            "stale_motion_execution_count",
            "direct_motion_bypass_count",
        )
    }
    result = load_json(args.result_dir / "result.json").get("val_unseen", {})
    per_episode = load_json(args.result_dir / "per_episode.json")

    checks = {
        "stable_local_and_global_free_for_5_updates": all(
            evidence["stable_record_count"] >= 5
            and evidence["maximum_consecutive_connected_free"] >= 5
            for evidence in stream_evidence.values()
        ),
        "footprint_in_connected_traversable_space": all(
            evidence["footprint_all_traversable_count"] >= 5
            and (evidence["connected_free_cell_count"]["max"] or 0) > 0
            for evidence in stream_evidence.values()
        ),
        "forward_0p5_to_1p0_path": all(
            evidence["forward_connected_count"]["0.5_m"] >= 1
            and evidence["forward_connected_count"]["1.0_m"] >= 1
            for evidence in stream_evidence.values()
        ),
        "nonzero_cmd_vel_at_least_5": nonzero_cmd_vel_count >= 5,
        "movement_at_least_0p5_m": measured_distance_m >= 0.5,
        "physical_collision_zero": safety["physical_collision_count"] == 0,
        "fall_zero": safety["fall_count"] == 0,
        "stale_motion_zero": safety["stale_motion_execution_count"] == 0,
        "direct_safety_bypass_zero": safety["direct_motion_bypass_count"] == 0,
        "collision_monitor_stop_and_recover": int(
            controller_summary.get("collision_monitor_stop_count", 0)
        )
        > 0
        and int(controller_summary.get("collision_monitor_recovery_count", 0)) > 0,
    }
    passed = all(checks.values())
    output = {
        "schema_version": 1,
        "gate": "T4.2-R2 strengthened repaired smoke",
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "costmap": stream_evidence,
        "motion": {
            "nonzero_cmd_vel_count": nonzero_cmd_vel_count,
            "valid_plan_count": valid_plan_count,
            "measured_distance_m": measured_distance_m,
            "trajectory_length_metric_m": result.get("TL"),
            "termination_reasons": [
                record.get("termination_reason")
                for record in per_episode.get("episodes", [])
            ],
        },
        "safety": {
            **safety,
            "collision_monitor_stop_count": int(
                controller_summary.get("collision_monitor_stop_count", 0)
            ),
            "collision_monitor_recovery_count": int(
                controller_summary.get("collision_monitor_recovery_count", 0)
            ),
        },
    }
    if not math.isfinite(measured_distance_m):
        raise RuntimeError("non-finite measured distance")
    output_path = args.output or args.result_dir / "repaired_smoke_analysis.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
