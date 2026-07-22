#!/usr/bin/env python3
"""Validate one immutable T4 Oracle or InternVLA run without T3 map assumptions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--minimum-sr", type=float, required=True)
    parser.add_argument("--map-source", choices=("nvblox_online", "static_map"), required=True)
    parser.add_argument("--pose-source", choices=("ground_truth", "external_odometry"), required=True)
    parser.add_argument(
        "--runtime-policy",
        choices=("strict_evidence", "completion_sim"),
        default="strict_evidence",
        help=(
            "strict_evidence keeps physical collisions fatal; completion_sim "
            "retains them as metric-only quality warnings"
        ),
    )
    args = parser.parse_args()
    root = args.result_dir
    result = load(root / "result.json")
    metrics = result.get("val_unseen", result)
    active = load(root / "active_summary.json")
    controller = load(root / "controller_summary.json")
    per_episode = load(root / "per_episode.json")
    client = load(root / "client_summary.json") if (root / "client_summary.json").is_file() else None
    client_pose_records = jsonl(root / "client_pose_source_records.jsonl")
    client_pose_ok = (
        client is None
        or (
            len(client_pose_records) == int(client.get("step_count", -1))
            and len(client_pose_records) > 0
            and all(
                item.get("source") == "navigation_odometry"
                and item.get("evaluator_ground_truth_pose_discarded") is True
                for item in client_pose_records
            )
        )
    )
    sr = float(metrics.get("SR", 0.0))
    count = int(metrics.get("Count", metrics.get("length", 0)))
    hz = float(controller.get("measured_control_hz", 0.0))
    legacy_stale = max(
        int(controller.get("stale_count", 0)),
        int(controller.get("stale_or_identity_reject_count", 0)),
    )
    safety = {
        "nan_count": int(controller.get("nan_count", 0)),
        "fall_count": int(controller.get("fall_count", 0)),
        "physical_collision_count": int(controller.get("physical_collision_count", 0)),
        "stale_motion_execution_count": int(
            controller.get("stale_motion_execution_count", legacy_stale)
        ),
        "identity_safe_stop_count": int(
            controller.get("identity_safe_stop_count", 0)
        ),
    }
    nonfatal_safety = {"identity_safe_stop_count"}
    if args.runtime_policy == "completion_sim":
        nonfatal_safety.add("physical_collision_count")
    fatal_safety = {
        name: value for name, value in safety.items() if name not in nonfatal_safety
    }
    quality_warnings = []
    if (
        args.runtime_policy == "completion_sim"
        and safety["physical_collision_count"] > 0
    ):
        quality_warnings.append(
            {
                "code": "PHYSICAL_COLLISION_METRIC_ONLY",
                "count": safety["physical_collision_count"],
            }
        )
    mapping_ok = True
    mapping_evidence: dict[str, object] | None = None
    sensor_ok = True
    sensor_evidence: dict[str, object] | None = None
    if args.map_source == "nvblox_online":
        resets = jsonl(root / "nvblox_resets.jsonl")
        reset_generations = {
            int(item["generation"])
            for item in resets
            if item.get("event") == "reset_complete"
        }
        ready_generations = {
            int(item["generation"])
            for item in resets
            if item.get("event") == "first_map_slice"
        }
        pids = [
            int(item["pid"])
            for item in resets
            if item.get("event") == "start" and int(item.get("generation", -1)) >= 0
        ]
        map_pollution_count = sum(
            int(item.get("map_pollution_count", 0))
            for item in resets
            if item.get("event") == "first_map_slice"
        )
        class_generations = controller.get("costmap_class_generations", {})
        expected_generations = set(range(args.expected))
        class_coverage = {
            name: {int(value) for value in class_generations.get(name, [])}
            for name in ("free", "occupied", "unknown")
        }
        slice_records = jsonl(root / "nvblox_slice_classes.jsonl")
        slice_class_coverage = {
            "free": {
                int(item["generation"])
                for item in slice_records
                if int(item.get("free_positive_count", 0)) > 0
            },
            "occupied": {
                int(item["generation"])
                for item in slice_records
                if int(item.get("occupied_nonpositive_count", 0)) > 0
            },
            "unknown": {
                int(item["generation"])
                for item in slice_records
                if int(item.get("unknown_count", 0)) > 0
            },
        }
        mapping_ok = (
            len(reset_generations) == args.expected
            and ready_generations == reset_generations
            and len(pids) == args.expected
            and len(set(pids)) == args.expected
            and not any(item.get("event") == "fatal_unexpected_exit" for item in resets)
            and map_pollution_count == 0
            and all(values == expected_generations for values in class_coverage.values())
            and all(
                values == expected_generations
                for values in slice_class_coverage.values()
            )
            and int(controller.get("static_map_publish_count", -1)) == 0
            and int(controller.get("static_map_selection_count", -1)) == 0
            and int(controller.get("map_ready_generation", -1)) == args.expected - 1
        )
        mapping_evidence = {
            "reset_generations": sorted(reset_generations),
            "ready_generations": sorted(ready_generations),
            "unique_episode_process_count": len(set(pids)),
            "unexpected_exit_count": sum(
                item.get("event") == "fatal_unexpected_exit" for item in resets
            ),
            "pre_epoch_slice_reject_count": sum(
                item.get("event") == "pre_epoch_slice_rejected" for item in resets
            ),
            "map_pollution_count": map_pollution_count,
            "costmap_class_generation_coverage": {
                name: sorted(values) for name, values in class_coverage.items()
            },
            "raw_nvblox_slice_class_generation_coverage": {
                name: sorted(values) for name, values in slice_class_coverage.items()
            },
        }
        depth_frames = int(controller.get("metric_depth_frame_count", 0))
        sync_error_ms = controller.get("metric_depth_pose_sync_max_error_ms")
        transport_latency_ms = controller.get("metric_depth_transport_latency_max_ms")
        obstacle_detection = float(controller.get("obstacle_target_detection_rate", 0.0))
        costmap_detection = float(controller.get("costmap_detection_rate", 0.0))
        sensor_ok = (
            depth_frames > 0
            and controller.get("depth_camera_model")
            == "Intel RealSense D435i depth imager"
            and float(controller.get("depth_hfov_deg", 0.0)) == 87.0
            and float(controller.get("depth_vfov_deg", 0.0)) == 58.0
            and sync_error_ms is not None
            and float(sync_error_ms) <= 1.0
            and transport_latency_ms is not None
            and math.isfinite(float(transport_latency_ms))
            and float(transport_latency_ms) <= 1000.0
            and obstacle_detection >= 0.95
            and costmap_detection >= 0.95
        )
        sensor_evidence = {
            "metric_depth_frame_count": depth_frames,
            "depth_camera_model": controller.get("depth_camera_model"),
            "depth_hfov_deg": controller.get("depth_hfov_deg"),
            "depth_vfov_deg": controller.get("depth_vfov_deg"),
            "pose_depth_sync_max_error_ms": sync_error_ms,
            "transport_latency_max_ms": transport_latency_ms,
            "obstacle_target_detection_rate": obstacle_detection,
            "costmap_detection_rate": costmap_detection,
        }
    else:
        mapping_ok = (
            int(controller.get("static_map_publish_count", -1)) == args.expected
            and int(controller.get("static_map_selection_count", -1)) == args.expected
            and all(
                item.get("selection") == "dataset_episode_id_no_runtime_pose"
                for item in controller.get("static_map_selections", [])
            )
        )
    odometry = None
    odometry_ok = True
    if args.pose_source == "external_odometry":
        odometry = load(root / "odometry_metrics.json")
        odometry_ok = (
            odometry.get("status") == "PASS"
            and int(controller.get("external_odometry_stale_stop_count", -1)) == 0
            and not bool(controller.get("ground_truth_pose_used_for_nav", True))
        )
    else:
        odometry_ok = bool(controller.get("ground_truth_pose_used_for_nav", False))
    passing = (
        count == args.expected
        and sr >= args.minimum_sr
        and active.get("status") == "FINISHED"
        and int(active.get("failure_count", 0)) == 0
        and controller.get("status") == "FINISHED"
        and per_episode.get("completed_episode_count") == args.expected
        and (client is None or client.get("status") == "FINISHED")
        and client_pose_ok
        and str(controller.get("map_source", "")) == args.map_source
        and str(controller.get("pose_source", "")) == args.pose_source
        and all(value == 0 for value in fatal_safety.values())
        and int(controller.get("cmd_vel_quantization_count", 0)) == 0
        and int(controller.get("direct_motion_bypass_count", 0)) == 0
        and math.isfinite(hz)
        and 20.0 <= hz <= 50.0
        and mapping_ok
        and sensor_ok
        and odometry_ok
    )
    payload = {
        "schema_version": 2,
        "status": "PASS" if passing else "FAIL",
        "quality_status": "WARN" if quality_warnings else "PASS",
        "quality_warnings": quality_warnings,
        "runtime_policy": args.runtime_policy,
        "required": {
            "episode_count": args.expected,
            "minimum_sr": args.minimum_sr,
            "map_source": args.map_source,
            "pose_source": args.pose_source,
            "fatal_safety_zero": sorted(fatal_safety),
            "physical_collision_policy": (
                "metric_only_warn"
                if args.runtime_policy == "completion_sim"
                else "must_be_zero"
            ),
            "control_hz_range": [20.0, 50.0],
        },
        "metrics": metrics,
        "safety": safety,
        "active": active,
        "controller": controller,
        "client": client,
        "client_pose_ok": client_pose_ok,
        "mapping_ok": mapping_ok,
        "mapping_evidence": mapping_evidence,
        "sensor_ok": sensor_ok,
        "sensor_evidence": sensor_evidence,
        "odometry_ok": odometry_ok,
        "odometry": odometry,
    }
    (root / "validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if passing else 1)


if __name__ == "__main__":
    main()
