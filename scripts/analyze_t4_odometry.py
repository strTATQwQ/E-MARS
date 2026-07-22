#!/usr/bin/env python3
"""Compute T4 ATE/RPE, drift, timing, reset, and tracking-loss evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rmse(values: list[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def angle_delta(left: float, right: float) -> float:
    return math.atan2(math.sin(left - right), math.cos(left - right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maximum-ate-rmse-m", type=float, default=0.20)
    args = parser.parse_args()
    records = read_jsonl(args.result_dir / "odometry_evaluation_records.jsonl")
    supervisor = read_jsonl(args.result_dir / "odometry_supervisor_records.jsonl")
    controller = json.loads(
        (args.result_dir / "controller_summary.json").read_text(encoding="utf-8")
    )
    by_generation: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_generation[int(record["reset_generation"])].append(record)
    ate_values: list[float] = []
    yaw_errors: list[float] = []
    tf_ages: list[float] = []
    intervals: list[float] = []
    translation_rpe: list[float] = []
    rotation_rpe: list[float] = []
    episodes: list[dict[str, object]] = []
    for generation, items in sorted(by_generation.items()):
        items.sort(key=lambda item: float(item["monotonic_sec"]))
        episode_ate = [float(item["xy_error_m"]) for item in items]
        episode_yaw = [float(item["yaw_error_rad"]) for item in items]
        episode_tf = [float(item["tf_age_sec"]) for item in items]
        ate_values.extend(episode_ate)
        yaw_errors.extend(episode_yaw)
        tf_ages.extend(episode_tf)
        for previous, current in zip(items, items[1:]):
            interval = float(current["monotonic_sec"]) - float(
                previous["monotonic_sec"]
            )
            if interval > 0.0:
                intervals.append(interval)
            previous_truth = [float(x) for x in previous["truth_xy_yaw"]]
            current_truth = [float(x) for x in current["truth_xy_yaw"]]
            previous_estimate = [float(x) for x in previous["estimated_xy_yaw"]]
            current_estimate = [float(x) for x in current["estimated_xy_yaw"]]
            truth_delta = (
                current_truth[0] - previous_truth[0],
                current_truth[1] - previous_truth[1],
            )
            estimate_delta = (
                current_estimate[0] - previous_estimate[0],
                current_estimate[1] - previous_estimate[1],
            )
            translation_rpe.append(
                math.hypot(
                    estimate_delta[0] - truth_delta[0],
                    estimate_delta[1] - truth_delta[1],
                )
            )
            rotation_rpe.append(
                abs(
                    angle_delta(
                        current_estimate[2] - previous_estimate[2],
                        current_truth[2] - previous_truth[2],
                    )
                )
            )
        episodes.append(
            {
                "reset_generation": generation,
                "episode_id": str(items[-1].get("episode_id", "")),
                "sample_count": len(items),
                "ate_rmse_m": rmse(episode_ate),
                "final_yaw_error_rad": episode_yaw[-1],
                "maximum_tf_age_sec": max(episode_tf),
            }
        )
    tracking_lost_count = sum(
        record.get("event") == "tracking_lost" for record in supervisor
    )
    unexpected_exit_count = sum(
        record.get("event") == "fatal_unexpected_exit" for record in supervisor
    )
    reset_start_generations = {
        int(record["reset_generation"])
        for record in supervisor
        if record.get("event") == "start" and int(record["reset_generation"]) >= 0
    }
    ate = rmse(ate_values)
    payload = {
        "schema_version": 1,
        "status": "PASS"
        if (
            ate is not None
            and ate <= args.maximum_ate_rmse_m
            and tracking_lost_count == 0
            and unexpected_exit_count == 0
            and len(reset_start_generations) == len(by_generation)
        )
        else "FAIL",
        "required": {
            "maximum_ate_rmse_m": args.maximum_ate_rmse_m,
            "tracking_lost_count": 0,
            "unexpected_exit_count": 0,
            "restart_per_episode": True,
        },
        "episode_count": len(by_generation),
        "sample_count": len(records),
        "ate_rmse_m": ate,
        "translation_rpe_rmse_m": rmse(translation_rpe),
        "rotation_rpe_rmse_rad": rmse(rotation_rpe),
        "maximum_absolute_yaw_drift_rad": (
            max(abs(value) for value in yaw_errors) if yaw_errors else None
        ),
        "pose_frequency_hz": (
            1.0 / statistics.median(intervals) if intervals else None
        ),
        "mean_tf_age_sec": statistics.fmean(tf_ages) if tf_ages else None,
        "maximum_tf_age_sec": max(tf_ages) if tf_ages else None,
        "stereo_pair_max_sync_error_ms": controller.get(
            "stereo_pair_max_sync_error_ms"
        ),
        "tracking_lost_count": tracking_lost_count,
        "unexpected_exit_count": unexpected_exit_count,
        "reset_restart_count": len(reset_start_generations),
        "relocalization_available": False,
        "relocalization_count": 0,
        "reset_policy": "fresh cuVSLAM process and origin alignment per episode",
        "episodes": episodes,
    }
    output = args.output or args.result_dir / "odometry_metrics.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
