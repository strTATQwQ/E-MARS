#!/usr/bin/env python3
"""Fail-closed acceptance summary for one T5 cuVSLAM GT-authority shadow run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def _rows(path: Path) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("{}:{} must contain an object".format(path, number))
        result.append(value)
    return result


def _rmse(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def _percentile95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _angle_delta(left: float, right: float) -> float:
    return math.atan2(math.sin(left - right), math.cos(left - right))


def analyze(result_root: Path) -> Dict[str, Any]:
    root = result_root.resolve()
    dgx = root / "remote" / "dgx"
    supervisor = _rows(dgx / "cuvslam" / "odometry_supervisor_records.jsonl")
    samples = _rows(dgx / "cuvslam" / "cuvslam_shadow_samples.jsonl")
    controller = _object(dgx / "onboard" / "controller_summary.json")
    lane_status = _object(dgx / "lane_status.json")
    isaac_contract = _object(root / "remote" / "x86" / "isaac_contract.json")

    starts_by_generation: Dict[int, Set[int]] = defaultdict(set)
    for row in supervisor:
        if row.get("event") != "start":
            continue
        generation = int(row.get("reset_generation", -1))
        start_index = int(row.get("start_index", -1))
        if generation >= 0 and start_index > 0:
            starts_by_generation[generation].add(start_index)

    samples_by_generation: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    reset_pollution_count = 0
    for row in samples:
        generation = int(row.get("reset_generation", -1))
        start_index = int(row.get("source_start_index", -1))
        if generation < 0 or start_index not in starts_by_generation.get(
            generation, set()
        ):
            reset_pollution_count += 1
        samples_by_generation[generation].append(row)

    sim_duration_sec = 0.0
    translation_rpe: List[float] = []
    rotation_rpe: List[float] = []
    episode_metrics: List[Dict[str, Any]] = []
    for generation, generation_rows in sorted(samples_by_generation.items()):
        generation_rows.sort(key=lambda item: int(item["estimate_stamp_ns"]))
        stamps = [int(row["estimate_stamp_ns"]) for row in generation_rows]
        if len(stamps) >= 2:
            sim_duration_sec += (max(stamps) - min(stamps)) / 1e9
        for previous, current in zip(generation_rows, generation_rows[1:]):
            previous_truth = [float(item) for item in previous["truth_xy_yaw"]]
            current_truth = [float(item) for item in current["truth_xy_yaw"]]
            previous_estimate = [
                float(item) for item in previous["estimated_xy_yaw"]
            ]
            current_estimate = [float(item) for item in current["estimated_xy_yaw"]]
            translation_rpe.append(
                math.hypot(
                    (current_estimate[0] - previous_estimate[0])
                    - (current_truth[0] - previous_truth[0]),
                    (current_estimate[1] - previous_estimate[1])
                    - (current_truth[1] - previous_truth[1]),
                )
            )
            rotation_rpe.append(
                abs(
                    _angle_delta(
                        current_estimate[2] - previous_estimate[2],
                        current_truth[2] - previous_truth[2],
                    )
                )
            )
        episode_metrics.append(
            {
                "reset_generation": generation,
                "sample_count": len(generation_rows),
                "sim_duration_sec": (
                    (max(stamps) - min(stamps)) / 1e9 if len(stamps) >= 2 else 0.0
                ),
                "ate_rmse_m": _rmse(
                    [float(row["xy_error_m"]) for row in generation_rows]
                ),
                "maximum_yaw_error_rad": max(
                    (float(row["yaw_error_rad"]) for row in generation_rows),
                    default=None,
                ),
            }
        )

    xy_errors = [float(row["xy_error_m"]) for row in samples]
    yaw_errors = [float(row["yaw_error_rad"]) for row in samples]
    tf_ages = [float(row["tf_age_sec"]) for row in samples]
    ate_rmse_m = _rmse(xy_errors)
    maximum_yaw_error_rad = max(yaw_errors) if yaw_errors else None
    tf_age_p95_sec = _percentile95(tf_ages)
    maximum_tf_age_sec = max(tf_ages) if tf_ages else None
    stereo_skew = controller.get("stereo_pair_max_sync_error_ms")
    reset_count = len(starts_by_generation)
    tracking_lost_count = sum(
        row.get("event") == "tracking_lost" for row in supervisor
    )
    unexpected_exit_count = sum(
        row.get("event") == "fatal_unexpected_exit" for row in supervisor
    )
    stop_authority_count = sum(
        row.get("fail_safe_stop_published") is True for row in supervisor
    )
    checks = {
        "exact_lane_a_strict_profile": lane_status.get("lane") == "a"
        and isaac_contract.get("strict_extension_profile") == "cuvslam_shadow"
        and isaac_contract.get("stereo_odometry_enabled") is True,
        "gt_remains_navigation_authority": controller.get("pose_source")
        == "ground_truth"
        and controller.get("ground_truth_pose_used_for_nav") is True
        and int(controller.get("external_odometry_nav_publish_count", -1)) == 0,
        "real_cuvslam_odometry_nonempty": bool(samples),
        "minimum_two_resets": reset_count >= 2,
        "minimum_120_sim_seconds": sim_duration_sec >= 120.0,
        "stereo_pair_skew": isinstance(stereo_skew, (int, float))
        and float(stereo_skew) <= 0.1,
        "ate_rmse": ate_rmse_m is not None and ate_rmse_m <= 0.20,
        "maximum_yaw_error": maximum_yaw_error_rad is not None
        and maximum_yaw_error_rad <= 0.15,
        "tf_age_p95": tf_age_p95_sec is not None and tf_age_p95_sec <= 0.5,
        "tf_age_maximum": maximum_tf_age_sec is not None
        and maximum_tf_age_sec <= 2.5,
        "tracking_loss_zero": tracking_lost_count == 0,
        "unexpected_exit_zero": unexpected_exit_count == 0,
        "reset_pollution_zero": reset_pollution_count == 0,
        "shadow_has_no_motion_or_stop_authority": stop_authority_count == 0,
    }
    return {
        "schema_version": 1,
        "status": "COMPONENT_SMOKE_PASS" if all(checks.values()) else "FAIL",
        "mode": "cuvslam_shadow_gt_authority",
        "checks": checks,
        "sample_count": len(samples),
        "reset_count": reset_count,
        "minimum_reset_count": 2,
        "sim_duration_sec": sim_duration_sec,
        "minimum_sim_duration_sec": 120.0,
        "ate_rmse_m": ate_rmse_m,
        "translation_rpe_rmse_m": _rmse(translation_rpe),
        "rotation_rpe_rmse_rad": _rmse(rotation_rpe),
        "maximum_yaw_error_rad": maximum_yaw_error_rad,
        "tf_age_p95_sec": tf_age_p95_sec,
        "maximum_tf_age_sec": maximum_tf_age_sec,
        "stereo_pair_max_sync_error_ms": stereo_skew,
        "tracking_lost_count": tracking_lost_count,
        "unexpected_exit_count": unexpected_exit_count,
        "reset_pollution_count": reset_pollution_count,
        "shadow_stop_authority_count": stop_authority_count,
        "navigation_pose_authority": "ground_truth",
        "online_navigation_acceptance": "NOT_EVALUATED",
        "nav_candidate_pass": False,
        "episodes": episode_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = analyze(args.result_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0 if payload["status"] == "COMPONENT_SMOKE_PASS" else 2
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        failure = {"schema_version": 1, "status": "FAIL", "error": str(exc)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
