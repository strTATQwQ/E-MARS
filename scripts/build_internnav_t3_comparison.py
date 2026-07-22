#!/usr/bin/env python3
"""Build the paired three-mode T3 comparison without inventing missing metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def mean_finite(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def summarize(name: str, root: Path, *, flash: bool) -> dict[str, Any]:
    result = load(root / "result.json")
    active = load(root / "active_summary.json")
    per_episode = load(root / "per_episode.json")
    controller = load(root / "controller_summary.json")
    controller_episode = load(root / "controller_per_episode.json")
    client = load(root / "client_summary.json")
    validation = load(root / "validation.json")
    if result is None or active is None or per_episode is None or validation is None:
        raise FileNotFoundError(f"incomplete comparison input: {name}")
    metrics = result.get("val_unseen", result)
    active_records = []
    records_path = root / "active_records.jsonl"
    if records_path.is_file():
        active_records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    nav2_latencies = [
        float(item["resolution_latency_sec"])
        for item in active_records
        if "resolution_latency_sec" in item
    ]
    episode_rows = controller_episode.get("episodes", []) if controller_episode else []
    output = {
        "mode": name,
        "validation_status": validation.get("status"),
        "Count": int(metrics.get("Count", 0)),
        "SR": metrics.get("SR"),
        "OS": metrics.get("OS"),
        "SPL": metrics.get("SPL"),
        "NE": metrics.get("NE"),
        "fall_count": sum(
            item.get("termination_reason") == "fall" for item in per_episode.get("episodes", [])
        ) if flash else (controller or {}).get("fall_count"),
        "physical_collision_count": None if flash else (controller or {}).get("physical_collision_count"),
        "mean_lateral_error_m": (
            controller_episode.get("aggregate", {}).get("mean_lateral_error_m")
            if controller_episode else None
        ),
        "p95_lateral_error_m": (
            controller_episode.get("aggregate", {}).get("p95_lateral_error_m")
            if controller_episode else None
        ),
        "mean_heading_error_rad": (
            controller_episode.get("aggregate", {}).get("mean_heading_error_rad")
            if controller_episode else None
        ),
        "measured_path_length_m": (
            sum(float(item.get("measured_path_length_m", 0.0)) for item in episode_rows)
            if controller_episode else None
        ),
        "recovery_count": (
            sum(int(item.get("recovery_count", 0)) for item in episode_rows)
            if controller_episode else None
        ),
        "goal_or_path_rejection_count": int(active.get("failure_count", 0)),
        "mean_model_inference_latency_sec": (
            client.get("mean_inference_latency_sec") if client else None
        ),
        "mean_nav2_resolution_latency_sec": (
            client.get("mean_nav2_resolution_latency_sec")
            if client else mean_finite(nav2_latencies)
        ),
        "control_hz": None if flash else (controller or {}).get("measured_control_hz"),
        "stale_or_identity_reject_count": (
            None if flash else (controller or {}).get("stale_or_identity_reject_count")
        ),
        "command_timeout_count": None if flash else (controller or {}).get("command_timeout_count"),
        "reset_barrier_count": None if flash else (controller or {}).get("reset_barrier_count"),
        "flash_call_count": (
            sum(
                int(count)
                for action, count in validation.get("selected_action_distribution", {}).items()
                if int(action) != 0
            )
            if flash else (controller or {}).get("flash_call_count")
        ),
        "cmd_vel_quantization_count": (
            int(active.get("nav2_control_count", 0))
            if flash else (controller or {}).get("cmd_vel_quantization_count")
        ),
        "direct_motion_bypass_count": (
            int(active.get("system2_passthrough_count", 0))
            if flash else (controller or {}).get("direct_motion_bypass_count")
        ),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flash", required=True, type=Path)
    parser.add_argument("--continuous", required=True, type=Path)
    parser.add_argument("--obstacle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "episode_policy": {
            "continuous_pair": "same_frozen_safe_twenty_episode_model_set",
            "flash_reference": "frozen_t3_0_twenty_episode_set_unpaired_to_continuous",
        },
        "interpretation_limit": (
            "continuous_no_obstacle_vs_obstacle_aware_is_paired; "
            "flash_vs_continuous_is_descriptive_only because the frozen flash "
            "episodes did not pass the Go2 continuous static-clearance gate"
        ),
        "missing_metric_policy": "null_not_imputed",
        "modes": [
            summarize("go2_flash_nav2_sampled_baseline", args.flash, flash=True),
            summarize("continuous_no_obstacle", args.continuous, flash=False),
            summarize("continuous_obstacle_aware", args.obstacle, flash=False),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
