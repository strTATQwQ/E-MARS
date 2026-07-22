#!/usr/bin/env python3
"""Aggregate one T4 gate from immutable Oracle and InternVLA result trees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def metric(result_dir: Path, name: str) -> float:
    result = load(result_dir / "result.json")
    metrics = result.get("val_unseen", result)
    return float(metrics[name])


def run_contract(result_dir: Path, expected: int) -> dict[str, Any]:
    phase = load(result_dir / "phase_status.json")
    controller = load(result_dir / "controller_summary.json")
    per_episode = load(result_dir / "per_episode.json")
    nvblox = jsonl(result_dir / "nvblox_resets.jsonl")
    odometry = (
        load(result_dir / "odometry_metrics.json")
        if (result_dir / "odometry_metrics.json").is_file()
        else None
    )
    generations = {
        int(record["generation"])
        for record in nvblox
        if record.get("event") == "reset_complete"
    }
    ready_generations = {
        int(record["generation"])
        for record in nvblox
        if record.get("event") == "first_map_slice"
    }
    child_pids = [
        int(record["pid"])
        for record in nvblox
        if record.get("event") == "start" and int(record.get("generation", -1)) >= 0
    ]
    map_pollution_count = sum(
        int(record.get("map_pollution_count", 0))
        for record in nvblox
        if record.get("event") == "first_map_slice"
    )
    mapping_enabled = str(controller.get("map_source", "")) == "nvblox_online"
    mapping_reset_ok = (
        not mapping_enabled
        or (
            len(generations) == expected
            and ready_generations == generations
            and len(child_pids) == expected
            and len(set(child_pids)) == expected
            and not any(item.get("event") == "fatal_unexpected_exit" for item in nvblox)
            and map_pollution_count == 0
        )
    )
    stale = max(
        int(controller.get("stale_or_identity_reject_count", 0)),
        int(controller.get("stale_count", 0)),
    )
    safety = {
        "nan_count": int(controller.get("nan_count", 0)),
        "fall_count": int(controller.get("fall_count", 0)),
        "physical_collision_count": int(controller.get("physical_collision_count", 0)),
        "stale_execution_count": stale,
    }
    completed = int(per_episode.get("completed_episode_count", 0))
    passed = (
        phase.get("status") == "PASS"
        and completed == expected
        and all(value == 0 for value in safety.values())
        and mapping_reset_ok
        and (odometry is None or odometry.get("status") == "PASS")
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "result_dir": result_dir.as_posix(),
        "episode_count": completed,
        "sr": metric(result_dir, "SR"),
        "os": metric(result_dir, "OS"),
        "safety": safety,
        "controller": controller,
        "mapping_enabled": mapping_enabled,
        "mapping_reset_ok": mapping_reset_ok,
        "mapping_reset_generations": sorted(generations),
        "map_ready_generations": sorted(ready_generations),
        "map_pollution_count": map_pollution_count,
        "pre_epoch_slice_reject_count": sum(
            record.get("event") == "pre_epoch_slice_rejected" for record in nvblox
        ),
        "odometry": odometry,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("t4_2", "t4_3", "t4_4"), required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--t3-sr", type=float)
    args = parser.parse_args()
    oracle = run_contract(args.oracle_dir, 10)
    model = run_contract(args.model_dir, 20)
    oracle_minimum = 0.9 if args.gate == "t4_2" else 0.8
    model_minimum = 0.2 if args.gate == "t4_4" else 0.0
    model_sr = float(model["sr"])
    model_sr_ok = model_sr >= model_minimum and (model_sr > 0.0 or args.gate == "t4_4")
    relative_ok: bool | None = None
    if args.gate == "t4_2":
        if args.t3_sr is None:
            raise RuntimeError("T4.2 requires --t3-sr for the <=10 percentage-point check")
        relative_ok = model_sr >= args.t3_sr - 0.10
    mapping_required = args.gate in {"t4_2", "t4_4"}
    odometry_required = args.gate in {"t4_3", "t4_4"}
    controller = model["controller"]
    truth_contract_ok = bool(controller.get("ground_truth_pose_used_for_nav", True)) != odometry_required
    map_contract_ok = (
        str(controller.get("map_source", "")) == "nvblox_online"
        if mapping_required
        else str(controller.get("map_source", "")) == "static_map"
    )
    # The frozen 10-episode obstacle Oracle split carries audited targets; the
    # 20-episode model pilot is used for SR and need not inject obstacles into
    # every route.
    obstacle_detection = float(
        oracle["controller"].get("obstacle_detection_rate", 0.0)
    )
    costmap_detection = float(
        oracle["controller"].get("costmap_detection_rate", 0.0)
    )
    obstacle_ok = not mapping_required or (
        obstacle_detection >= 0.95 and costmap_detection >= 0.95
    )
    passed = (
        oracle["status"] == "PASS"
        and model["status"] == "PASS"
        and float(oracle["sr"]) >= oracle_minimum
        and model_sr_ok
        and (relative_ok is not False)
        and truth_contract_ok
        and map_contract_ok
        and obstacle_ok
    )
    payload = {
        "schema_version": 1,
        "gate": args.gate,
        "status": "PASS" if passed else "FAIL",
        "required": {
            "oracle_sr": oracle_minimum,
            "model_sr": model_minimum if args.gate == "t4_4" else "nonzero",
            "maximum_t3_sr_drop": 0.10 if args.gate == "t4_2" else None,
            "obstacle_detection_rate": 0.95 if mapping_required else None,
            "nvblox_costmap_detection_rate": 0.95 if mapping_required else None,
            "collision_fall_stale": 0,
        },
        "oracle": oracle,
        "model": model,
        "relative_to_t3_ok": relative_ok,
        "obstacle_detection_rate_minimum_observed": obstacle_detection,
        "nvblox_costmap_detection_rate_minimum_observed": costmap_detection,
        "truth_contract_ok": truth_contract_ok,
        "map_contract_ok": map_contract_ok,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
