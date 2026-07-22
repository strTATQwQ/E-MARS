#!/usr/bin/env python3
"""Summarize one sensor-fed T5 Nvblox run without overclaiming Nav success."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("{} must contain an object".format(path))
    return value


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("{}:{} must contain an object".format(path, index))
        rows.append(value)
    return rows


def _jsonl_optional(path: Path) -> List[Dict[str, Any]]:
    return _jsonl(path) if path.is_file() else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("shadow", "active_local_gt"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        root = args.result_dir.resolve()
        events = _jsonl(root / "nvblox_runtime.jsonl")
        slices = _jsonl(root / "nvblox_slice_classes.jsonl")
        costmap_evidence = _jsonl_optional(root / "nvblox_costmap_evidence.jsonl")
        ready = _json(root / "nvblox_ready.json")
        child_starts = [row for row in events if row.get("event") == "nvblox_child_start"]
        child_stops = [row for row in events if row.get("event") == "nvblox_child_stop"]
        resets = [row for row in events if row.get("event") == "episode_reset_complete"]
        layer_enable = [
            row
            for row in events
            if row.get("event") == "nvblox_layer_set"
            and row.get("enabled") is True
            and row.get("success") is True
        ]
        maximum_run = max(
            (int(row.get("consecutive_valid_slices", 0)) for row in slices),
            default=0,
        )
        classes = set()
        slice_stamps_by_generation: Dict[int, List[int]] = {}
        for row in slices:
            stamp = row.get("slice_stamp_ns")
            if isinstance(stamp, int) and stamp > 0:
                generation = int(row.get("generation", -1))
                slice_stamps_by_generation.setdefault(generation, []).append(stamp)
            if int(row.get("unknown_count", 0)) > 0:
                classes.add("unknown")
            if int(row.get("free_positive_count", 0)) > 0:
                classes.add("free")
            if int(row.get("occupied_nonpositive_count", 0)) > 0:
                classes.add("occupied")
        expected_ready = (
            "SHADOW_SENSOR_FED_READY"
            if args.mode == "shadow"
            else "ACTIVE_LOCAL_READY"
        )
        sim_duration_sec = sum(
            (max(stamps) - min(stamps)) / 1_000_000_000.0
            for stamps in slice_stamps_by_generation.values()
            if len(stamps) >= 2
        )
        minimum_resets = 2 if args.mode == "shadow" else 0
        minimum_sim_duration_sec = 120.0 if args.mode == "shadow" else 0.0
        ready_window_samples = [
            row
            for row in costmap_evidence
            if row.get("event") == "ready_window_sample"
        ]
        ready_window_fraction = (
            sum(row.get("ready") is True for row in ready_window_samples)
            / len(ready_window_samples)
            if ready_window_samples
            else 0.0
        )
        maximum_costmap_updates = max(
            (
                int(row.get("consecutive_updates", 0))
                for row in costmap_evidence
                if row.get("event") == "nvblox_backed_costmap_update"
            ),
            default=0,
        )
        checks = {
            "real_nvblox_child_started": bool(child_starts),
            "sensor_fed_ready": ready.get("status") == expected_ready
            and ready.get("real_nvblox_node") is True
            and ready.get("depth_and_lidar_required") is True,
            "ten_consecutive_valid_slices": maximum_run >= 10,
            "free_occupied_unknown_observed": classes
            == {"free", "occupied", "unknown"},
            "minimum_resets": len(resets) >= minimum_resets,
            "minimum_sim_duration": sim_duration_sec >= minimum_sim_duration_sec,
            "active_layer_enabled_after_gate": (
                bool(layer_enable) if args.mode == "active_local_gt" else not layer_enable
            ),
            "no_cmd_vel_or_stop_authority": not any(
                row.get("event") in {"cmd_vel", "terminal_stop"} for row in events
            ),
        }
        if args.mode == "active_local_gt":
            checks.update(
                {
                    "ready_window_sampled": len(ready_window_samples) >= 10,
                    "costmap_ready_window_fraction_gte_0_95": (
                        ready_window_fraction >= 0.95
                    ),
                    "ten_consecutive_costmap_layer_updates": (
                        maximum_costmap_updates >= 10
                    ),
                }
            )
        component_status = (
            "SHADOW_COMPONENT_SMOKE_PASS"
            if args.mode == "shadow"
            else "ACTIVE_LOCAL_COMPONENT_READY"
        )
        payload = {
            "schema_version": 1,
            "status": component_status if all(checks.values()) else "FAIL",
            "mode": args.mode,
            "checks": checks,
            "child_start_count": len(child_starts),
            "child_stop_count": len(child_stops),
            "reset_count": len(resets),
            "sim_duration_sec": sim_duration_sec,
            "minimum_reset_count": minimum_resets,
            "minimum_sim_duration_sec": minimum_sim_duration_sec,
            "maximum_consecutive_valid_slices": maximum_run,
            "costmap_ready_window_sample_count": len(ready_window_samples),
            "costmap_ready_window_fraction": ready_window_fraction,
            "maximum_consecutive_costmap_layer_updates": maximum_costmap_updates,
            "observed_slice_classes": sorted(classes),
            "online_navigation_acceptance": "NOT_EVALUATED",
            "nav_candidate_pass": False,
            "note": (
                "ACTIVE_LOCAL_COMPONENT_READY is not NAV_CANDIDATE_PASS; "
                "fixed-3 Oracle, costmap READY fraction, reset pollution, "
                "and final zero-residual evidence remain coordinator-owned."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0 if payload["status"] != "FAIL" else 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema_version": 1, "status": "FAIL", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
