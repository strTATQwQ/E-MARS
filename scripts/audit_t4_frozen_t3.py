#!/usr/bin/env python3
"""Freeze T3 evidence into a separate, deterministic T4.0 audit namespace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
MOVE_THRESHOLD_MPS = 0.02
INTERVAL_CAP_SEC = 1.0

FROZEN_MODES = {
    "go2_flash_nav2_sampled_baseline":
        "t3_0_go2_flash_nav2_sampled_baseline_attempt_007",
    "continuous_no_obstacle": "t3_4_pilot_no_obstacle_attempt_002",
    "continuous_obstacle_aware": "t3_4_pilot_obstacle_aware_attempt_002",
}
STRESS_DIR = "t3_4_obstacle_stress_attempt_009"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def directory_manifest(root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    tree = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest = sha256(path)
        size = path.stat().st_size
        entries.append({"path": relative, "bytes": size, "sha256": digest})
        tree.update(relative.encode("utf-8"))
        tree.update(b"\0")
        tree.update(digest.encode("ascii"))
        tree.update(b"\0")
        tree.update(str(size).encode("ascii"))
        tree.update(b"\n")
    return {
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "tree_sha256": tree.hexdigest(),
        "files": entries,
    }


def safe_rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def horizontal_speed(record: dict[str, Any]) -> float:
    velocity = record.get("actual_linear_velocity_world") or [0.0, 0.0, 0.0]
    return math.hypot(float(velocity[0]), float(velocity[1]))


def summarize_intervals(records_path: Path) -> dict[str, Any]:
    previous: dict[str, Any] | None = None
    total_duration = 0.0
    timeout_duration = 0.0
    moving_duration = 0.0
    motion_enabled_duration = 0.0
    moving_while_enabled_duration = 0.0
    excluded_gap_duration = 0.0
    timeout_sample_count = 0
    evaluated_sample_count = 0
    per_episode: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "observed_duration_sec": 0.0,
            "moving_duration_sec": 0.0,
            "motion_enabled_duration_sec": 0.0,
            "timeout_duration_sec": 0.0,
            "longest_hold_sec": 0.0,
            "current_hold_sec": 0.0,
            "sample_count": 0,
        }
    )

    def consume(record: dict[str, Any], next_timestamp: float) -> None:
        nonlocal total_duration, timeout_duration, moving_duration
        nonlocal motion_enabled_duration, moving_while_enabled_duration
        nonlocal excluded_gap_duration, timeout_sample_count, evaluated_sample_count
        episode_id = str(record.get("episode_id", ""))
        if not episode_id.startswith("isaac-evaluator-episode-"):
            return
        if bool(record.get("state_only", False)):
            return
        timestamp = float(record["wall_time_unix"])
        raw_dt = max(0.0, next_timestamp - timestamp)
        dt = min(raw_dt, INTERVAL_CAP_SEC)
        excluded_gap_duration += max(0.0, raw_dt - dt)
        if dt <= 0.0:
            return
        item = per_episode[episode_id]
        item["sample_count"] += 1
        item["observed_duration_sec"] += dt
        total_duration += dt
        evaluated_sample_count += 1
        moving = horizontal_speed(record) >= MOVE_THRESHOLD_MPS
        enabled = bool(record.get("motion_enabled", False))
        timed_out = bool(record.get("command_timeout", False))
        if moving:
            moving_duration += dt
            item["moving_duration_sec"] += dt
            item["current_hold_sec"] = 0.0
        else:
            item["current_hold_sec"] += dt
            item["longest_hold_sec"] = max(
                item["longest_hold_sec"], item["current_hold_sec"]
            )
        if enabled:
            motion_enabled_duration += dt
            item["motion_enabled_duration_sec"] += dt
            if moving:
                moving_while_enabled_duration += dt
        if timed_out:
            timeout_sample_count += 1
            timeout_duration += dt
            item["timeout_duration_sec"] += dt

    with records_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            current = json.loads(line)
            if previous is not None:
                consume(previous, float(current["wall_time_unix"]))
            previous = current
    if previous is not None:
        consume(previous, float(previous["wall_time_unix"]))

    rows = []
    for episode_id in sorted(
        per_episode,
        key=lambda value: int(value.rsplit("-", 1)[-1]),
    ):
        item = per_episode[episode_id]
        item["motion_duty_cycle"] = safe_rate(
            item["moving_duration_sec"], item["observed_duration_sec"]
        )
        item["moving_while_enabled_duty_cycle"] = safe_rate(
            item["moving_duration_sec"], item["motion_enabled_duration_sec"]
        )
        item.pop("current_hold_sec", None)
        rows.append({"episode_id": episode_id, **item})

    return {
        "definitions": {
            "movement": f"horizontal actual speed >= {MOVE_THRESHOLD_MPS:.3f} m/s",
            "eligible_interval": (
                "non-bootstrap, non-state-only evaluator record to its successor"
            ),
            "interval_cap_sec": INTERVAL_CAP_SEC,
            "interval_cap_reason": (
                "do not assign long model/evaluator gaps to the preceding controller state"
            ),
            "timeout": "eligible interval whose source record has command_timeout=true",
            "hold": "consecutive eligible intervals below the movement threshold",
        },
        "evaluated_sample_count": evaluated_sample_count,
        "timeout_sample_count": timeout_sample_count,
        "observed_duration_sec": total_duration,
        "excluded_gap_duration_sec": excluded_gap_duration,
        "command_timeout_duration_sec": timeout_duration,
        "motion_duty_cycle": safe_rate(moving_duration, total_duration),
        "motion_enabled_duty_cycle": safe_rate(motion_enabled_duration, total_duration),
        "moving_while_enabled_duty_cycle": safe_rate(
            moving_while_enabled_duration, motion_enabled_duration
        ),
        "longest_hold_sec": max(
            (item["longest_hold_sec"] for item in per_episode.values()), default=0.0
        ),
        "per_episode": rows,
    }


def compact_metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("val_unseen", result)
    names = ("Count", "SR", "OS", "SPL", "NE", "TL", "FR", "StR")
    return {name: metrics.get(name) for name in names}


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("required T3 evidence is missing: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/internnav_t4/t4_0_frozen_t3"),
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = (workspace / args.output).resolve() if not args.output.is_absolute() else args.output
    t3_root = workspace / "results/internnav_t3"
    comparison_path = t3_root / "t3_4_comparison.json"
    stress = t3_root / STRESS_DIR
    model_audit_path = t3_root / "t3_4_model_attempt_017/model_weight_audit.json"
    require_files(
        [
            comparison_path,
            stress / "validation.json",
            stress / "controller_records.jsonl",
            stress / "controller_per_episode.json",
            stress / "phase_status.json",
            stress / "ros_graph_snapshot.txt",
            stress / "input_manifest.json",
            model_audit_path,
        ]
    )

    comparison = load_json(comparison_path)
    mode_by_name = {item["mode"]: item for item in comparison["modes"]}
    frozen_modes = []
    manifests = {}
    for mode, directory in FROZEN_MODES.items():
        source = t3_root / directory
        require_files([source / "result.json", source / "phase_status.json"])
        tree = directory_manifest(source)
        manifests[mode] = tree
        phase_status = load_json(source / "phase_status.json")
        frozen_modes.append(
            {
                "mode": mode,
                "source_directory": f"results/internnav_t3/{directory}",
                "status": phase_status.get("status"),
                "exit_codes": phase_status.get("exit_codes"),
                "duration_sec": phase_status.get("duration_sec"),
                "metrics": compact_metrics(load_json(source / "result.json")),
                "comparison": mode_by_name[mode],
                "evidence_tree_sha256": tree["tree_sha256"],
                "evidence_file_count": tree["file_count"],
                "evidence_total_bytes": tree["total_bytes"],
            }
        )

    validation = load_json(stress / "validation.json")
    phase_status = load_json(stress / "phase_status.json")
    per_episode = load_json(stress / "per_episode.json")
    controller_per_episode = load_json(stress / "controller_per_episode.json")
    controller_rows = controller_per_episode["episodes"]
    movement_distances = [float(item["measured_path_length_m"]) for item in controller_rows]
    interval_summary = summarize_intervals(stress / "controller_records.jsonl")

    official_sr = float(validation["metrics"]["SR"])
    termination_sr = float(validation["termination_label_success_rate"])
    reconciliation = {
        "official_metric_sr": official_sr,
        "official_metric_success_count": round(official_sr * int(validation["metrics"]["Count"])),
        "termination_label_success_rate": termination_sr,
        "termination_label_success_count": int(per_episode["success_count"]),
        "difference_rate": official_sr - termination_sr,
        "interpretation": (
            "Official SR is the authoritative task metric computed from final navigation "
            "state. The termination-label rate counts only episodes explicitly labeled "
            "success by the runtime. They are different aggregations and are reported "
            "separately; no value is imputed."
        ),
    }

    ros_graph_path = stress / "ros_graph_snapshot.txt"
    graph_text = ros_graph_path.read_text(encoding="utf-8")
    required_graph_items = [
        "/internvla_go2_controller_bridge",
        "/internvla_nav2_active_adapter",
        "/controller_server",
        "/collision_monitor",
        "/go2/depth/points",
        "/odom",
        "/map",
        "/cmd_vel_safe",
    ]
    graph_audit = {
        "snapshot_sha256": sha256(ros_graph_path),
        "required_items": {
            item: item in graph_text for item in required_graph_items
        },
    }
    graph_audit["status"] = (
        "PASS" if all(graph_audit["required_items"].values()) else "FAIL"
    )

    sanitized_logs = []
    logs_root = stress / "logs_sanitized"
    for path in sorted(item for item in logs_root.rglob("*") if item.is_file()):
        sanitized_logs.append(
            {
                "path": path.relative_to(stress).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    input_manifest = load_json(stress / "input_manifest.json")
    input_hashes = {item["label"]: item["sha256"] for item in input_manifest["files"]}
    model_audit = load_json(model_audit_path)
    hash_contract = {
        "model": {
            "inventory_sha256": model_audit["inventory_sha256"],
            "model_revision": model_audit["model_revision"],
            "checkpoint_revision": model_audit["checkpoint_revision"],
            "weight_audit_sha256": sha256(model_audit_path),
        },
        "map": {
            "static_map_manifest_sha256": input_hashes.get("static_map_manifest")
        },
        "sensors": {
            "go2_asset_manifest_sha256": sha256(stress / "go2_asset_manifest.json"),
            "go2_runtime_sha256": input_hashes.get("go2_runtime"),
            "controller_bridge_sha256": input_hashes.get("controller_bridge"),
        },
        "config": {
            key: input_hashes.get(key)
            for key in (
                "dataset",
                "eval_config",
                "nav2_params",
                "obstacle_exclusions",
                "collision_monitor_contract",
                "phase_launcher",
                "continuous_agent",
                "nav2_adapter",
            )
        },
    }

    final_status = {
        "phase": phase_status.get("phase"),
        "status": phase_status.get("status"),
        "duration_sec": phase_status.get("duration_sec"),
        "exit_codes": phase_status.get("exit_codes"),
        "active_status": validation["active"].get("status"),
        "client_status": validation["client"].get("status"),
        "controller_status": validation["controller"].get("status"),
        "validation_status": validation.get("status"),
    }
    all_exits_zero = all(value == 0 for value in final_status["exit_codes"].values())
    final_status["all_exit_codes_zero"] = all_exits_zero

    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "source_policy": "read-only derivation; no T3 artifact was modified",
        "missing_metric_policy": "null_not_imputed",
        "frozen_modes": frozen_modes,
        "stress": {
            "source_directory": f"results/internnav_t3/{STRESS_DIR}",
            "metrics": validation["metrics"],
            "official_vs_termination": reconciliation,
            "timing_and_motion": interval_summary,
            "minimum_actual_movement_distance_m": min(movement_distances),
            "movement_distance_by_episode_m": [
                {
                    "episode_id": item["episode_id"],
                    "measured_path_length_m": item["measured_path_length_m"],
                }
                for item in controller_rows
            ],
            "final_status": final_status,
            "ros_graph": graph_audit,
            "sanitized_logs": {
                "file_count": len(sanitized_logs),
                "files": sanitized_logs,
            },
        },
        "hash_contract": hash_contract,
    }
    checks = {
        "three_modes_frozen": len(frozen_modes) == 3,
        "all_frozen_modes_pass": all(item["status"] == "PASS" for item in frozen_modes),
        "stress_validation_pass": validation.get("status") == "PASS",
        "stress_all_exits_zero": all_exits_zero,
        "ros_graph_complete": graph_audit["status"] == "PASS",
        "sanitized_logs_present": bool(sanitized_logs),
        "hash_contract_complete": all(
            value is not None
            for section in hash_contract.values()
            for value in section.values()
        ),
    }
    audit["checks"] = checks
    audit["status"] = "PASS" if all(checks.values()) else "FAIL"

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "audit.json", audit)
    for mode, manifest in manifests.items():
        write_json(output / "freeze_manifests" / f"{mode}.json", manifest)
    write_json(output / "stress_timing_motion.json", interval_summary)
    write_json(output / "stress_metric_reconciliation.json", reconciliation)
    write_json(output / "hash_contract.json", hash_contract)
    print(json.dumps({"status": audit["status"], "output": str(output)}, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
