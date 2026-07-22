from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config_loader import load_data


def evaluate_low_speed_probe(run_dir: Path, scheduler_config: Path) -> dict[str, Any]:
    metrics = load_data(run_dir / "metrics.json")
    episodes = list(metrics.get("episodes") or [])
    config = load_data(scheduler_config)
    safe_linear: list[float] = []
    policy_linear: list[float] = []
    servo_active_samples = 0
    servo_samples = 0
    actual_speeds_after_settle: list[float] = []
    poses: list[tuple[float, float]] = []
    counts = {
        "runtime_stale_discards": 0,
        "timebase_error": 0,
        "parse_error": 0,
        "collision_count": 0,
        "fall_count": 0,
        "stale_action_executed": 0,
    }
    with (run_dir / "events.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            name = str(event.get("event") or "")
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            event_type = str(details.get("event_type") or name)
            if name == "safe_cmd_vel":
                safe_linear.append(float(((details.get("linear") or {}).get("x") or 0.0)))
            if name == "isaac_ground_truth_pose":
                pose = details.get("pose") or []
                velocity = details.get("linear_velocity") or []
                command = details.get("command") if isinstance(details.get("command"), dict) else {}
                if bool(command.get("low_speed_servo_enabled")):
                    servo_samples += 1
                    servo_active_samples += int(bool(command.get("low_speed_servo_active")))
                    policy_linear.append(float(command.get("vx") or 0.0))
                if len(pose) >= 2:
                    poses.append((float(pose[0]), float(pose[1])))
                if float(event.get("t") or 0.0) >= 10.0 and len(velocity) >= 2:
                    actual_speeds_after_settle.append(math.hypot(float(velocity[0]), float(velocity[1])))
            if event_type == "runtime_stale_discard":
                counts["runtime_stale_discards"] += 1
            if event_type in counts:
                counts[event_type] += 1
            if name == "collision":
                counts["collision_count"] += 1
            if name in {"fall", "robot_fallen_or_unstable"}:
                counts["fall_count"] += 1

    max_linear = float((config.get("primitive") or {}).get("max_linear_x", 999.0))
    safe_max = max((abs(value) for value in safe_linear), default=0.0)
    nonzero = sum(abs(value) > 1e-4 for value in safe_linear)
    mean_safe = sum(abs(value) for value in safe_linear) / max(len(safe_linear), 1)
    mean_actual = sum(actual_speeds_after_settle) / max(len(actual_speeds_after_settle), 1)
    max_actual = max(actual_speeds_after_settle, default=0.0)
    p95_actual = _nearest_rank(actual_speeds_after_settle, 0.95)
    displacement = 0.0
    if poses:
        displacement = math.hypot(poses[-1][0] - poses[0][0], poses[-1][1] - poses[0][1])
    episode_success = len(episodes) == 1 and bool(episodes[0].get("success"))
    command_present = len(safe_linear) > 0 and nonzero / len(safe_linear) >= 0.80
    speed_tracks = mean_actual >= 0.08
    actual_speed_bounded = p95_actual is not None and p95_actual <= 0.2001 and max_actual <= 0.2601
    safety_zero = all(value == 0 for value in counts.values())
    passed = episode_success and command_present and speed_tracks and actual_speed_bounded and safe_max <= 0.2001 and safety_zero
    blocker = "none"
    if not actual_speed_bounded:
        blocker = "actual_speed_limit_exceeded"
    elif command_present and not speed_tracks:
        blocker = "isaac_locomotion_policy_low_speed_stall"
    elif not episode_success:
        blocker = str(episodes[0].get("failure_reason") if episodes else "missing_episode")
    return {
        "schema_version": 1,
        "pass": passed,
        "run_id": run_dir.name,
        "max_linear_x_mps": max_linear,
        "max_yaw_rate_radps": float((config.get("primitive") or {}).get("max_yaw_rate", 999.0)),
        "accel_limit_mps2": float((config.get("primitive") or {}).get("accel_limit_mps2", 999.0)),
        "isaac_min_stable_linear_command": 0.0,
        "episode_success": episode_success,
        "failure_reason": episodes[0].get("failure_reason") if episodes else "missing_episode",
        "path_length_m": float(episodes[0].get("path_length_m", 0.0)) if episodes else 0.0,
        "final_distance_to_target_m": float(episodes[0].get("final_distance_to_target_m", 999.0)) if episodes else 999.0,
        "safe_cmd_samples": len(safe_linear),
        "safe_cmd_nonzero_ratio": round(nonzero / max(len(safe_linear), 1), 6),
        "safe_cmd_mean_abs_mps": round(mean_safe, 6),
        "safe_cmd_max_abs_mps": round(safe_max, 6),
        "policy_cmd_max_abs_mps": round(max((abs(value) for value in policy_linear), default=0.0), 6),
        "low_speed_servo_active_ratio": round(servo_active_samples / max(servo_samples, 1), 6),
        "actual_speed_mean_after_10s_mps": round(mean_actual, 6),
        "actual_speed_max_after_10s_mps": round(max_actual, 6),
        "actual_speed_p95_after_10s_mps": p95_actual,
        "ground_truth_displacement_m": round(displacement, 6),
        **counts,
        "blocker": blocker,
        "real_robot_motion_enabled": False,
    }


def write_low_speed_probe(output: Path, result: dict[str, Any]) -> None:
    (output / "low_speed_qualification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Low-Speed Qualification Probe",
        "",
        f"- pass: {result['pass']}",
        f"- blocker: `{result['blocker']}`",
        f"- episode success: {result['episode_success']}",
        f"- safe command mean/max: {result['safe_cmd_mean_abs_mps']:.3f}/{result['safe_cmd_max_abs_mps']:.3f} m/s",
        f"- actual mean speed after 10 s: {result['actual_speed_mean_after_10s_mps']:.3f} m/s",
        f"- actual max speed after 10 s: {result['actual_speed_max_after_10s_mps']:.3f} m/s",
        f"- policy pulse max: {result['policy_cmd_max_abs_mps']:.3f} m/s",
        f"- ground-truth displacement: {result['ground_truth_displacement_m']:.3f} m",
        f"- final target distance: {result['final_distance_to_target_m']:.3f} m",
        "- real robot motion enabled: false",
    ]
    (output / "low_speed_qualification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _nearest_rank(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return round(ordered[index], 6)
