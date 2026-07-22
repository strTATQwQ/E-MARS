from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config_loader import load_data


def evaluate_sim2real_session(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    metrics = load_data(run_dir / "metrics.json")
    episodes = list(metrics.get("episodes") or [])
    episode_types = {str(row.get("episode_id") or ""): str(row.get("task_type") or "") for row in episodes}
    successes = {
        task_type: sum(bool(row.get("success")) for row in episodes if row.get("task_type") == task_type)
        for task_type in ("simple_navigation", "turn_choice", "semantic_target")
    }
    totals = {
        task_type: sum(row.get("task_type") == task_type for row in episodes)
        for task_type in successes
    }
    safe_max = 0.0
    actual_max = 0.0
    actual_speeds: list[float] = []
    policy_max = 0.0
    safe_yaw_max = 0.0
    actual_yaw_max = 0.0
    actual_yaw_rates: list[float] = []
    policy_yaw_max = 0.0
    servo_samples = 0
    servo_active = 0
    yaw_servo_samples = 0
    yaw_servo_active = 0
    motion_guard_samples = 0
    motion_guard_active = 0
    step_http = {"accepted": 0, "route_choice": 0, "semantic_stop": 0, "errors": 0, "fallback_or_mock": 0, "latency_s": []}
    semantic_requests: dict[str, float] = {}
    semantic_stop_latencies: list[float] = []
    semantic_stop_seen: set[str] = set()
    fallen_episodes: set[str] = set()
    counts = {
        "runtime_stale_discards": 0,
        "timebase_error": 0,
        "parse_error": 0,
        "collision_count": 0,
        "fall_count": 0,
        "stale_action_executed": 0,
        "old_response_after_reset": 0,
        "episode_mismatch": 0,
    }
    with (run_dir / "events.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            name = str(event.get("event") or "")
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            event_type = str(details.get("event_type") or name)
            episode_id = str(event.get("episode_id") or details.get("episode_id") or "")
            if name == "safe_cmd_vel":
                safe_max = max(safe_max, abs(float(((details.get("linear") or {}).get("x") or 0.0))))
                safe_yaw_max = max(safe_yaw_max, abs(float(((details.get("angular") or {}).get("z") or 0.0))))
            if name == "isaac_ground_truth_pose":
                event_time = float(event.get("t") or 0.0)
                try:
                    if event_time >= 0.5 and float(details.get("z")) < 0.18 and episode_id:
                        fallen_episodes.add(episode_id)
                except (TypeError, ValueError):
                    pass
                if event_time < 10.0:
                    continue
                velocity = details.get("linear_velocity") or []
                if len(velocity) >= 2:
                    actual_speed = math.hypot(float(velocity[0]), float(velocity[1]))
                    actual_speeds.append(actual_speed)
                    actual_max = max(actual_max, actual_speed)
                angular_velocity = details.get("angular_velocity") or []
                if len(angular_velocity) >= 3:
                    actual_yaw = abs(float(angular_velocity[2]))
                    actual_yaw_rates.append(actual_yaw)
                    actual_yaw_max = max(actual_yaw_max, actual_yaw)
                command = details.get("command") if isinstance(details.get("command"), dict) else {}
                if bool(command.get("low_speed_servo_enabled")):
                    servo_samples += 1
                    servo_active += int(bool(command.get("low_speed_servo_active")))
                    policy_max = max(policy_max, abs(float(command.get("vx") or 0.0)))
                if bool(command.get("low_speed_yaw_servo_enabled")):
                    yaw_servo_samples += 1
                    yaw_servo_active += int(bool(command.get("low_speed_yaw_servo_active")))
                    policy_yaw_max = max(policy_yaw_max, abs(float(command.get("wz") or 0.0)))
                if bool(command.get("low_speed_motion_guard_enabled")):
                    motion_guard_samples += 1
                    motion_guard_active += int(bool(command.get("low_speed_motion_guard_active")))
            if event_type == "step_http_response":
                model = str(details.get("model") or "").lower()
                result = str(details.get("result") or "")
                if result == "accepted" and model == "step_http":
                    step_http["accepted"] += 1
                    role = str(details.get("role") or "")
                    if role in {"route_choice", "semantic_stop"}:
                        step_http[role] += 1
                    try:
                        step_http["latency_s"].append(float(details.get("latency_s")))
                    except (TypeError, ValueError):
                        pass
                else:
                    step_http["errors"] += 1
            model = str(details.get("model") or "").lower()
            if (event_type.lower().startswith("step_") or model.startswith("step")) and (
                "fallback" in event_type.lower() or "mock" in model
            ):
                step_http["fallback_or_mock"] += 1
            if name == "step_request_json" and str(details.get("role") or "") == "semantic_stop":
                semantic_requests.setdefault(episode_id, float(event.get("t") or 0.0))
            if event_type == "primitive_command" and str(details.get("action_type") or "") == "stop":
                request_time = semantic_requests.get(episode_id)
                if request_time is not None and episode_types.get(episode_id) == "semantic_target":
                    latency = float(event.get("t") or 0.0) - request_time
                    if latency >= 0.0 and episode_id not in semantic_stop_seen:
                        semantic_stop_latencies.append(latency)
                        semantic_stop_seen.add(episode_id)
            mapping = {
                "runtime_stale_discard": "runtime_stale_discards",
                "timebase_error": "timebase_error",
                "parse_error": "parse_error",
                "collision": "collision_count",
                "fall": "fall_count",
                "robot_fallen_or_unstable": "fall_count",
                "stale_action_executed": "stale_action_executed",
                "old_response_after_reset": "old_response_after_reset",
                "episode_mismatch": "episode_mismatch",
            }
            if event_type in mapping:
                counts[mapping[event_type]] += 1

    counts["fall_count"] = max(counts["fall_count"], len(fallen_episodes))
    expected_route = totals["turn_choice"]
    expected_semantic = totals["semantic_target"]
    expected_step_calls = expected_route + expected_semantic
    p95_http = _nearest_rank(step_http["latency_s"], 0.95)
    p95_stop = _nearest_rank(semantic_stop_latencies, 0.95)
    p95_actual = _nearest_rank(actual_speeds, 0.95)
    p95_actual_yaw = _nearest_rank(actual_yaw_rates, 0.95)
    failures: list[str] = []
    if not episodes or any(not bool(row.get("success")) for row in episodes):
        failures.append("not all episodes succeeded")
    if step_http["accepted"] != expected_step_calls:
        failures.append(f"accepted Step calls {step_http['accepted']} != {expected_step_calls}")
    if step_http["route_choice"] != expected_route or step_http["semantic_stop"] != expected_semantic:
        failures.append("Step role call matrix mismatch")
    if step_http["errors"] or step_http["fallback_or_mock"]:
        failures.append("Step HTTP errors/mock/fallback present")
    if expected_semantic and (p95_stop is None or p95_stop > 2.0):
        failures.append("semantic request-to-stop p95 > 2s")
    if safe_max > 0.2001:
        failures.append("safe command exceeded 0.20 m/s")
    if p95_actual is None:
        failures.append("no post-warmup actual speed telemetry")
    elif p95_actual > 0.2001:
        failures.append("actual speed p95 exceeded 0.20 m/s")
    if actual_max > 0.2601:
        failures.append("actual speed raw max exceeded 0.26 m/s")
    if safe_yaw_max > 0.3001:
        failures.append("safe yaw command exceeded 0.30 rad/s")
    if p95_actual_yaw is None:
        failures.append("no post-warmup actual yaw telemetry")
    elif p95_actual_yaw > 0.3001:
        failures.append("actual yaw rate p95 exceeded 0.30 rad/s")
    if actual_yaw_max > 0.3901:
        failures.append("actual yaw rate raw max exceeded 0.39 rad/s")
    if any(counts.values()):
        failures.append("safety/stale/timebase/parse count nonzero")
    step_http.pop("latency_s")
    step_http["p95_latency_s"] = p95_http
    result = {
        "schema_version": 1,
        "pass": not failures,
        "run_id": run_dir.name,
        "episodes": len(episodes),
        "task_totals": totals,
        "task_successes": successes,
        "max_linear_x_mps": round(safe_max, 6),
        "actual_speed_max_mps": round(actual_max, 6) if actual_speeds else None,
        "actual_speed_p95_mps": p95_actual,
        "policy_cmd_max_mps": round(policy_max, 6),
        "low_speed_servo_active_ratio": round(servo_active / max(servo_samples, 1), 6),
        "max_yaw_cmd_radps": round(safe_yaw_max, 6),
        "actual_yaw_rate_max_radps": round(actual_yaw_max, 6) if actual_yaw_rates else None,
        "actual_yaw_rate_p95_radps": p95_actual_yaw,
        "policy_yaw_cmd_max_radps": round(policy_yaw_max, 6),
        "low_speed_yaw_servo_active_ratio": round(yaw_servo_active / max(yaw_servo_samples, 1), 6),
        "low_speed_motion_guard_active_ratio": round(motion_guard_active / max(motion_guard_samples, 1), 6),
        "real_step_http": step_http,
        "semantic_request_to_stop_p95_sec": p95_stop,
        "semantic_request_to_stop_latencies_sec": [round(value, 6) for value in semantic_stop_latencies],
        **counts,
        "failures": failures,
        "real_robot_motion_enabled": False,
    }
    return result


def write_sim2real_session(run_dir: Path, result: dict[str, Any]) -> None:
    run_dir = Path(run_dir)
    (run_dir / "session_qualification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Sim2Real Low-Speed Session",
        "",
        f"- pass: {result['pass']}",
        f"- episodes: {result['episodes']}",
        f"- successes: {result['task_successes']}",
        f"- safe max / actual p95 / actual raw max: {_fmt_metric(result['max_linear_x_mps'])}/{_fmt_metric(result['actual_speed_p95_mps'])}/{_fmt_metric(result['actual_speed_max_mps'])} m/s",
        f"- safe yaw max / actual p95 / actual raw max: {_fmt_metric(result['max_yaw_cmd_radps'])}/{_fmt_metric(result['actual_yaw_rate_p95_radps'])}/{_fmt_metric(result['actual_yaw_rate_max_radps'])} rad/s",
        f"- low-speed motion guard active ratio: {result['low_speed_motion_guard_active_ratio']:.3f}",
        f"- semantic request-to-stop p95: {result['semantic_request_to_stop_p95_sec']}",
        f"- Step HTTP: {result['real_step_http']}",
        f"- failures: {result['failures']}",
        "- real robot motion enabled: false",
    ]
    (run_dir / "session_qualification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _nearest_rank(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return round(ordered[index], 6)
