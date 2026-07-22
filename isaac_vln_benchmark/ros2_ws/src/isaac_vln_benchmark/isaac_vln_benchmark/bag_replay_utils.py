from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def evaluate_bag_records(records: Iterable[tuple[str, Any]], expected_metrics: dict[str, Any]) -> dict[str, Any]:
    topic_counts: Counter[str] = Counter()
    active_episode = ""
    observed_episodes: set[str] = set()
    successful_episodes: set[str] = set()
    safe_linear_max = 0.0
    safe_yaw_max = 0.0
    collisions = 0
    stale_action_executed = 0
    timebase_error = 0
    parse_error = 0
    step_route = 0
    step_stop = 0
    primitives = 0
    for topic, value in records:
        topic_counts[topic] += 1
        payload = _json_payload(value)
        if topic == "/benchmark/mode_json":
            active_episode = str(payload.get("episode_id") or active_episode)
            if active_episode:
                observed_episodes.add(active_episode)
        elif topic == "/isaac/episode_status":
            episode_id = str(payload.get("episode_id") or active_episode)
            if episode_id:
                observed_episodes.add(episode_id)
            if bool(payload.get("success")) and episode_id:
                successful_episodes.add(episode_id)
        elif topic == "/safe_cmd_vel":
            linear, angular = _twist_components(value)
            safe_linear_max = max(safe_linear_max, abs(linear))
            safe_yaw_max = max(safe_yaw_max, abs(angular))
        elif topic == "/isaac/collision_event":
            collisions += 1
        elif topic == "/step/route_choice_json":
            step_route += 1
        elif topic == "/step/semantic_stop_json":
            step_stop += 1
        elif topic == "/primitive/command_json":
            primitives += 1
        elif topic == "/metrics/event_jsonl":
            event_type = str(payload.get("event_type") or "")
            attribution = str(payload.get("attribution") or "")
            stale_action_executed += int(event_type == "stale_action_executed")
            timebase_error += int(event_type == "timebase_error" or attribution == "timebase_error")
            parse_error += int("parse_error" in event_type)

    expected = list(expected_metrics.get("episodes") or [])
    expected_ids = {str(row.get("episode_id") or "") for row in expected if row.get("episode_id")}
    expected_successes = sum(bool(row.get("success")) for row in expected)
    expected_route = sum(row.get("task_type") == "turn_choice" for row in expected)
    expected_stop = sum(row.get("task_type") == "semantic_target" for row in expected)
    required_topics = {"/benchmark/mode_json", "/isaac/episode_status", "/safe_cmd_vel", "/metrics/event_jsonl"}
    checks = {
        "records_present": sum(topic_counts.values()) > 0,
        "required_topics_present": required_topics.issubset(topic_counts),
        "expected_episode_ids_observed": bool(expected_ids) and expected_ids.issubset(observed_episodes),
        "success_count_matches": len(successful_episodes & expected_ids) == expected_successes,
        "route_role_count_matches": step_route == expected_route,
        "semantic_stop_count_matches": step_stop == expected_stop,
        "primitive_chain_observed": primitives > 0,
        "safe_linear_lte_0p20": safe_linear_max <= 0.2001,
        "safe_yaw_lte_0p30": safe_yaw_max <= 0.3001,
        "collision_zero": collisions == 0,
        "stale_action_executed_zero": stale_action_executed == 0,
        "timebase_error_zero": timebase_error == 0,
        "parse_error_zero": parse_error == 0,
    }
    return {
        "schema_version": 1,
        "pass": all(checks.values()),
        "replay_kind": "rosbag2_offline_message_replay",
        "checks": checks,
        "topic_counts": dict(sorted(topic_counts.items())),
        "expected_episode_count": len(expected_ids),
        "observed_episode_count": len(observed_episodes),
        "expected_success_count": expected_successes,
        "observed_success_count": len(successful_episodes & expected_ids),
        "step_route_count": step_route,
        "step_stop_count": step_stop,
        "primitive_count": primitives,
        "safe_linear_max_mps": round(safe_linear_max, 6),
        "safe_yaw_max_radps": round(safe_yaw_max, 6),
        "collision_count": collisions,
        "stale_action_executed": stale_action_executed,
        "timebase_error": timebase_error,
        "parse_error": parse_error,
        "real_robot_motion_enabled": False,
    }


def read_rosbag2(bag_dir: Path) -> list[tuple[str, Any]]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    type_map = {row.name: row.type for row in reader.get_all_topics_and_types()}
    rows: list[tuple[str, Any]] = []
    while reader.has_next():
        topic, raw, _timestamp = reader.read_next()
        rows.append((topic, deserialize_message(raw, get_message(type_map[topic]))))
    return rows


def write_bag_replay(output: Path, result: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "replay_gate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Qualification Bag Offline Replay",
        "",
        f"- pass: {result['pass']}",
        f"- kind: {result['replay_kind']}",
        f"- episodes expected/observed: {result['expected_episode_count']}/{result['observed_episode_count']}",
        f"- success expected/observed: {result['expected_success_count']}/{result['observed_success_count']}",
        f"- safe linear/yaw max: {result['safe_linear_max_mps']}/{result['safe_yaw_max_radps']}",
        "- real robot motion enabled: false",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_payload(value: Any) -> dict[str, Any]:
    raw = getattr(value, "data", value)
    if isinstance(raw, dict):
        return raw
    try:
        payload = json.loads(str(raw))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _twist_components(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        linear = value.get("linear") if isinstance(value.get("linear"), dict) else {}
        angular = value.get("angular") if isinstance(value.get("angular"), dict) else {}
        return float(linear.get("x", 0.0) or 0.0), float(angular.get("z", 0.0) or 0.0)
    return float(getattr(getattr(value, "linear", None), "x", 0.0)), float(getattr(getattr(value, "angular", None), "z", 0.0))
