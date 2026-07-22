from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .config_loader import load_data, object_by_id, scene_by_type
from .metrics import aggregate_metrics, bearing_deg, distance_xy, object_visible


ROUTE_COLUMNS = [
    "episode_id",
    "mode",
    "task_id",
    "task_type",
    "expected_branch",
    "near_intersection_detected",
    "near_intersection_time",
    "route_oracle_triggered",
    "route_oracle_json_published",
    "route_oracle_choice",
    "route_oracle_source",
    "route_stop_bridge_received_route",
    "turn_primitive_generated",
    "safe_cmd_vel_turn_nonzero",
    "yaw_delta_after_turn_deg",
    "entered_correct_branch",
    "failure_stage",
]

STOP_COLUMNS = [
    "episode_id",
    "mode",
    "task_id",
    "target_object",
    "target_visible_first_time",
    "distance_at_first_visible",
    "semantic_stop_oracle_triggered",
    "semantic_stop_json_published",
    "bridge_received_stop",
    "stop_primitive_generated",
    "safe_cmd_vel_zero_time",
    "robot_velocity_below_threshold_time",
    "visible_to_stop_latency",
    "distance_at_stop",
    "target_visible_at_stop",
    "success_judge_stop_condition",
    "success_judge_visible_condition",
    "success_judge_distance_condition",
    "semantic_target_success",
    "failure_stage",
]

STALE_CLASSES = [
    "reset_cleanup_discard",
    "old_response_after_reset",
    "episode_mismatch",
    "runtime_stale_discard",
    "stale_step_route",
    "stale_step_stop",
    "stale_omninav",
    "queue_delay",
    "robot_motion",
    "timestamp_mismatch",
    "timeout",
    "timebase_error",
    "unknown",
]

ALIAS_CANONICAL = {
    "fire_extinguisher": "fire_extinguisher",
    "fire_extinguisher_1": "fire_extinguisher",
    "fire_extinguisher_2": "fire_extinguisher",
    "fire_extinguisher_3": "fire_extinguisher",
    "fire_extinguisher_near_exit": "fire_extinguisher",
    "fire_extinguisher_verify": "fire_extinguisher",
    "red_exit_sign": "exit_sign",
    "exit_sign": "exit_sign",
    "exit_sign_1": "exit_sign",
    "exit_sign_2": "exit_sign",
    "exit_sign_3": "exit_sign",
    "exit_sign_4": "exit_sign",
    "exit_sign_verify": "exit_sign",
    "blue_box": "blue_box",
    "blue_box_1": "blue_box",
    "blue_box_2": "blue_box",
    "blue_box_3": "blue_box",
    "blue_box_right": "blue_box",
    "blue_box_verify": "blue_box",
    "red_cone": "red_cone",
    "red_cone_left": "red_cone",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def load_jsonl(path: Path, limit: int = 1_000_000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = headers or sorted({key for row in rows for key in row.keys()})
    if not headers:
        headers = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def postprocess_route_coverage_run(output: Path, *, deterministic: bool = False) -> dict[str, Any]:
    output = Path(output)
    metrics = load_json(output / "metrics.json")
    rows = route_coverage_rows(output, metrics)
    summary = evaluate_route_coverage(rows, deterministic=deterministic)
    stale = analyze_stale_attribution_v8(load_jsonl(output / "events.jsonl"))
    write_csv(output / "coverage_chain.csv", rows, ROUTE_COLUMNS)
    (output / "stale_discard_analysis.md").write_text(render_stale_report(stale), encoding="utf-8")
    metrics["v8_route_coverage"] = summary
    metrics["v8_stale_attribution"] = stale
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(render_route_summary(output, summary, stale, deterministic=deterministic), encoding="utf-8")
    return metrics


def postprocess_stop_coverage_run(output: Path) -> dict[str, Any]:
    output = Path(output)
    metrics = load_json(output / "metrics.json")
    rows = stop_coverage_rows(output, metrics)
    summary = evaluate_stop_coverage(rows)
    stale = analyze_stale_attribution_v8(load_jsonl(output / "events.jsonl"))
    write_csv(output / "stop_coverage_chain.csv", rows, STOP_COLUMNS)
    (output / "stale_discard_analysis.md").write_text(render_stale_report(stale), encoding="utf-8")
    metrics["v8_semantic_stop_coverage"] = summary
    metrics["v8_stale_attribution"] = stale
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(render_stop_summary(output, summary, stale), encoding="utf-8")
    return metrics


def route_coverage_rows(output: Path, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    aggregate_events = load_jsonl(output / "events.jsonl")
    rows: list[dict[str, Any]] = []
    for episode in metrics.get("episodes", []):
        if str(episode.get("task_type") or "") not in {"turn_choice", "turn_microbench"}:
            continue
        episode_dir = episode_dir_for(output, episode)
        task = load_yamlish(episode_dir / "task.yaml")
        scene = load_yamlish(episode_dir / "scene.yaml")
        events = load_jsonl(episode_dir / "events.jsonl") or events_for_episode(aggregate_events, episode)
        trajectory = load_trajectory(episode_dir / "trajectory.csv")
        rows.append(analyze_route_episode(episode, task, scene, events, trajectory))
    return rows


def stop_coverage_rows(output: Path, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    aggregate_events = load_jsonl(output / "events.jsonl")
    rows: list[dict[str, Any]] = []
    for episode in metrics.get("episodes", []):
        if str(episode.get("task_type") or "") != "semantic_target":
            continue
        episode_dir = episode_dir_for(output, episode)
        task = load_yamlish(episode_dir / "task.yaml")
        scene = load_yamlish(episode_dir / "scene.yaml")
        events = load_jsonl(episode_dir / "events.jsonl") or events_for_episode(aggregate_events, episode)
        trajectory = load_trajectory(episode_dir / "trajectory.csv")
        rows.append(analyze_stop_episode(episode, task, scene, events, trajectory))
    return rows


def analyze_route_episode(
    episode: dict[str, Any],
    task: dict[str, Any],
    scene: dict[str, Any],
    events: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = expected_branch_for_task(task, scene)
    near_time = first_event_time(
        events,
        lambda e, d, t: bool(d.get("near_intersection_detected") or d.get("near_intersection") or d.get("near_doorway")),
    )
    published_event = first_event(
        events,
        lambda e, d, t: t in {"oracle_route_choice_json", "route_oracle_json_published"}
        or (t == "route_oracle_coverage" and bool(d.get("route_oracle_json_published"))),
    )
    bridge_event = first_event(events, lambda e, d, t: t == "route_choice_bridge" and d.get("result") == "primitive_published")
    bridge_t = event_time(bridge_event)
    primitive_event = (
        first_event_after(
            events,
            bridge_t,
            lambda e, d, t: t == "primitive_command"
            and (
                str(d.get("action_type") or "") == "follow_waypoint"
                or str((d.get("primitive") or {}).get("primitive") if isinstance(d.get("primitive"), dict) else "") == "follow_waypoint"
            ),
        )
        if bridge_event is not None
        else None
    )
    primitive_t = event_time(primitive_event)
    safe_turn_event = (
        first_event_after(events, primitive_t, lambda e, d, t: t in {"safe_cmd_mux", "safe_cmd_vel"} and abs(cmd_angular_z(d)) > 0.02)
        if primitive_event is not None
        else None
    )
    route_choice = route_choice_from_event(published_event) or route_choice_from_event(bridge_event) or ""
    source = route_source_from_event(published_event) or route_source_from_event(bridge_event) or ""
    trigger_time = event_time(published_event) if published_event else event_time(bridge_event)
    yaw_delta = yaw_delta_deg_after(trajectory, trigger_time)
    entered = entered_branch(trajectory, expected)
    triggered = any(
        _event_type(event) == "route_oracle_coverage" and bool(_details(event).get("route_oracle_triggered"))
        for event in events
    ) or published_event is not None
    row = {
        "episode_id": episode.get("episode_id"),
        "mode": episode.get("mode"),
        "task_id": task.get("task_id") or episode.get("task_id"),
        "task_type": episode.get("task_type"),
        "expected_branch": expected,
        "near_intersection_detected": near_time is not None,
        "near_intersection_time": none_or_round(near_time),
        "route_oracle_triggered": triggered,
        "route_oracle_json_published": published_event is not None,
        "route_oracle_choice": route_choice,
        "route_oracle_source": source,
        "route_stop_bridge_received_route": bridge_event is not None,
        "turn_primitive_generated": primitive_event is not None,
        "safe_cmd_vel_turn_nonzero": safe_turn_event is not None,
        "yaw_delta_after_turn_deg": round(yaw_delta, 3),
        "entered_correct_branch": entered,
    }
    row["failure_stage"] = route_failure_stage(row)
    return row


def analyze_stop_episode(
    episode: dict[str, Any],
    task: dict[str, Any],
    scene: dict[str, Any],
    events: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
) -> dict[str, Any]:
    visible_event = first_event(events, lambda e, d, t: bool(d.get("target_visible")) and d.get("distance_to_target") is not None)
    trigger_event = first_event(
        events,
        lambda e, d, t: t in {"oracle_semantic_stop_json", "oracle_stop_json", "semantic_stop_json_published"}
        or (t == "semantic_stop_coverage" and bool(d.get("semantic_stop_oracle_triggered"))),
    )
    bridge_event = first_event(events, lambda e, d, t: t == "semantic_stop_bridge" and d.get("result") == "primitive_published")
    bridge_t = event_time(bridge_event)
    primitive_event = (
        first_event_after(
            events,
            bridge_t,
            lambda e, d, t: t == "primitive_command"
            and (
                str(d.get("action_type") or "") == "stop"
                or str((d.get("primitive") or {}).get("primitive") if isinstance(d.get("primitive"), dict) else "") == "stop"
            ),
        )
        if bridge_event is not None
        else None
    )
    primitive_t = event_time(primitive_event)
    safe_zero_event = (
        first_event_after(events, primitive_t, lambda e, d, t: t in {"safe_cmd_mux", "safe_cmd_vel"} and cmd_is_zero(d))
        if primitive_event is not None
        else None
    )
    final_status = last_event(events, lambda e, d, t: t == "isaac_episode_status" or e.get("event") == "success" or e.get("event") == "failure")
    first_visible_t = event_time(visible_event)
    trigger_t = event_time(trigger_event)
    safe_zero_t = event_time(safe_zero_event)
    distance_first = _float(_details(visible_event).get("distance_to_target"), None) if visible_event else None
    distance_stop = distance_at_time(events, safe_zero_t)
    visible_stop = visible_at_time(events, safe_zero_t)
    success_cfg = task.get("success", {}) if isinstance(task.get("success"), dict) else {}
    distance_limit = _float(success_cfg.get("distance_to_target_m"), 2.0)
    distance_condition = distance_stop is not None and distance_stop <= distance_limit
    visible_required = bool(success_cfg.get("target_visible", False))
    visible_condition = (not visible_required) or bool(visible_stop)
    stop_condition = safe_zero_t is not None
    semantic_success = bool(episode.get("success"))
    row = {
        "episode_id": episode.get("episode_id"),
        "mode": episode.get("mode"),
        "task_id": task.get("task_id") or episode.get("task_id"),
        "target_object": task.get("target_object"),
        "target_visible_first_time": none_or_round(first_visible_t),
        "distance_at_first_visible": none_or_round(distance_first),
        "semantic_stop_oracle_triggered": trigger_event is not None,
        "semantic_stop_json_published": trigger_event is not None,
        "bridge_received_stop": bridge_event is not None,
        "stop_primitive_generated": primitive_event is not None,
        "safe_cmd_vel_zero_time": none_or_round(safe_zero_t),
        "robot_velocity_below_threshold_time": none_or_round(safe_zero_t),
        "visible_to_stop_latency": none_or_round(None if first_visible_t is None or safe_zero_t is None else safe_zero_t - first_visible_t),
        "distance_at_stop": none_or_round(distance_stop),
        "target_visible_at_stop": bool(visible_stop),
        "success_judge_stop_condition": stop_condition,
        "success_judge_visible_condition": visible_condition,
        "success_judge_distance_condition": distance_condition,
        "semantic_target_success": semantic_success,
    }
    row["failure_stage"] = stop_failure_stage(row, final_status)
    return row


def evaluate_route_coverage(rows: list[dict[str, Any]], *, deterministic: bool = False) -> dict[str, Any]:
    episodes = len(rows)
    counts = count_true(rows, [
        "near_intersection_detected",
        "route_oracle_triggered",
        "route_oracle_json_published",
        "route_stop_bridge_received_route",
        "turn_primitive_generated",
        "safe_cmd_vel_turn_nonzero",
        "entered_correct_branch",
    ])
    failures: list[str] = []
    if episodes < 6:
        failures.append(f"turn episodes={episodes} < 6")
    for key in (
        "route_oracle_triggered",
        "route_stop_bridge_received_route",
        "turn_primitive_generated",
        "safe_cmd_vel_turn_nonzero",
    ):
        if counts[key] < min(6, episodes):
            failures.append(f"{key}={counts[key]} < {min(6, episodes)}")
    correct_min = 4 if episodes >= 6 else max(1, math.ceil(episodes * 0.5))
    if counts["entered_correct_branch"] < correct_min:
        failures.append(f"entered_correct_branch={counts['entered_correct_branch']} < {correct_min}")
    return {
        "pass": not failures,
        "deterministic": deterministic,
        "episodes": episodes,
        **counts,
        "correct_branch_rate": counts["entered_correct_branch"] / max(1, episodes),
        "failures": failures,
    }


def evaluate_stop_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = len(rows)
    counts = count_true(rows, [
        "target_visible_first_time",
        "semantic_stop_oracle_triggered",
        "bridge_received_stop",
        "stop_primitive_generated",
        "safe_cmd_vel_zero_time",
        "semantic_target_success",
    ])
    latencies = [_float(row.get("visible_to_stop_latency"), None) for row in rows if row.get("semantic_target_success")]
    latencies = [lat for lat in latencies if lat is not None]
    max_latency = max(latencies) if latencies else None
    failures: list[str] = []
    if episodes < 3:
        failures.append(f"semantic episodes={episodes} < 3")
    for key in ("target_visible_first_time", "semantic_stop_oracle_triggered", "bridge_received_stop", "safe_cmd_vel_zero_time"):
        if counts[key] < min(3, episodes):
            failures.append(f"{key}={counts[key]} < {min(3, episodes)}")
    if counts["semantic_target_success"] < 1:
        failures.append("semantic_target_success=0 < 1")
    if max_latency is None:
        failures.append("visible_to_stop_latency missing for success cases")
    elif max_latency > 2.0:
        failures.append(f"visible_to_stop_latency={max_latency:g} > 2")
    return {
        "pass": not failures,
        "episodes": episodes,
        **counts,
        "semantic_stop_accuracy": counts["semantic_target_success"] / max(1, episodes),
        "visible_to_stop_latency_max_sec": max_latency,
        "failures": failures,
    }


def analyze_stale_attribution_v8(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter({key: 0 for key in STALE_CLASSES})
    records: list[dict[str, Any]] = []
    for event in events:
        details = _details(event)
        if not looks_stale_or_discard(event):
            continue
        cls = classify_stale_event(event)
        counts[cls] += 1
        if is_reset_cleanup_stale(cls, event):
            counts["reset_cleanup_discard"] += 0 if cls == "reset_cleanup_discard" else 1
        else:
            counts["runtime_stale_discard"] += 1
        records.append(
            {
                "class": cls,
                "episode_id": event.get("episode_id") or details.get("episode_id"),
                "event_type": _event_type(event),
                "t": event.get("t"),
                "reason": details.get("reason") or details.get("attribution") or details.get("result"),
            }
        )
    total = len(records)
    return {
        "total_stale_discards": total,
        "reset_cleanup_discards": counts["reset_cleanup_discard"],
        "runtime_stale_discards": counts["runtime_stale_discard"],
        "old_response_after_reset": counts["old_response_after_reset"],
        "episode_mismatch": counts["episode_mismatch"],
        "timebase_error": counts["timebase_error"] + counts["timestamp_mismatch"],
        "counts": dict(counts),
        "records": records[:500],
    }


def is_reset_cleanup_stale(cls: str, event: dict[str, Any]) -> bool:
    if cls in {"old_response_after_reset", "reset_cleanup_discard"}:
        return True
    if cls == "episode_mismatch":
        timestamp = event_time(event)
        return timestamp is not None and timestamp <= 0.5
    return False


def classify_stale_event(event: dict[str, Any]) -> str:
    details = _details(event)
    event_type = _event_type(event).lower()
    text = json.dumps(event, ensure_ascii=False).lower()
    attr = str(details.get("attribution") or details.get("discard_reason") or details.get("reason") or "").lower()
    if "old_response_after_reset" in text or attr == "old_response_after_reset":
        return "old_response_after_reset"
    if "reset_cleanup" in text:
        return "reset_cleanup_discard"
    if "episode_mismatch" in text:
        return "episode_mismatch"
    if "timestamp_mismatch" in text:
        return "timestamp_mismatch"
    if "timebase_error" in text or "clock_domain_mismatch" in text:
        return "timebase_error"
    if "route_choice" in event_type or "step_route" in text:
        return "stale_step_route"
    if "semantic_stop" in event_type or "step_stop" in text:
        return "stale_step_stop"
    if "omninav" in text:
        return "stale_omninav"
    if "queue" in text:
        return "queue_delay"
    if "pose_delta" in text or "robot_motion" in text or "motion" in text:
        return "robot_motion"
    if "timeout" in text or "ttl" in text or "age" in text:
        return "timeout"
    return "unknown"


def looks_stale_or_discard(event: dict[str, Any]) -> bool:
    details = _details(event)
    event_type = _event_type(event).lower()
    result = str(details.get("result") or "").lower()
    if bool(details.get("discard") or details.get("stale")):
        return True
    if event_type.endswith("_stale") or "stale" in event_type:
        return True
    return result in {"discarded", "timeout", "expired", "stale", "dropped"}


def audit_target_success_judge(tasks_path: Path, scenes_path: Path, output: Path) -> dict[str, Any]:
    tasks_doc = load_data(tasks_path)
    scenes_doc = load_data(scenes_path)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for task in tasks_doc.get("tasks", []):
        target_id = str(task.get("target_object") or "")
        if not target_id:
            continue
        try:
            scene = scene_by_type(scenes_doc, task["scene_type"])
        except KeyError:
            failures.append(f"missing_scene_type:{task.get('scene_type')}")
            continue
        try:
            target = object_by_id(scene, target_id)
            found = True
        except KeyError:
            target = {}
            found = False
            failures.append(f"missing_target:{task.get('task_id')}:{target_id}")
        canonical_task = canonical_target_id(target_id)
        canonical_scene = canonical_target_id(str(target.get("id") or target_id))
        rows.append(
            {
                "task_id": task.get("task_id"),
                "task_type": task.get("task_type"),
                "scene_id": scene.get("scene_id"),
                "task_target_object": target_id,
                "isaac_object_id": target.get("id", ""),
                "semantic_oracle_id": target_id,
                "visible_object_id": target.get("id", ""),
                "success_judge_target_id": target.get("id", ""),
                "canonical_task_target": canonical_task,
                "canonical_scene_target": canonical_scene,
                "target_found": found,
                "alias_match": canonical_task == canonical_scene,
                "distance_source": "SuccessJudgeCore.distance_xy(robot_pose,target.pose)",
                "visible_source": "SuccessJudgeCore.object_visible(robot_pose,target,scene.obstacles)",
                "stop_required_source": "task.success.stop_required",
            }
        )
        if canonical_task != canonical_scene:
            failures.append(f"alias_mismatch:{task.get('task_id')}:{canonical_task}!={canonical_scene}")
    alias_rows = [{"alias": alias, "canonical": canonical_target_id(alias)} for alias in sorted(set(ALIAS_CANONICAL) | {
        "fire_extinguisher_near_exit",
        "red_exit_sign",
        "exit_sign",
        "blue_box",
    })]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "target_alias_table.csv", alias_rows, ["alias", "canonical"])
    write_csv(output / "target_success_judge_audit.csv", rows)
    result = {"pass": not failures, "failures": failures, "rows": len(rows)}
    (output / "target_success_judge_audit.md").write_text(render_target_audit(result, rows, alias_rows), encoding="utf-8")
    return result


def canonical_target_id(value: str) -> str:
    key = str(value or "").strip().lower()
    if key in ALIAS_CANONICAL:
        return ALIAS_CANONICAL[key]
    for suffix in ("_left", "_right", "_near", "_near_exit"):
        if key.endswith(suffix) and key[: -len(suffix)] in ALIAS_CANONICAL:
            return ALIAS_CANONICAL[key[: -len(suffix)]]
    if "_" in key:
        stem = key.rsplit("_", 1)[0]
        if stem in ALIAS_CANONICAL:
            return ALIAS_CANONICAL[stem]
    return key


def evaluate_sim2real_v8(metrics: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    required = gate.get("required", gate) if isinstance(gate.get("required"), dict) else gate
    route = metrics.get("v8_route_coverage", {}) if isinstance(metrics.get("v8_route_coverage"), dict) else {}
    stop = metrics.get("v8_semantic_stop_coverage", {}) if isinstance(metrics.get("v8_semantic_stop_coverage"), dict) else {}
    stale = metrics.get("v8_stale_attribution", {}) if isinstance(metrics.get("v8_stale_attribution"), dict) else {}
    events = metrics.get("_events") if isinstance(metrics.get("_events"), list) else []
    if not events and metrics.get("output_dir"):
        events = load_jsonl(Path(metrics.get("output_dir", "")) / "events.jsonl")
    max_linear = max_linear_x_from_events(events)
    parse_error = parse_error_count(events)
    aggregate = aggregate_metrics(metrics.get("episodes", [])) if isinstance(metrics.get("episodes"), list) else {}
    failures: list[str] = []

    def min_check(name: str, actual: float, expected: float) -> None:
        if actual < expected:
            failures.append(f"{name}={actual:g} < {expected:g}")

    def max_check(name: str, actual: float | None, expected: float) -> None:
        if actual is None:
            failures.append(f"{name}=missing/unverified")
        elif actual > expected:
            failures.append(f"{name}={actual:g} > {expected:g}")

    min_check("route_correct_branch_rate", float(route.get("correct_branch_rate", 0.0) or 0.0), float(required.get("route_correct_branch_rate_min", 0.60)))
    min_check("semantic_stop_accuracy", float(stop.get("semantic_stop_accuracy", 0.0) or 0.0), float(required.get("semantic_stop_accuracy_min", 0.75)))
    max_check("visible_to_stop_latency", stop.get("visible_to_stop_latency_max_sec"), float(required.get("visible_to_stop_latency_max_sec", 2.0)))
    if int(stale.get("runtime_stale_discards", 999) or 0) != int(required.get("runtime_stale_discards_required", 0)):
        failures.append(f"runtime_stale_discards={stale.get('runtime_stale_discards')} != {required.get('runtime_stale_discards_required', 0)}")
    if int(aggregate.get("collision_count", 0) or 0) != int(required.get("collision_count_required", 0)):
        failures.append(f"collision_count={aggregate.get('collision_count')} != {required.get('collision_count_required', 0)}")
    stale_action_executed = sum(1 for row in metrics.get("episodes", []) if row.get("failure_reason") == "stale_action_executed")
    if stale_action_executed != int(required.get("stale_action_executed_required", 0)):
        failures.append(f"stale_action_executed={stale_action_executed} != {required.get('stale_action_executed_required', 0)}")
    if int(stale.get("timebase_error", 0) or 0) != int(required.get("timebase_error_required", 0)):
        failures.append(f"timebase_error={stale.get('timebase_error')} != {required.get('timebase_error_required', 0)}")
    if parse_error != int(required.get("parse_error_required", 0)):
        failures.append(f"parse_error={parse_error} != {required.get('parse_error_required', 0)}")
    max_check("max_linear_x_mps", max_linear, float(required.get("max_linear_x_mps_real_gate", 0.20)))
    if not actions_through_safe_mux(events):
        failures.append("actions_through_safe_mux=false")
    return {
        "ready": not failures,
        "status": "READY FOR SENSOR-ONLY DRY-RUN" if not failures else "NOT READY FOR REAL ROBOT AUTONOMY",
        "failures": failures,
        "allowed_next_steps": [
            "sensor-only dry-run",
            "bag replay",
            "offline scoring",
            "stationary camera validation",
            "manual-triggered primitive test",
        ],
        "disallowed_next_steps": [
            "real robot autonomous navigation",
            "semantic target approach on real robot",
            "dynamic obstacle real test",
            "autonomous turn/route-choice real test",
        ],
    }


def render_sim2real_gate_v8(result: dict[str, Any], metrics: dict[str, Any]) -> str:
    lines = [
        f"# Sim2Real Readiness V8: {result['status']}",
        "",
        "OmniNav+Step value remains unproven.",
        "InternNav remains CmaAgent/system1/fallback_static_cma_tokens, not full InternVLA-N1.",
        "",
        "## V8 Summary",
        "",
        json.dumps(
            {
                "route": metrics.get("v8_route_coverage", {}),
                "semantic_stop": metrics.get("v8_semantic_stop_coverage", {}),
                "stale": metrics.get("v8_stale_attribution", {}),
            },
            indent=2,
            ensure_ascii=False,
        ),
        "",
        "## Failures",
    ]
    lines.extend([f"- {item}" for item in result["failures"]] or ["- none"])
    lines.extend(["", "## Allowed Only"])
    lines.extend([f"- {item}" for item in result["allowed_next_steps"]])
    lines.extend(["", "## Disallowed"])
    lines.extend([f"- {item}" for item in result["disallowed_next_steps"]])
    return "\n".join(lines) + "\n"


def render_route_summary(output: Path, summary: dict[str, Any], stale: dict[str, Any], *, deterministic: bool) -> str:
    title = "V8 Deterministic Route Injection" if deterministic else "V8 Route Oracle Coverage"
    lines = [
        f"# {title}",
        "",
        f"- run_dir: {output}",
        f"- pass: {summary.get('pass')}",
        f"- episodes: {summary.get('episodes')}",
        f"- route_oracle_triggered: {summary.get('route_oracle_triggered')}",
        f"- route_stop_bridge_received_route: {summary.get('route_stop_bridge_received_route')}",
        f"- turn_primitive_generated: {summary.get('turn_primitive_generated')}",
        f"- safe_cmd_vel_turn_nonzero: {summary.get('safe_cmd_vel_turn_nonzero')}",
        f"- entered_correct_branch: {summary.get('entered_correct_branch')}",
        f"- correct_branch_rate: {float(summary.get('correct_branch_rate', 0.0)):.3f}",
        f"- runtime_stale_discards: {stale.get('runtime_stale_discards')}",
        "",
        "Step experiments remain frozen until forced oracle full V8 passes.",
        "",
        "## Failures",
    ]
    lines.extend([f"- {item}" for item in summary.get("failures", [])] or ["- none"])
    lines.extend(["", "## Artifacts", "", f"- coverage_chain.csv: {output / 'coverage_chain.csv'}", f"- events.jsonl: {output / 'events.jsonl'}"])
    return "\n".join(lines) + "\n"


def render_stop_summary(output: Path, summary: dict[str, Any], stale: dict[str, Any]) -> str:
    lines = [
        "# V8 Semantic Stop Coverage",
        "",
        f"- run_dir: {output}",
        f"- pass: {summary.get('pass')}",
        f"- episodes: {summary.get('episodes')}",
        f"- semantic_stop_oracle_triggered: {summary.get('semantic_stop_oracle_triggered')}",
        f"- bridge_received_stop: {summary.get('bridge_received_stop')}",
        f"- safe_cmd_vel_zero_time: {summary.get('safe_cmd_vel_zero_time')}",
        f"- semantic_target_success: {summary.get('semantic_target_success')}",
        f"- semantic_stop_accuracy: {float(summary.get('semantic_stop_accuracy', 0.0)):.3f}",
        f"- visible_to_stop_latency_max_sec: {summary.get('visible_to_stop_latency_max_sec')}",
        f"- runtime_stale_discards: {stale.get('runtime_stale_discards')}",
        "",
        "Step experiments remain frozen until forced oracle full V8 passes.",
        "",
        "## Failures",
    ]
    lines.extend([f"- {item}" for item in summary.get("failures", [])] or ["- none"])
    lines.extend(["", "## Artifacts", "", f"- stop_coverage_chain.csv: {output / 'stop_coverage_chain.csv'}", f"- events.jsonl: {output / 'events.jsonl'}"])
    return "\n".join(lines) + "\n"


def render_stale_report(stale: dict[str, Any]) -> str:
    lines = [
        "# V8 Stale Discard Attribution",
        "",
        f"- total_stale_discards: {stale.get('total_stale_discards')}",
        f"- reset_cleanup_discards: {stale.get('reset_cleanup_discards')}",
        f"- runtime_stale_discards: {stale.get('runtime_stale_discards')}",
        f"- old_response_after_reset: {stale.get('old_response_after_reset')}",
        f"- episode_mismatch: {stale.get('episode_mismatch')}",
        f"- timebase_error: {stale.get('timebase_error')}",
        "",
        "| class | count |",
        "| --- | ---: |",
    ]
    counts = stale.get("counts", {}) if isinstance(stale.get("counts"), dict) else {}
    for key in STALE_CLASSES:
        lines.append(f"| {key} | {counts.get(key, 0)} |")
    return "\n".join(lines) + "\n"


def render_target_audit(result: dict[str, Any], rows: list[dict[str, Any]], alias_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# V8 Target Success Judge Audit",
        "",
        f"- pass: {result.get('pass')}",
        f"- audited_tasks: {len(rows)}",
        "",
        "## Required Alias Table",
        "",
        "| alias | canonical |",
        "| --- | --- |",
    ]
    for row in alias_rows:
        lines.append(f"| {row['alias']} | {row['canonical']} |")
    lines.extend(["", "## Failures", ""])
    lines.extend([f"- {item}" for item in result.get("failures", [])] or ["- none"])
    return "\n".join(lines) + "\n"


def episode_dir_for(output: Path, episode: dict[str, Any]) -> Path:
    delay = str(episode.get("delay_profile") or "")
    mode = str(episode.get("mode") or "")
    task_id = str(episode.get("task_id") or "")
    if delay and delay != "none" and (output / delay / mode / task_id).exists():
        return output / delay / mode / task_id
    return output / mode / task_id


def events_for_episode(events: list[dict[str, Any]], episode: dict[str, Any]) -> list[dict[str, Any]]:
    episode_id = str(episode.get("episode_id") or "")
    task_id = str(episode.get("task_id") or "")
    return [event for event in events if str(event.get("episode_id") or "") == episode_id or str(_details(event).get("task_id") or "") == task_id]


def load_yamlish(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = load_data(path)
    return value if isinstance(value, dict) else {}


def load_trajectory(path: Path) -> list[dict[str, Any]]:
    rows = load_csv(path)
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "t": _float(row.get("t"), 0.0),
                "x": _float(row.get("x"), 0.0),
                "y": _float(row.get("y"), 0.0),
                "yaw": _float(row.get("yaw"), 0.0),
                "source": row.get("source", ""),
            }
        )
    return result


def expected_branch_for_task(task: dict[str, Any], scene: dict[str, Any]) -> str:
    instruction = str(task.get("instruction") or "").lower()
    if "left" in instruction:
        return "left"
    if "right" in instruction:
        return "right"
    target_id = str(task.get("target_object") or "")
    try:
        target = object_by_id(scene, target_id)
        start = scene.get("robot_start_pose", [0.0, 0.0, 0.0])
        lateral = float(target.get("pose", [0.0, 0.0, 0.0])[1]) - float(start[1])
        if lateral > 0.35:
            return "left"
        if lateral < -0.35:
            return "right"
    except Exception:
        pass
    return "front"


def entered_branch(trajectory: list[dict[str, Any]], expected: str) -> bool:
    if not trajectory:
        return False
    final_y = max((row["y"] for row in trajectory), default=0.0) if expected == "left" else min((row["y"] for row in trajectory), default=0.0)
    if expected == "left":
        return final_y > 0.35
    if expected == "right":
        return final_y < -0.35
    return abs(float(trajectory[-1]["y"])) <= 0.45


def yaw_delta_deg_after(trajectory: list[dict[str, Any]], start_t: float | None, window_sec: float = 10.0) -> float:
    if len(trajectory) < 2:
        return 0.0
    if start_t is None:
        start = trajectory[0]
        end = trajectory[-1]
    else:
        start = nearest_pose(trajectory, start_t)
        end = nearest_pose(trajectory, min(float(start_t) + window_sec, float(trajectory[-1]["t"])))
    return abs(math.degrees(wrap_to_pi(float(end["yaw"]) - float(start["yaw"]))))


def nearest_pose(trajectory: list[dict[str, Any]], timestamp: float) -> dict[str, Any]:
    return min(trajectory, key=lambda row: abs(float(row.get("t", 0.0)) - timestamp))


def route_failure_stage(row: dict[str, Any]) -> str | None:
    checks = [
        ("near_intersection_detected", "near_intersection_not_detected"),
        ("route_oracle_triggered", "route_oracle_not_triggered"),
        ("route_oracle_json_published", "route_oracle_not_published"),
        ("route_stop_bridge_received_route", "route_stop_bridge_missing"),
        ("turn_primitive_generated", "turn_primitive_missing"),
        ("safe_cmd_vel_turn_nonzero", "safe_cmd_vel_turn_missing"),
    ]
    for key, stage in checks:
        if not bool(row.get(key)):
            return stage
    if _float(row.get("yaw_delta_after_turn_deg"), 0.0) < 5.0:
        return "isaac_state_no_yaw_change"
    if not bool(row.get("entered_correct_branch")):
        return "wrong_branch"
    return None


def stop_failure_stage(row: dict[str, Any], final_status: dict[str, Any] | None) -> str | None:
    checks = [
        ("target_visible_first_time", "target_never_visible"),
        ("semantic_stop_oracle_triggered", "semantic_stop_oracle_not_triggered"),
        ("semantic_stop_json_published", "semantic_stop_not_published"),
        ("bridge_received_stop", "semantic_stop_bridge_missing"),
        ("stop_primitive_generated", "stop_primitive_missing"),
        ("safe_cmd_vel_zero_time", "safe_cmd_vel_zero_missing"),
        ("success_judge_distance_condition", "distance_condition_failed"),
        ("success_judge_visible_condition", "visible_condition_failed"),
        ("success_judge_stop_condition", "stop_hold_failed"),
    ]
    for key, stage in checks:
        if not bool(row.get(key)):
            return stage
    if not bool(row.get("semantic_target_success")):
        details = _details(final_status or {})
        reason = details.get("reason")
        return f"success_judge_failed:{reason or 'unknown'}"
    return None


def first_event(events: list[dict[str, Any]], predicate: Any) -> dict[str, Any] | None:
    for event in sorted(events, key=event_time_sort_key):
        details = _details(event)
        if predicate(event, details, _event_type(event)):
            return event
    return None


def first_event_after(events: list[dict[str, Any]], after: float | None, predicate: Any) -> dict[str, Any] | None:
    for event in sorted(events, key=event_time_sort_key):
        if after is not None and event_time(event) is not None and float(event_time(event)) < after:
            continue
        details = _details(event)
        if predicate(event, details, _event_type(event)):
            return event
    return None


def last_event(events: list[dict[str, Any]], predicate: Any) -> dict[str, Any] | None:
    result = None
    for event in sorted(events, key=event_time_sort_key):
        details = _details(event)
        if predicate(event, details, _event_type(event)):
            result = event
    return result


def first_event_time(events: list[dict[str, Any]], predicate: Any) -> float | None:
    event = first_event(events, predicate)
    return event_time(event)


def event_time(event: dict[str, Any] | None) -> float | None:
    if not event:
        return None
    return _float(event.get("t"), None)


def event_time_sort_key(event: dict[str, Any]) -> float:
    return _float(event.get("t"), 0.0)


def route_choice_from_event(event: dict[str, Any] | None) -> str:
    if not event:
        return ""
    details = _details(event)
    for payload in (details, details.get("payload"), details.get("decision"), details.get("output"), details.get("primitive"), details.get("gate")):
        if isinstance(payload, dict) and payload.get("route_choice"):
            return str(payload.get("route_choice"))
    return str(details.get("route_oracle_choice") or "")


def route_source_from_event(event: dict[str, Any] | None) -> str:
    if not event:
        return ""
    details = _details(event)
    for payload in (details, details.get("payload"), details.get("decision"), details.get("output"), details.get("primitive"), details.get("gate")):
        if isinstance(payload, dict) and payload.get("source"):
            return str(payload.get("source"))
    return str(details.get("route_oracle_source") or "")


def cmd_angular_z(details: dict[str, Any]) -> float:
    cmd = details.get("cmd_vel") if isinstance(details.get("cmd_vel"), dict) else details
    angular = cmd.get("angular") if isinstance(cmd.get("angular"), dict) else {}
    return _float(angular.get("z"), 0.0)


def cmd_linear_x(details: dict[str, Any]) -> float:
    cmd = details.get("cmd_vel") if isinstance(details.get("cmd_vel"), dict) else details
    linear = cmd.get("linear") if isinstance(cmd.get("linear"), dict) else {}
    return _float(linear.get("x"), 0.0)


def cmd_is_zero(details: dict[str, Any]) -> bool:
    return abs(cmd_linear_x(details)) <= 0.05 and abs(cmd_angular_z(details)) <= 0.10


def distance_at_time(events: list[dict[str, Any]], timestamp: float | None) -> float | None:
    candidates = [event for event in events if _details(event).get("distance_to_target") is not None]
    if not candidates:
        return None
    if timestamp is None:
        event = candidates[-1]
    else:
        event = min(candidates, key=lambda e: abs(_float(e.get("t"), 0.0) - timestamp))
    return _float(_details(event).get("distance_to_target"), None)


def visible_at_time(events: list[dict[str, Any]], timestamp: float | None) -> bool:
    candidates = [event for event in events if "target_visible" in _details(event)]
    if not candidates:
        return False
    if timestamp is None:
        event = candidates[-1]
    else:
        event = min(candidates, key=lambda e: abs(_float(e.get("t"), 0.0) - timestamp))
    return bool(_details(event).get("target_visible"))


def count_true(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key in keys:
        result[key] = sum(1 for row in rows if truthy(row.get(key)))
    return result


def max_linear_x_from_events(events: list[dict[str, Any]]) -> float | None:
    best = None
    for event in events:
        details = _details(event)
        if _event_type(event) not in {"safe_cmd_vel", "cmd_vel_candidate", "safe_cmd_mux", "primitive_command"}:
            continue
        value = abs(cmd_linear_x(details))
        best = value if best is None else max(best, value)
    return best


def parse_error_count(events: list[dict[str, Any]]) -> int:
    return sum(1 for event in events if "parse_error" in _event_type(event))


def actions_through_safe_mux(events: list[dict[str, Any]]) -> bool:
    return any(_event_type(event) == "safe_cmd_mux" for event in events)


def _details(event: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(event, dict):
        return {}
    return event.get("details") if isinstance(event.get("details"), dict) else event


def _event_type(event: dict[str, Any] | None) -> str:
    details = _details(event)
    event_dict = event if isinstance(event, dict) else {}
    return str(details.get("event_type") or event_dict.get("event_type") or details.get("event") or event_dict.get("event") or "")


def _float(value: Any, default: Any = 0.0) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def none_or_round(value: Any, digits: int = 3) -> Any:
    number = _float(value, None)
    return None if number is None else round(float(number), digits)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
