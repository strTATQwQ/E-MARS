from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .config_loader import normalize_scene_to_robot_origin

COMPLETION_EVENTS = {
    "target_track_confirmed",
    "landmark_passed",
    "region_entered",
    "target_within_stop_distance",
    "completion_verified",
    "clarification_received",
}


def analyze_compositional_run(
    run_dir: str | Path,
    *,
    task_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    wanted = {str(value) for value in task_ids or []}
    episode_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for episode_dir in episode_dirs:
        task_path = episode_dir / "task.yaml"
        if not task_path.is_file():
            continue
        task = _load_document(task_path)
        task_id = str(task.get("task_id") or episode_dir.name)
        if wanted and task_id not in wanted:
            continue
        if str(task.get("category") or "") != "compositional":
            continue
        result = analyze_compositional_episode(episode_dir)
        rows.extend(result["subgoals"])
        episodes.append(result["episode"])

    failures = Counter(
        str(row["failure_attribution"])
        for row in rows
        if not bool(row["completed"])
    )
    return {
        "schema_version": 1,
        "run_dir": str(root.resolve()),
        "episodes": episodes,
        "subgoals": rows,
        "summary": {
            "episodes": len(episodes),
            "subgoals": len(rows),
            "completed_subgoals": sum(int(bool(row["completed"])) for row in rows),
            "failure_attribution": dict(failures),
            "failure_top1": failures.most_common(1)[0][0] if failures else "none",
            "qualification_evidence": False,
        },
    }


def analyze_compositional_episode(episode_dir: str | Path) -> dict[str, Any]:
    root = Path(episode_dir)
    task = _load_document(root / "task.yaml")
    scene = normalize_scene_to_robot_origin(_load_document(root / "scene.yaml"))
    metrics = _load_document(root / "metrics.json")
    trajectory = _read_trajectory(root / "trajectory.csv")
    events = _read_relevant_events(root / "events.jsonl")
    task_id = str(task.get("task_id") or root.name)
    plan = list(task.get("oracle_plan") or [])
    runtime = task.get("semantic_runtime") if isinstance(task.get("semantic_runtime"), dict) else {}
    object_ids = runtime.get("subgoal_object_ids") if isinstance(runtime.get("subgoal_object_ids"), dict) else {}
    objects = {str(obj.get("id")): obj for obj in scene.get("objects", [])}

    completion_by_index: dict[int, dict[str, Any]] = {}
    for event in events:
        details = event["details"]
        if str(details.get("type") or "") not in COMPLETION_EVENTS:
            continue
        index = _optional_int(details.get("index"))
        if index is not None:
            completion_by_index[index] = event

    first_t = trajectory[0]["t"] if trajectory else 0.0
    last_t = trajectory[-1]["t"] if trajectory else float(metrics.get("mission_time_sec", 0.0) or 0.0)
    rows: list[dict[str, Any]] = []
    previous_end: float | None = first_t
    for index, subgoal in enumerate(plan):
        start_t = previous_end
        completion = completion_by_index.get(index)
        end_t = float(completion["t"]) if completion else (last_t if start_t is not None else None)
        if index > 0 and index - 1 not in completion_by_index:
            start_t = None
            end_t = None
        window_trajectory = [] if start_t is None else _window(trajectory, start_t, end_t)
        window_events = [] if start_t is None else _window(events, start_t, end_t)
        object_id = str(object_ids.get(str(index)) or "")
        target = objects.get(object_id)
        target_pose = list((target or {}).get("pose") or [])
        route_origin = _route_origin(index, object_ids, objects, scene)
        geometry = _geometry_metrics(window_trajectory, route_origin, target_pose)
        action_events = [event for event in window_events if event["event"] == "omninav_action_candidate_json"]
        action_counts = Counter(str(event["details"].get("primitive") or "unknown") for event in action_events)
        action_total = sum(action_counts.values())
        turn_sequence = [
            str(event["details"].get("primitive") or "")
            for event in action_events
            if str(event["details"].get("primitive") or "") in {"turn_left", "turn_right"}
        ]
        no_progress = [
            event for event in window_events
            if str(event["details"].get("type") or "") == "no_progress_timeout"
        ]
        recoveries = [
            event for event in window_events
            if str(event["details"].get("type") or "") == "semantic_recovery_completed"
        ]
        requests = [event for event in window_events if event["event"] == "omninav_request_json"]
        stale = [
            event for event in window_events
            if str(event["details"].get("type") or "") == "stale_result"
        ]
        row = {
            "task_id": task_id,
            "episode_id": str(metrics.get("episode_id") or ""),
            "subgoal_index": index,
            "subgoal_type": str(subgoal.get("subgoal_type") or ""),
            "target": str(subgoal.get("target") or ""),
            "relation": str(subgoal.get("relation") or ""),
            "start_t": _rounded(start_t),
            "end_t": _rounded(end_t),
            "duration_sec": _rounded(None if start_t is None or end_t is None else end_t - start_t),
            "completed": completion is not None,
            "completion_event": "" if completion is None else str(completion["details"].get("type") or ""),
            "completion_method": "" if completion is None else str(completion["details"].get("completion_method") or "legacy"),
            "object_id": object_id,
            **geometry,
            "omninav_requests": len(requests),
            "omninav_actions": action_total,
            "forward_actions": action_counts["move_forward"],
            "left_actions": action_counts["turn_left"],
            "right_actions": action_counts["turn_right"],
            "stop_actions": action_counts["stop"],
            "forward_action_ratio": _ratio(action_counts["move_forward"], action_total),
            "turn_action_ratio": _ratio(action_counts["turn_left"] + action_counts["turn_right"], action_total),
            "stop_action_ratio": _ratio(action_counts["stop"], action_total),
            "turn_direction_switches": _direction_switches(turn_sequence),
            "no_progress_events": len(no_progress),
            "recovery_completed_events": len(recoveries),
            "stale_discards": len(stale),
        }
        row["failure_attribution"] = classify_subgoal_failure(row)
        rows.append(row)
        previous_end = float(completion["t"]) if completion else None

    return {
        "episode": {
            "task_id": task_id,
            "episode_id": str(metrics.get("episode_id") or ""),
            "success": bool(metrics.get("success", False)),
            "failure_reason": str(metrics.get("failure_reason") or ""),
            "path_length_m": float(metrics.get("path_length_m", 0.0) or 0.0),
            "subgoals_completed": int(metrics.get("semantic_subgoals_completed", 0) or 0),
            "subgoals_expected": int(metrics.get("semantic_subgoals_expected", len(plan)) or len(plan)),
        },
        "subgoals": rows,
    }


def classify_subgoal_failure(row: dict[str, Any]) -> str:
    if bool(row.get("completed")):
        return "completed"
    if row.get("start_t") is None:
        return "not_reached_due_to_prior_subgoal"
    subgoal_type = str(row.get("subgoal_type") or "")
    start_distance = _optional_float(row.get("start_distance_m"))
    min_distance = _optional_float(row.get("min_distance_m"))
    end_distance = _optional_float(row.get("end_distance_m"))
    if subgoal_type == "verify" and start_distance is not None and start_distance > 2.0:
        return "verify_without_approach_gap"
    if (
        float(row.get("turn_action_ratio", 0.0) or 0.0) >= 0.70
        and float(row.get("forward_action_ratio", 0.0) or 0.0) <= 0.10
        and int(row.get("turn_direction_switches", 0) or 0) >= 2
    ):
        return "turn_oscillation_no_progress"
    if subgoal_type in {"pass", "enter"}:
        near_limit = 1.65 if subgoal_type == "pass" else 1.50
        if min_distance is not None and min_distance <= near_limit:
            if end_distance is None or end_distance >= min_distance + 0.30:
                return "completion_geometry_miss_or_overshoot"
        if bool(row.get("crossed_completion_plane")) and float(row.get("min_cross_track_after_plane_m", 999.0) or 999.0) <= 1.50:
            return "completion_plane_not_judged"
    if int(row.get("recovery_completed_events", 0) or 0) > 0:
        return "recovery_did_not_restore_progress"
    if float(row.get("distance_decrease_step_ratio", 0.0) or 0.0) < 0.40:
        return "local_distance_progress_failure"
    return "subgoal_timeout_unknown"


def write_compositional_analysis(result: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "compositional_progress_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "subgoal_progress.csv", result["subgoals"])
    failure_rows = [row for row in result["subgoals"] if not bool(row["completed"])]
    _write_csv(output / "failure_attribution.csv", failure_rows)
    summary = result["summary"]
    lines = [
        "# V19 Compositional Progress Analysis",
        "",
        f"- episodes: {summary['episodes']}",
        f"- completed subgoals: {summary['completed_subgoals']}/{summary['subgoals']}",
        f"- failure_top1: `{summary['failure_top1']}`",
        "- controller truth input: none (offline judge/trace analysis only)",
        "- qualification_evidence: false",
        "",
        "## Failure Attribution",
        "",
    ]
    for name, count in sorted(summary["failure_attribution"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{name}`: {count}")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _geometry_metrics(
    trajectory: list[dict[str, float]],
    route_origin: list[float],
    target_pose: list[float],
) -> dict[str, Any]:
    empty = {
        "start_distance_m": None,
        "min_distance_m": None,
        "end_distance_m": None,
        "distance_reduction_m": None,
        "distance_decrease_step_ratio": 0.0,
        "crossed_completion_plane": False,
        "min_cross_track_after_plane_m": None,
        "max_route_progress_m": None,
        "heading_error_sign_switches": 0,
    }
    if not trajectory or len(target_pose) < 2 or len(route_origin) < 2:
        return empty
    tx, ty = float(target_pose[0]), float(target_pose[1])
    ox, oy = float(route_origin[0]), float(route_origin[1])
    route_dx, route_dy = tx - ox, ty - oy
    norm = math.hypot(route_dx, route_dy)
    if norm <= 1e-6:
        route_dx, route_dy, norm = 1.0, 0.0, 1.0
    ux, uy = route_dx / norm, route_dy / norm
    distances: list[float] = []
    after_plane_cross: list[float] = []
    progress_values: list[float] = []
    heading_signs: list[int] = []
    for point in trajectory:
        dx, dy = tx - point["x"], ty - point["y"]
        distances.append(math.hypot(dx, dy))
        from_target_x, from_target_y = point["x"] - tx, point["y"] - ty
        progress = from_target_x * ux + from_target_y * uy
        cross_track = abs(from_target_x * (-uy) + from_target_y * ux)
        progress_values.append(progress)
        if progress >= 0.0:
            after_plane_cross.append(cross_track)
        heading_error = _wrap(math.atan2(dy, dx) - point["yaw"])
        if abs(heading_error) >= math.radians(8.0):
            heading_signs.append(1 if heading_error > 0 else -1)
    decreasing = sum(1 for left, right in zip(distances, distances[1:]) if right <= left - 0.005)
    return {
        "start_distance_m": _rounded(distances[0]),
        "min_distance_m": _rounded(min(distances)),
        "end_distance_m": _rounded(distances[-1]),
        "distance_reduction_m": _rounded(distances[0] - min(distances)),
        "distance_decrease_step_ratio": _ratio(decreasing, max(1, len(distances) - 1)),
        "crossed_completion_plane": bool(after_plane_cross),
        "min_cross_track_after_plane_m": _rounded(min(after_plane_cross)) if after_plane_cross else None,
        "max_route_progress_m": _rounded(max(progress_values)),
        "heading_error_sign_switches": _direction_switches(heading_signs),
    }


def _route_origin(
    index: int,
    object_ids: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    scene: dict[str, Any],
) -> list[float]:
    current_id = str(object_ids.get(str(index)) or "")
    for previous in reversed(range(index)):
        previous_id = str(object_ids.get(str(previous)) or "")
        if previous_id and previous_id != current_id and previous_id in objects:
            return list(objects[previous_id].get("pose") or [0.0, 0.0])
    return list(scene.get("robot_start_pose") or [0.0, 0.0])


def _read_trajectory(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                "t": float(row.get("t", 0.0) or 0.0),
                "x": float(row.get("x", 0.0) or 0.0),
                "y": float(row.get("y", 0.0) or 0.0),
                "yaw": float(row.get("yaw", 0.0) or 0.0),
            }
            for row in csv.DictReader(handle)
        ]


def _read_relevant_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    keep = {
        "mission_event_json",
        "omninav_action_candidate_json",
        "omninav_request_json",
        "safe_cmd_vel",
    }
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if str(value.get("event") or "") in keep:
                events.append(
                    {
                        "t": float(value.get("t", 0.0) or 0.0),
                        "event": str(value.get("event") or ""),
                        "details": value.get("details") if isinstance(value.get("details"), dict) else {},
                    }
                )
    return events


def _window(rows: list[dict[str, Any]], start: float, end: float | None) -> list[dict[str, Any]]:
    upper = float("inf") if end is None else float(end)
    return [row for row in rows if float(start) <= float(row["t"]) <= upper]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        import yaml

        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _direction_switches(values: list[Any]) -> int:
    return sum(1 for left, right in zip(values, values[1:]) if left != right)


def _wrap(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _ratio(value: int | float, total: int | float) -> float:
    return float(value) / float(total) if total else 0.0


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
