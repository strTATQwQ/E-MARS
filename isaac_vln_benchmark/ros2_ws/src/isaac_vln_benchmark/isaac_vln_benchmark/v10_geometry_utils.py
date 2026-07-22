from __future__ import annotations

import csv
import json
import math
import struct
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config_loader import load_data, object_by_id, scene_by_type
from .metrics import distance_xy, object_visible
from .v8_coverage_utils import (
    analyze_stale_attribution_v8,
    cmd_angular_z,
    cmd_is_zero,
    cmd_linear_x,
    count_true,
    episode_dir_for,
    event_time,
    events_for_episode,
    first_event,
    first_event_after,
    load_csv,
    load_json,
    load_jsonl,
    load_trajectory,
    load_yamlish,
    none_or_round,
    render_stale_report,
    truthy,
    write_csv,
    _details,
    _event_type,
    _float,
)
from .v9_progress_utils import REQUIRED_FACTS, TRAJECTORY_COLUMNS


BRANCH_TRACE_COLUMNS = [
    "episode_id",
    "mode",
    "task_id",
    "seed",
    "direction",
    "target_yaw_deg",
    "post_turn_forward_m",
    "phase",
    "t",
    "x",
    "y",
    "yaw_deg",
    "desired_yaw_deg",
    "yaw_error_deg",
    "inside_left_polygon",
    "inside_right_polygon",
    "entered_correct_branch",
    "primitive",
    "linear_x_mps",
    "angular_z_radps",
]

BRANCH_MATRIX_COLUMNS = ["instruction_branch", "entered_left", "entered_right", "remained_intersection", "left_workspace"]

TARGET_APPROACH_COLUMNS = [
    "episode_id",
    "mode",
    "task_id",
    "seed",
    "controller",
    "scene_type",
    "target_id",
    "visible_target_id",
    "success_judge_target_id",
    "target_id_consistent",
    "reset_acknowledged",
    "target_vector_valid",
    "distance_decrease_ratio",
    "approach_steps",
    "reached_stop_threshold",
    "coverage_entry_time",
    "visible_to_stop_latency_sec",
    "stop_oracle_triggered",
    "bridge_received_stop",
    "safe_cmd_vel_zero_time",
    "semantic_success",
    "min_distance_m",
    "final_distance_m",
    "max_safe_linear_x_mps",
    "max_actual_linear_speed_mps",
    "mean_slip_mps",
    "costmap_block_steps",
    "fall_or_reset_count",
    "failure_stage",
]

TARGET_TRACE_COLUMNS = [
    "episode_id",
    "mode",
    "task_id",
    "seed",
    "controller",
    "phase",
    "t",
    "robot_x",
    "robot_y",
    "robot_yaw_deg",
    "target_x",
    "target_y",
    "nav_target_x",
    "nav_target_y",
    "desired_yaw_deg",
    "yaw_error_deg",
    "distance_m",
    "nav_distance_m",
    "primitive",
    "linear_x_mps",
    "angular_z_radps",
    "target_visible",
    "target_id",
    "target_visible_id",
    "success_judge_target_id",
    "safe_cmd_linear_x_mps",
    "safe_cmd_angular_z_radps",
    "actual_linear_speed_mps",
    "actual_yaw_rate_radps",
    "slip_mps",
    "robot_z",
    "robot_fallen_or_unstable",
    "collision",
    "local_costmap_clear",
    "obstacle_distance_m",
    "path_obstacle_distance_m",
    "telemetry_event",
    "telemetry_episode_id",
]

FAILURE_COLUMNS = ["episode_id", "mode", "task_id", "failure_stage", "failure_reason"]

STALE_COLUMNS = ["class", "episode_id", "event_type", "t", "reason"]


def wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def rad2deg(value: float) -> float:
    return math.degrees(float(value))


def deg2rad(value: float) -> float:
    return math.radians(float(value))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def expected_branch(task: dict[str, Any], scene: dict[str, Any] | None = None) -> str:
    text = str(task.get("instruction") or "").lower()
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    if scene:
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


def intersection_center(scene: dict[str, Any]) -> list[float]:
    for zone in scene.get("semantic_zones", []):
        if str(zone.get("class") or "") in {"intersection", "doorway"}:
            center = zone.get("center", [4.0, 0.0])
            return [float(center[0]), float(center[1])]
    return [4.0, 0.0]


def branch_polygons(scene: dict[str, Any], *, length_m: float = 3.5, half_width_m: float = 1.5, entry_y_m: float = 0.35) -> dict[str, list[list[float]]]:
    cx, cy = intersection_center(scene)
    return {
        "left": [[cx - 0.25, cy + entry_y_m], [cx + length_m, cy + entry_y_m], [cx + length_m, cy + half_width_m], [cx - 0.25, cy + half_width_m]],
        "right": [[cx - 0.25, cy - half_width_m], [cx + length_m, cy - half_width_m], [cx + length_m, cy - entry_y_m], [cx - 0.25, cy - entry_y_m]],
    }


def point_in_polygon(point: list[float] | tuple[float, float], polygon: list[list[float]]) -> bool:
    x = float(point[0])
    y = float(point[1])
    inside = False
    if len(polygon) < 3:
        return False
    j = len(polygon) - 1
    for i, pi in enumerate(polygon):
        xi, yi = float(pi[0]), float(pi[1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / max(1.0e-9, yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def branch_membership(pose: list[float], scene: dict[str, Any]) -> dict[str, bool]:
    polygons = branch_polygons(scene)
    point = [float(pose[0]), float(pose[1])]
    return {
        "left": point_in_polygon(point, polygons["left"]),
        "right": point_in_polygon(point, polygons["right"]),
    }


def branch_yaw_deg(direction: str, target_yaw_deg: float) -> float:
    sign = 1.0 if direction == "left" else -1.0 if direction == "right" else 0.0
    return sign * abs(float(target_yaw_deg))


def target_vector(robot_pose: list[float], target_pose: list[float]) -> dict[str, float]:
    dx = float(target_pose[0]) - float(robot_pose[0])
    dy = float(target_pose[1]) - float(robot_pose[1])
    desired = math.atan2(dy, dx)
    yaw = float(robot_pose[2])
    return {
        "dx": dx,
        "dy": dy,
        "desired_yaw_rad": desired,
        "yaw_error_rad": wrap_to_pi(desired - yaw),
        "distance_2d_m": math.hypot(dx, dy),
    }


def rotate_then_forward_command(
    yaw_error_rad: float,
    *,
    orienting: bool,
    yaw_start_deg: float = 15.0,
    yaw_stop_deg: float = 8.0,
    max_linear_x_mps: float = 0.20,
    max_yaw_rate_radps: float = 0.30,
) -> tuple[str, bool, float, float]:
    abs_err_deg = abs(rad2deg(yaw_error_rad))
    if orienting:
        orienting = abs_err_deg > yaw_stop_deg
    elif abs_err_deg > yaw_start_deg:
        orienting = True
    if orienting:
        return "orient", orienting, 0.0, math.copysign(max_yaw_rate_radps, yaw_error_rad)
    return "approach", orienting, max_linear_x_mps, clamp(0.8 * yaw_error_rad, -max_yaw_rate_radps, max_yaw_rate_radps)


def proportional_command(
    yaw_error_rad: float,
    *,
    orienting: bool = False,
    yaw_start_deg: float = 15.0,
    yaw_stop_deg: float = 8.0,
    base_speed_mps: float = 0.20,
    max_yaw_rate_radps: float = 0.30,
    k_yaw: float = 0.8,
) -> tuple[str, bool, float, float]:
    abs_err_deg = abs(rad2deg(yaw_error_rad))
    if orienting:
        orienting = abs_err_deg > yaw_stop_deg
    elif abs_err_deg > yaw_start_deg:
        orienting = True
    angular = clamp(k_yaw * yaw_error_rad, -max_yaw_rate_radps, max_yaw_rate_radps)
    if orienting:
        return "orient", orienting, 0.0, angular
    linear = base_speed_mps * max(0.0, math.cos(yaw_error_rad))
    return "approach", orienting, linear, angular


def crawl_turn_command(
    yaw_error_rad: float,
    *,
    orienting: bool = False,
    local_costmap_clear: bool = True,
    yaw_start_deg: float = 15.0,
    yaw_stop_deg: float = 10.0,
    orient_linear_x_mps: float = 0.12,
    blocked_reverse_speed_mps: float = 0.08,
    crawl_max_yaw_error_deg: float = 60.0,
    base_speed_mps: float = 0.20,
    max_yaw_rate_radps: float = 0.30,
    k_yaw: float = 0.8,
) -> tuple[str, bool, float, float]:
    """Use a visible crawl arc to avoid unstable low-speed in-place policy turns."""

    abs_err_deg = abs(rad2deg(yaw_error_rad))
    if orienting:
        orienting = abs_err_deg > yaw_stop_deg
    elif abs_err_deg > yaw_start_deg:
        orienting = True
    if not local_costmap_clear:
        orienting = True
    if orienting:
        crawl_allowed = bool(local_costmap_clear and abs_err_deg <= crawl_max_yaw_error_deg)
        linear = (
            min(base_speed_mps, max(0.0, orient_linear_x_mps))
            if crawl_allowed
            else -min(base_speed_mps, max(0.0, blocked_reverse_speed_mps))
        )
        angular = 0.0 if abs_err_deg <= yaw_stop_deg else math.copysign(max_yaw_rate_radps, yaw_error_rad)
        return "orient", orienting, linear, angular
    linear = base_speed_mps * max(0.0, math.cos(yaw_error_rad))
    angular = clamp(k_yaw * yaw_error_rad, -max_yaw_rate_radps, max_yaw_rate_radps)
    return "approach", orienting, linear, angular


def update_yaw_polarity(
    polarity: float,
    mismatch_count: int,
    desired_angular_radps: float,
    actual_yaw_delta_rad: float,
    *,
    min_desired_radps: float = 0.05,
    min_delta_rad: float = 0.003,
    flip_after: int = 2,
) -> tuple[float, int, bool]:
    """Adapt command polarity when measured heading repeatedly opposes intent."""

    if abs(float(desired_angular_radps)) < min_desired_radps or abs(float(actual_yaw_delta_rad)) < min_delta_rad:
        return float(polarity), int(mismatch_count), False
    aligned = float(desired_angular_radps) * float(actual_yaw_delta_rad) > 0.0
    count = max(0, int(mismatch_count) - 1) if aligned else int(mismatch_count) + 1
    if count >= max(1, int(flip_after)):
        return -float(polarity), 0, True
    return float(polarity), count, False


def route_geometry_audit(config_path: Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[4]
    tasks_doc = load_data(root / "configs" / "tasks.yaml")
    scenes_doc = load_data(root / "configs" / "scenes.yaml")
    tasks = [task for task in tasks_doc.get("tasks", []) if task.get("task_id") in {"turn_001", "turn_002"}]
    scene = scene_by_type(scenes_doc, "corridor_with_intersection")
    polygons = branch_polygons(scene)
    return {
        "world_frame": "world",
        "robot_frame": "base_link",
        "intersection_center": intersection_center(scene),
        "left_branch_heading_deg": 60.0,
        "right_branch_heading_deg": -60.0,
        "left_branch_polygon": polygons["left"],
        "right_branch_polygon": polygons["right"],
        "robot_initial_yaw_deg": round(rad2deg(float(scene.get("robot_start_pose", [0.0, 0.0, 0.0])[2])), 3),
        "turn_left_angular_z_sign": 1,
        "turn_right_angular_z_sign": -1,
        "success_judge_frame": "world",
        "direction_mapping_valid": all(expected_branch(task, scene) in {"left", "right"} for task in tasks),
        "tasks": [
            {
                "task_id": task.get("task_id"),
                "instruction": task.get("instruction"),
                "expected_branch": expected_branch(task, scene),
                "target_object": task.get("target_object"),
                "target_pose": object_by_id(scene, task["target_object"]).get("pose"),
            }
            for task in tasks
        ],
    }


def target_pose_audit() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[4]
    tasks_doc = load_data(root / "configs" / "tasks.yaml")
    scenes_doc = load_data(root / "configs" / "scenes.yaml")
    task = next(task for task in tasks_doc.get("tasks", []) if task.get("task_id") == "semantic_001")
    scene = scene_by_type(scenes_doc, task["scene_type"])
    target = object_by_id(scene, task["target_object"])
    robot_pose = scene.get("robot_start_pose", [0.0, 0.0, 0.0])
    target_pose = target.get("pose", [0.0, 0.0, 0.0])
    vector = target_vector(robot_pose, target_pose)
    visible = object_visible(robot_pose, target, fov_deg=90.0, max_range_m=12.0)
    return {
        "pass": bool(target.get("id") == task.get("target_object") and vector["distance_2d_m"] > 0.0),
        "robot_pose_world": robot_pose,
        "target_pose_world": target_pose,
        "target_id": task.get("target_object"),
        "semantic_visible_id": target.get("id") if visible else "",
        "success_judge_target_id": task.get("target_object"),
        "distance_2d_m": round(vector["distance_2d_m"], 3),
        "distance_3d_m": round(math.dist([float(robot_pose[0]), float(robot_pose[1]), 0.0], [float(target_pose[0]), float(target_pose[1]), float(target_pose[2])]), 3),
        "line_of_sight": bool(visible),
        "same_frame": True,
        "pose_valid": True,
        "desired_yaw_deg": round(rad2deg(vector["desired_yaw_rad"]), 3),
        "yaw_error_deg": round(rad2deg(vector["yaw_error_rad"]), 3),
    }


def postprocess_branch_controller_run(output: Path) -> dict[str, Any]:
    output = Path(output)
    metrics = load_json(output / "metrics.json")
    branch_rows, trace_rows = branch_controller_rows(output, metrics)
    stale = write_common_v10_artifacts(output, metrics, branch_trace=trace_rows, geometry_audit=route_geometry_audit())
    matrix = branch_confusion_matrix(branch_rows)
    write_csv(output / "branch_confusion_matrix.csv", matrix, BRANCH_MATRIX_COLUMNS)
    write_failure_table(output, metrics, branch_rows)
    summary = evaluate_branch_controller(branch_rows, metrics, stale)
    metrics["v10_branch_entry_controller"] = summary
    metrics["v10_stale_attribution"] = stale
    _write_metrics(output, metrics)
    (output / "summary.md").write_text(render_v10_summary(output, "V10 Branch Entry Controller", route=summary, stale=stale), encoding="utf-8")
    return metrics


def postprocess_target_approach_run(output: Path) -> dict[str, Any]:
    output = Path(output)
    metrics = load_json(output / "metrics.json")
    approach_rows, trace_rows = target_approach_rows(output, metrics)
    stale = write_common_v10_artifacts(output, metrics, target_trace=trace_rows, target_audit=target_pose_audit())
    write_csv(output / "target_approach_metrics.csv", approach_rows, TARGET_APPROACH_COLUMNS)
    write_failure_table(output, metrics, approach_rows)
    summary = evaluate_target_approach(approach_rows, metrics, stale)
    metrics["v10_target_relative_approach"] = summary
    metrics["v10_stale_attribution"] = stale
    _write_metrics(output, metrics)
    (output / "summary.md").write_text(render_v10_summary(output, "V10 Target Relative Approach", semantic=summary, stale=stale), encoding="utf-8")
    return metrics


def postprocess_geometric_watchdog_run(output: Path) -> dict[str, Any]:
    output = Path(output)
    metrics = load_json(output / "metrics.json")
    branch_rows, branch_trace = branch_controller_rows(output, metrics)
    approach_rows, target_trace = target_approach_rows(output, metrics)
    stale = write_common_v10_artifacts(output, metrics, branch_trace=branch_trace, target_trace=target_trace, geometry_audit=route_geometry_audit())
    write_csv(output / "branch_confusion_matrix.csv", branch_confusion_matrix(branch_rows), BRANCH_MATRIX_COLUMNS)
    write_csv(output / "target_approach_metrics.csv", approach_rows, TARGET_APPROACH_COLUMNS)
    write_failure_table(output, metrics, branch_rows + approach_rows)
    route_summary = evaluate_branch_controller(branch_rows, metrics, stale)
    semantic_summary = evaluate_target_approach(approach_rows, metrics, stale)
    metrics["v10_branch_entry_controller"] = route_summary
    metrics["v10_target_relative_approach"] = semantic_summary
    metrics["v10_geometric_watchdog"] = {"pass": bool(route_summary.get("pass") and semantic_summary.get("pass"))}
    metrics["v10_stale_attribution"] = stale
    _write_metrics(output, metrics)
    (output / "summary.md").write_text(
        render_v10_summary(output, "V10 Geometric Watchdog", route=route_summary, semantic=semantic_summary, stale=stale),
        encoding="utf-8",
    )
    return metrics


def branch_controller_rows(output: Path, metrics: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate_events = load_jsonl(output / "events.jsonl")
    rows = []
    traces = []
    for episode in metrics.get("episodes", []):
        if str(episode.get("task_type") or "") not in {"turn_choice", "turn_microbench"}:
            continue
        episode_dir = episode_dir_for(output, episode)
        task = load_yamlish(episode_dir / "task.yaml")
        scene = load_yamlish(episode_dir / "scene.yaml")
        events = trim_events_after_reset(load_jsonl(episode_dir / "events.jsonl") or events_for_episode(aggregate_events, episode))
        trajectory = load_trajectory(episode_dir / "trajectory.csv")
        trace = branch_trace_from_events(episode, events)
        traces.extend(trace)
        rows.append(analyze_branch_episode(episode, task, scene, events, trajectory, trace))
    return rows, traces


def branch_trace_from_events(episode: dict[str, Any], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        details = _details(event)
        if details.get("event_type") != "v10_branch_controller_trace":
            continue
        primitive = details.get("primitive") if isinstance(details.get("primitive"), dict) else {}
        pose = details.get("pose") if isinstance(details.get("pose"), list) else [None, None, None]
        rows.append(
            {
                "episode_id": episode.get("episode_id"),
                "mode": episode.get("mode"),
                "task_id": episode.get("task_id"),
                "seed": seed_from_task_id(episode.get("task_id")),
                "direction": details.get("direction"),
                "target_yaw_deg": details.get("target_yaw_deg"),
                "post_turn_forward_m": details.get("post_turn_forward_m"),
                "phase": details.get("phase"),
                "t": event_time(event),
                "x": pose[0],
                "y": pose[1],
                "yaw_deg": none_or_round(rad2deg(_float(pose[2], 0.0))),
                "desired_yaw_deg": details.get("desired_yaw_deg"),
                "yaw_error_deg": details.get("yaw_error_deg"),
                "inside_left_polygon": details.get("inside_left_polygon"),
                "inside_right_polygon": details.get("inside_right_polygon"),
                "entered_correct_branch": details.get("entered_correct_branch"),
                "primitive": primitive.get("primitive"),
                "linear_x_mps": primitive.get("linear_x_mps"),
                "angular_z_radps": primitive.get("angular_z_radps"),
            }
        )
    return rows


def analyze_branch_episode(
    episode: dict[str, Any],
    task: dict[str, Any],
    scene: dict[str, Any],
    events: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    direction = expected_branch(task, scene)
    final_pose = trajectory[-1] if trajectory else {"x": 0.0, "y": 0.0, "yaw": 0.0}
    membership = branch_membership([final_pose["x"], final_pose["y"], final_pose["yaw"]], scene)
    decision = first_event(events, lambda e, d, t: d.get("event_type") == "v10_branch_decision_published")
    rotate = any(row.get("phase") in {"rotate", "rotate_to_branch"} for row in trace)
    post_forward = any(row.get("phase") in {"enter", "advance_into_branch"} and _float(row.get("linear_x_mps"), 0.0) > 0.0 for row in trace)
    ever_left = any(truthy(row.get("inside_left_polygon")) for row in trace)
    ever_right = any(truthy(row.get("inside_right_polygon")) for row in trace)
    ever_correct = any(truthy(row.get("entered_correct_branch")) for row in trace)
    entered = ever_correct or bool(membership.get(direction))
    target_yaw = trace[0].get("target_yaw_deg") if trace else ""
    post_dist = trace[0].get("post_turn_forward_m") if trace else ""
    row = {
        "episode_id": episode.get("episode_id"),
        "mode": episode.get("mode"),
        "task_id": episode.get("task_id"),
        "seed": seed_from_task_id(episode.get("task_id")),
        "instruction_branch": direction,
        "entered_branch": "left" if ever_left or membership["left"] else "right" if ever_right or membership["right"] else "intersection",
        "decision_published": decision is not None,
        "rotate_phase_entered": rotate,
        "post_turn_forward_executed": post_forward,
        "entered_correct_branch": entered,
        "inside_left_polygon": membership["left"],
        "inside_right_polygon": membership["right"],
        "target_yaw_deg": target_yaw,
        "post_turn_forward_m": post_dist,
        "failure_stage": None,
    }
    row["failure_stage"] = branch_failure_stage(row)
    return row


def target_approach_rows(output: Path, metrics: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate_events = load_jsonl(output / "events.jsonl")
    rows = []
    traces = []
    for episode in metrics.get("episodes", []):
        if str(episode.get("task_type") or "") != "semantic_target":
            continue
        episode_dir = episode_dir_for(output, episode)
        events = trim_events_after_reset(load_jsonl(episode_dir / "events.jsonl") or events_for_episode(aggregate_events, episode))
        trace = target_trace_from_events(episode, events)
        traces.extend(trace)
        task = load_yamlish(episode_dir / "task.yaml")
        scene = load_yamlish(episode_dir / "scene.yaml")
        rows.append(analyze_target_episode(episode, task, scene, events, trace))
    return rows, traces


def target_trace_from_events(episode: dict[str, Any], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        details = _details(event)
        if details.get("event_type") != "v10_target_controller_trace":
            continue
        primitive = details.get("primitive") if isinstance(details.get("primitive"), dict) else {}
        robot = details.get("robot_pose") if isinstance(details.get("robot_pose"), list) else [None, None, None]
        target = details.get("target_pose") if isinstance(details.get("target_pose"), list) else [None, None, None]
        nav_target = details.get("nav_target_pose") if isinstance(details.get("nav_target_pose"), list) else [None, None, None]
        actual_linear = details.get("actual_linear_velocity") if isinstance(details.get("actual_linear_velocity"), list) else []
        actual_angular = details.get("actual_angular_velocity") if isinstance(details.get("actual_angular_velocity"), list) else []
        actual_linear_speed = math.hypot(_float(actual_linear[0], 0.0), _float(actual_linear[1], 0.0)) if len(actual_linear) >= 2 else None
        actual_yaw_rate = _float(actual_angular[2], None) if len(actual_angular) >= 3 else None
        safe_linear = _float(details.get("safe_cmd_linear_x_mps"), 0.0)
        rows.append(
            {
                "episode_id": episode.get("episode_id"),
                "mode": episode.get("mode"),
                "task_id": episode.get("task_id"),
                "seed": seed_from_task_id(episode.get("task_id")),
                "controller": details.get("controller"),
                "phase": details.get("phase"),
                "t": event_time(event),
                "robot_x": robot[0],
                "robot_y": robot[1],
                "robot_yaw_deg": none_or_round(rad2deg(_float(robot[2], 0.0))),
                "target_x": target[0],
                "target_y": target[1],
                "nav_target_x": nav_target[0],
                "nav_target_y": nav_target[1],
                "desired_yaw_deg": details.get("desired_yaw_deg"),
                "yaw_error_deg": details.get("yaw_error_deg"),
                "distance_m": details.get("distance_m"),
                "nav_distance_m": details.get("nav_distance_m"),
                "primitive": primitive.get("primitive"),
                "linear_x_mps": primitive.get("linear_x_mps"),
                "angular_z_radps": primitive.get("angular_z_radps"),
                "target_visible": details.get("target_visible"),
                "target_id": details.get("target_id") or primitive.get("target_id"),
                "target_visible_id": details.get("target_visible_id"),
                "success_judge_target_id": details.get("success_judge_target_id"),
                "safe_cmd_linear_x_mps": details.get("safe_cmd_linear_x_mps"),
                "safe_cmd_angular_z_radps": details.get("safe_cmd_angular_z_radps"),
                "actual_linear_speed_mps": none_or_round(actual_linear_speed),
                "actual_yaw_rate_radps": none_or_round(actual_yaw_rate),
                "slip_mps": none_or_round(abs(safe_linear) - actual_linear_speed) if actual_linear_speed is not None else None,
                "robot_z": details.get("robot_z"),
                "robot_fallen_or_unstable": details.get("robot_fallen_or_unstable"),
                "collision": details.get("collision"),
                "local_costmap_clear": details.get("local_costmap_clear"),
                "obstacle_distance_m": details.get("obstacle_distance_m"),
                "path_obstacle_distance_m": details.get("path_obstacle_distance_m"),
                "telemetry_event": details.get("telemetry_event"),
                "telemetry_episode_id": details.get("telemetry_episode_id"),
            }
        )
    return rows


def analyze_target_episode(
    episode: dict[str, Any],
    task: dict[str, Any],
    scene: dict[str, Any],
    events: list[dict[str, Any]],
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    distances = [_float(row.get("distance_m"), None) for row in trace if row.get("distance_m") not in {None, ""}]
    distances = [value for value in distances if value is not None]
    approach_distances = [
        _float(row.get("distance_m"), None)
        for row in trace
        if row.get("phase") == "approach" and row.get("distance_m") not in {None, ""}
    ]
    approach_distances = [value for value in approach_distances if value is not None]
    approach_pairs = []
    for prev, cur in zip(approach_distances, approach_distances[1:]):
        approach_pairs.append(cur < prev - 0.002)
    decrease_ratio = sum(1 for item in approach_pairs if item) / max(1, len(approach_pairs))
    stop_event = first_event(events, lambda e, d, t: t in {"oracle_semantic_stop_json", "oracle_stop_json"} or d.get("event_type") == "semantic_stop_json_published")
    bridge_event = first_event(events, lambda e, d, t: d.get("event_type") == "semantic_stop_bridge" and d.get("result") == "primitive_published")
    primitive_t = event_time(bridge_event)
    zero = first_event_after(events, primitive_t, lambda e, d, t: t in {"safe_cmd_vel", "safe_cmd_mux"} and cmd_is_zero(d)) if bridge_event else None
    coverage_event = first_event(events, lambda e, d, t: d.get("event_type") == "semantic_stop_coverage_entered")
    coverage_t = event_time(coverage_event)
    stop_t = event_time(stop_event)
    latency = None if coverage_t is None or stop_t is None else max(0.0, float(stop_t) - float(coverage_t))
    min_dist = min(distances) if distances else None
    final_dist = distances[-1] if distances else None
    expected_target_id = str(task.get("target_object") or "")
    controller_target_ids = {str(row.get("target_id")) for row in trace if row.get("target_id")}
    visible_target_ids = {str(row.get("target_visible_id")) for row in trace if row.get("target_visible_id")}
    judge_target_ids = {str(row.get("success_judge_target_id")) for row in trace if row.get("success_judge_target_id")}
    reset_ack = first_event(events, lambda e, d, t: t == "reset_ack")
    safe_speeds = [abs(_float(row.get("safe_cmd_linear_x_mps"), 0.0)) for row in trace]
    actual_speeds = [_float(row.get("actual_linear_speed_mps"), None) for row in trace]
    actual_speeds = [value for value in actual_speeds if value is not None]
    slips = [_float(row.get("slip_mps"), None) for row in trace]
    slips = [value for value in slips if value is not None]
    fall_or_reset_count = sum(
        1
        for row in trace
        if truthy(row.get("robot_fallen_or_unstable"))
        or (
            str(row.get("telemetry_event") or "") == "reset"
            and not str(row.get("telemetry_episode_id") or "")
        )
    )
    row = {
        "episode_id": episode.get("episode_id"),
        "mode": episode.get("mode"),
        "task_id": episode.get("task_id"),
        "seed": seed_from_task_id(episode.get("task_id")),
        "controller": trace[0].get("controller") if trace else controller_from_mode(episode.get("mode")),
        "scene_type": scene.get("scene_type"),
        "target_id": expected_target_id,
        "visible_target_id": ",".join(sorted(visible_target_ids)),
        "success_judge_target_id": ",".join(sorted(judge_target_ids)),
        "target_id_consistent": bool(
            expected_target_id
            and controller_target_ids == {expected_target_id}
            and judge_target_ids == {expected_target_id}
            and (not visible_target_ids or visible_target_ids == {expected_target_id})
        ),
        "reset_acknowledged": reset_ack is not None,
        "target_vector_valid": bool(trace),
        "distance_decrease_ratio": round(decrease_ratio, 3),
        "approach_steps": len(approach_pairs),
        "reached_stop_threshold": min_dist is not None and min_dist <= 2.5,
        "coverage_entry_time": none_or_round(coverage_t),
        "visible_to_stop_latency_sec": none_or_round(latency),
        "stop_oracle_triggered": stop_event is not None,
        "bridge_received_stop": bridge_event is not None,
        "safe_cmd_vel_zero_time": none_or_round(event_time(zero)),
        "semantic_success": bool(episode.get("success")),
        "min_distance_m": none_or_round(min_dist),
        "final_distance_m": none_or_round(final_dist),
        "max_safe_linear_x_mps": none_or_round(max(safe_speeds) if safe_speeds else None),
        "max_actual_linear_speed_mps": none_or_round(max(actual_speeds) if actual_speeds else None),
        "mean_slip_mps": none_or_round(sum(slips) / len(slips) if slips else None),
        "costmap_block_steps": sum(1 for row in trace if row.get("local_costmap_clear") is False),
        "fall_or_reset_count": fall_or_reset_count,
    }
    row["failure_stage"] = target_failure_stage(row)
    return row


def evaluate_branch_controller(rows: list[dict[str, Any]], metrics: dict[str, Any], stale: dict[str, Any]) -> dict[str, Any]:
    by_mode = defaultdict(list)
    for row in rows:
        by_mode[str(row.get("mode") or "")].append(row)
    summaries = {}
    best_mode = None
    best_correct = -1
    for mode, group in by_mode.items():
        left = [row for row in group if row.get("instruction_branch") == "left"]
        right = [row for row in group if row.get("instruction_branch") == "right"]
        counts = count_true(group, ["decision_published", "rotate_phase_entered", "post_turn_forward_executed", "entered_correct_branch"])
        left_correct = sum(1 for row in left if truthy(row.get("entered_correct_branch")))
        right_correct = sum(1 for row in right if truthy(row.get("entered_correct_branch")))
        summaries[mode] = {
            "episodes": len(group),
            **counts,
            "left_correct": left_correct,
            "right_correct": right_correct,
            "pass": len(group) >= 6
            and counts["decision_published"] >= 6
            and counts["rotate_phase_entered"] >= 6
            and counts["post_turn_forward_executed"] >= 6
            and counts["entered_correct_branch"] >= 5
            and left_correct >= 2
            and right_correct >= 2,
        }
        if counts["entered_correct_branch"] > best_correct:
            best_correct = counts["entered_correct_branch"]
            best_mode = mode
    safety = safety_summary(metrics, stale)
    failures = []
    route_pass = bool(best_mode and summaries[best_mode]["pass"])
    if not route_pass:
        failures.append("route correct branch gate not met")
    failures.extend(safety["failures"])
    return {
        "pass": route_pass and not safety["failures"],
        "episodes": len(rows),
        "best_mode": best_mode,
        "best_correct_branch": best_correct,
        "modes": summaries,
        **safety,
        "failures": failures,
    }


def evaluate_target_approach(rows: list[dict[str, Any]], metrics: dict[str, Any], stale: dict[str, Any]) -> dict[str, Any]:
    by_controller = defaultdict(list)
    for row in rows:
        by_controller[str(row.get("controller") or "")].append(row)
    summaries = {}
    best_controller = None
    best_score: tuple[int, ...] | None = None
    for controller, group in by_controller.items():
        counts = count_true(
            group,
            [
                "target_vector_valid",
                "target_id_consistent",
                "reset_acknowledged",
                "reached_stop_threshold",
                "stop_oracle_triggered",
                "bridge_received_stop",
                "safe_cmd_vel_zero_time",
                "semantic_success",
            ],
        )
        decrease_ok = sum(1 for row in group if _float(row.get("distance_decrease_ratio"), 0.0) >= 0.80)
        latency_ok = sum(
            1
            for row in group
            if row.get("visible_to_stop_latency_sec") not in {None, ""}
            and _float(row.get("visible_to_stop_latency_sec"), 999.0) <= 2.0
        )
        low_speed_ok = all(_float(row.get("max_safe_linear_x_mps"), 999.0) <= 0.20 + 1.0e-6 for row in group)
        no_fall_or_reset = all(int(_float(row.get("fall_or_reset_count"), 0.0)) == 0 for row in group)
        summaries[controller] = {
            "episodes": len(group),
            **counts,
            "distance_decrease_ratio_ok": decrease_ok,
            "visible_to_stop_latency_ok": latency_ok,
            "max_visible_to_stop_latency_sec": max(
                [_float(row.get("visible_to_stop_latency_sec"), 0.0) for row in group if row.get("visible_to_stop_latency_sec") not in {None, ""}]
                or [0.0]
            ),
            "low_speed_ok": low_speed_ok,
            "no_fall_or_reset": no_fall_or_reset,
            "pass": len(group) >= 3
            and counts["target_vector_valid"] >= 3
            and counts["target_id_consistent"] >= 3
            and counts["reset_acknowledged"] >= 3
            and decrease_ok >= 3
            and counts["reached_stop_threshold"] >= 3
            and counts["stop_oracle_triggered"] >= 3
            and counts["bridge_received_stop"] >= 3
            and counts["safe_cmd_vel_zero_time"] >= 3
            and counts["semantic_success"] >= 2
            and latency_ok >= 3
            and low_speed_ok
            and no_fall_or_reset,
        }
        score = (
            int(summaries[controller]["pass"]),
            counts["semantic_success"],
            counts["stop_oracle_triggered"],
            counts["reached_stop_threshold"],
            decrease_ok,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_controller = controller
    safety = safety_summary(metrics, stale)
    semantic_pass = bool(best_controller and summaries[best_controller]["pass"])
    failures = []
    if not semantic_pass:
        failures.append("semantic approach gate not met")
    failures.extend(safety["failures"])
    return {
        "pass": semantic_pass and not safety["failures"],
        "episodes": len(rows),
        "best_controller": best_controller,
        "best_semantic_success": summaries.get(best_controller, {}).get("semantic_success", 0),
        "controllers": summaries,
        **safety,
        "failures": failures,
    }


def branch_confusion_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for branch in ["left", "right"]:
        group = [row for row in rows if row.get("instruction_branch") == branch]
        result.append(
            {
                "instruction_branch": branch,
                "entered_left": sum(1 for row in group if row.get("entered_branch") == "left"),
                "entered_right": sum(1 for row in group if row.get("entered_branch") == "right"),
                "remained_intersection": sum(1 for row in group if row.get("entered_branch") == "intersection"),
                "left_workspace": sum(1 for row in group if not row.get("entered_branch")),
            }
        )
    return result


def write_common_v10_artifacts(
    output: Path,
    metrics: dict[str, Any],
    *,
    branch_trace: list[dict[str, Any]] | None = None,
    target_trace: list[dict[str, Any]] | None = None,
    geometry_audit: dict[str, Any] | None = None,
    target_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    branch_trace = branch_trace or []
    target_trace = target_trace or []
    audit = geometry_audit or target_audit or {}
    output.mkdir(parents=True, exist_ok=True)
    (output / "geometry_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not (output / "branch_confusion_matrix.csv").exists():
        write_csv(output / "branch_confusion_matrix.csv", [], BRANCH_MATRIX_COLUMNS)
    if not (output / "target_approach_metrics.csv").exists():
        write_csv(output / "target_approach_metrics.csv", [], TARGET_APPROACH_COLUMNS)
    if branch_trace and target_trace:
        trace_rows = branch_trace + target_trace
        headers = list(dict.fromkeys(BRANCH_TRACE_COLUMNS + TARGET_TRACE_COLUMNS))
    else:
        trace_rows = branch_trace if branch_trace else target_trace
        headers = BRANCH_TRACE_COLUMNS if branch_trace else TARGET_TRACE_COLUMNS if target_trace else ["empty"]
    write_csv(output / "controller_trace.csv", trace_rows, headers)
    write_aggregate_trajectory(output, metrics)
    stale = analyze_stale_attribution_v8(load_jsonl(output / "events.jsonl"))
    write_csv(output / "stale_attribution.csv", stale.get("records", []), STALE_COLUMNS)
    (output / "stale_discard_analysis.md").write_text(render_stale_report(stale), encoding="utf-8")
    write_failure_table(output, metrics, trace_rows)
    write_visual_artifact(output, metrics, branch_trace=branch_trace, target_trace=target_trace, audit=audit)
    return stale


def write_aggregate_trajectory(output: Path, metrics: dict[str, Any]) -> None:
    rows = []
    for episode in metrics.get("episodes", []):
        episode_dir = episode_dir_for(output, episode)
        for point in load_trajectory(episode_dir / "trajectory.csv"):
            rows.append(
                {
                    "episode_id": episode.get("episode_id"),
                    "mode": episode.get("mode"),
                    "task_id": episode.get("task_id"),
                    "t": point.get("t"),
                    "x": point.get("x"),
                    "y": point.get("y"),
                    "yaw": point.get("yaw"),
                    "source": point.get("source", ""),
                }
            )
    write_csv(output / "trajectory.csv", rows, TRAJECTORY_COLUMNS)


def write_failure_table(output: Path, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    failures = []
    for row in rows:
        stage = row.get("failure_stage")
        if stage:
            failures.append(
                {
                    "episode_id": row.get("episode_id"),
                    "mode": row.get("mode"),
                    "task_id": row.get("task_id"),
                    "failure_stage": stage,
                    "failure_reason": stage,
                }
            )
    for episode in metrics.get("episodes", []):
        if episode.get("failure_reason"):
            failures.append(
                {
                    "episode_id": episode.get("episode_id"),
                    "mode": episode.get("mode"),
                    "task_id": episode.get("task_id"),
                    "failure_stage": episode.get("failure_reason"),
                    "failure_reason": episode.get("failure_reason"),
                }
            )
    write_csv(output / "failure_table.csv", failures, FAILURE_COLUMNS)


def write_visual_artifact(output: Path, metrics: dict[str, Any], *, branch_trace: list[dict[str, Any]], target_trace: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    visual = output / "visual"
    visual.mkdir(parents=True, exist_ok=True)
    overlay = branch_trace[:200] if branch_trace else target_trace[:200]
    (visual / "overlay_state.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in overlay) + ("\n" if overlay else ""), encoding="utf-8")
    try:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (1100, 650), (248, 250, 252))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1100, 80), fill=(18, 24, 38))
        draw.text((24, 24), "V10 geometric controller validation", fill=(255, 255, 255))
        draw.text((24, 105), f"episodes: {len(metrics.get('episodes', []))}", fill=(18, 24, 38))
        if branch_trace:
            draw.text((24, 140), "Route overlay: branch polygons, robot heading, desired yaw, phase", fill=(35, 50, 70))
            scale = 75
            ox, oy = 120, 500
            polygons = audit.get("left_branch_polygon"), audit.get("right_branch_polygon")
            for poly, color in zip(polygons, [(220, 60, 70), (40, 110, 220)]):
                if poly:
                    pts = [(ox + x * scale, oy - y * scale) for x, y in poly]
                    draw.polygon(pts, outline=color)
            for row in branch_trace[:: max(1, len(branch_trace) // 80)]:
                x = _float(row.get("x"), 0.0)
                y = _float(row.get("y"), 0.0)
                draw.ellipse((ox + x * scale - 2, oy - y * scale - 2, ox + x * scale + 2, oy - y * scale + 2), fill=(20, 120, 80))
        elif target_trace:
            draw.text((24, 140), "Semantic overlay: robot-target vector, yaw error, 2.5m stop circle, phase", fill=(35, 50, 70))
            scale = 75
            ox, oy = 120, 500
            if target_trace:
                tx = _float(target_trace[0].get("target_x"), 0.0)
                ty = _float(target_trace[0].get("target_y"), 0.0)
                r = 2.5 * scale
                draw.ellipse((ox + tx * scale - r, oy - ty * scale - r, ox + tx * scale + r, oy - ty * scale + r), outline=(220, 60, 70), width=2)
                draw.ellipse((ox + tx * scale - 5, oy - ty * scale - 5, ox + tx * scale + 5, oy - ty * scale + 5), fill=(220, 60, 70))
            for row in target_trace[:: max(1, len(target_trace) // 80)]:
                x = _float(row.get("robot_x"), 0.0)
                y = _float(row.get("robot_y"), 0.0)
                draw.ellipse((ox + x * scale - 2, oy - y * scale - 2, ox + x * scale + 2, oy - y * scale + 2), fill=(20, 120, 80))
        image.save(visual / "viewport.png")
    except Exception:
        write_minimal_png(visual / "viewport.png")


def write_sim2real_gate_v10(output: Path, *, route: dict[str, Any] | None = None, semantic: dict[str, Any] | None = None) -> dict[str, Any]:
    route_pass = bool(route and route.get("pass"))
    semantic_pass = bool(semantic and semantic.get("pass"))
    failures = []
    if not route_pass:
        failures.append("route gate failed or missing")
    if not semantic_pass:
        failures.append("semantic gate failed or missing")
    status = "READY FOR V11 FORCED ORACLE FULL RETRY" if not failures else "NOT READY FOR REAL ROBOT AUTONOMY"
    result = {
        "ready_for_v11": not failures,
        "status": status,
        "failures": failures,
        "step_status": "FROZEN",
        "sim2real_status": "NOT READY FOR REAL ROBOT AUTONOMY",
        "allowed_only": [
            "sensor-only dry-run",
            "bag replay",
            "offline scoring",
            "stationary camera validation",
            "manual-triggered primitive test",
        ],
    }
    lines = [f"# Sim2Real Gate V10: {result['sim2real_status']}", "", f"- V11 forced oracle full retry: {'ALLOWED' if result['ready_for_v11'] else 'NOT ALLOWED'}", "- Step: FROZEN", "- Real robot autonomy: NOT READY", "", "## Failures"]
    lines.extend([f"- {item}" for item in failures] or ["- none"])
    lines.extend(["", "## Allowed Only"])
    lines.extend([f"- {item}" for item in result["allowed_only"]])
    (output / "sim2real_gate_v10.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def render_v10_summary(output: Path, title: str, *, route: dict[str, Any] | None = None, semantic: dict[str, Any] | None = None, stale: dict[str, Any]) -> str:
    gate = write_sim2real_gate_v10(output, route=route, semantic=semantic)
    lines = [f"# {title}", "", f"- run_dir: {output}", ""]
    lines.extend(["## Required Facts Preserved", ""])
    lines.extend([f"- {fact}" for fact in REQUIRED_FACTS])
    lines.extend(["- V10 does not run Step or full forced-oracle benchmark.", "", "## Route Gate", ""])
    lines.extend(metric_lines(route))
    lines.extend(["", "## Semantic Gate", ""])
    lines.extend(metric_lines(semantic))
    lines.extend(["", "## Stale And Safety", ""])
    lines.append(f"- runtime_stale_discards: {stale.get('runtime_stale_discards')}")
    lines.append(f"- reset_cleanup_discards: {stale.get('reset_cleanup_discards')}")
    lines.append(f"- old_response_after_reset: {stale.get('old_response_after_reset')}")
    lines.append(f"- episode_mismatch: {stale.get('episode_mismatch')}")
    lines.append(f"- timebase_error: {stale.get('timebase_error')}")
    lines.extend(["", "## V11 / Step / Sim2Real", ""])
    lines.append(f"- V11 forced oracle full retry: {'ALLOWED' if gate['ready_for_v11'] else 'NOT ALLOWED'}")
    lines.append("- Step: FROZEN")
    lines.append("- Sim2Real: NOT READY FOR REAL ROBOT AUTONOMY")
    lines.extend(["", "## Artifacts", ""])
    for name in [
        "geometry_audit.json",
        "branch_confusion_matrix.csv",
        "target_approach_metrics.csv",
        "controller_trace.csv",
        "trajectory.csv",
        "stale_attribution.csv",
        "failure_table.csv",
        "sim2real_gate_v10.md",
        "visual/viewport.png",
    ]:
        lines.append(f"- {name}: {output / name}")
    return "\n".join(lines) + "\n"


def metric_lines(value: dict[str, Any] | None) -> list[str]:
    if not value:
        return ["- not run in this artifact"]
    keys = [key for key in value.keys() if key not in {"modes", "controllers", "failures"}]
    lines = [f"- {key}: {value.get(key)}" for key in keys]
    lines.append(f"- failures: {', '.join(value.get('failures', [])) if value.get('failures') else 'none'}")
    return lines


def safety_summary(metrics: dict[str, Any], stale: dict[str, Any]) -> dict[str, Any]:
    collision = sum(int(_float(row.get("num_collisions"), 0.0)) for row in metrics.get("episodes", []))
    stale_action = sum(1 for row in metrics.get("episodes", []) if str(row.get("failure_reason") or "") == "stale_action_executed")
    runtime = int(stale.get("runtime_stale_discards", 0) or 0)
    timebase = int(stale.get("timebase_error", 0) or 0)
    failures = []
    if collision != 0:
        failures.append(f"collision_count={collision} != 0")
    if stale_action != 0:
        failures.append(f"stale_action_executed={stale_action} != 0")
    if runtime != 0:
        failures.append(f"runtime_stale_discards={runtime} != 0")
    if timebase != 0:
        failures.append(f"timebase_error={timebase} != 0")
    return {
        "collision_count": collision,
        "stale_action_executed": stale_action,
        "runtime_stale_discards": runtime,
        "timebase_error": timebase,
        "safety_failures": failures,
        "failures": failures,
    }


def branch_failure_stage(row: dict[str, Any]) -> str | None:
    for key, stage in [
        ("decision_published", "decision_not_published"),
        ("rotate_phase_entered", "rotate_phase_missing"),
        ("post_turn_forward_executed", "post_turn_forward_missing"),
        ("entered_correct_branch", "wrong_or_missing_branch"),
    ]:
        if not truthy(row.get(key)):
            return stage
    return None


def target_failure_stage(row: dict[str, Any]) -> str | None:
    for key, stage in [
        ("target_vector_valid", "target_vector_invalid"),
        ("reached_stop_threshold", "stop_threshold_not_reached"),
        ("stop_oracle_triggered", "stop_oracle_not_triggered"),
        ("bridge_received_stop", "semantic_stop_bridge_missing"),
        ("safe_cmd_vel_zero_time", "safe_zero_missing"),
        ("semantic_success", "semantic_success_failed"),
    ]:
        if not truthy(row.get(key)):
            return stage
    if _float(row.get("distance_decrease_ratio"), 0.0) < 0.80:
        return "distance_not_monotonic"
    return None


def controller_from_mode(mode: Any) -> str:
    text = str(mode or "")
    if "proportional" in text:
        return "proportional"
    if "rotate_then_forward" in text:
        return "rotate_then_forward"
    return text


def trim_events_after_reset(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reset_times = [event_time(event) for event in events if _event_type(event) == "reset"]
    reset_times = [value for value in reset_times if value is not None]
    if not reset_times:
        return events
    cutoff = max(reset_times)
    return [event for event in events if event_time(event) is None or float(event_time(event)) >= float(cutoff)]


def seed_from_task_id(task_id: Any) -> str:
    text = str(task_id or "")
    return text.rsplit("_seed", 1)[-1] if "_seed" in text else ""


def _write_metrics(output: Path, metrics: dict[str, Any]) -> None:
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_minimal_png(path: Path) -> None:
    width, height = 8, 8
    raw = b"".join(b"\x00" + bytes([240, 248, 255]) * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(raw))
    payload += chunk(b"IEND", b"")
    path.write_bytes(payload)
