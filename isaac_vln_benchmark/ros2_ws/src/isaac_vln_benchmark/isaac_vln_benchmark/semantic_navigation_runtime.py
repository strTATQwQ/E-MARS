from __future__ import annotations

import copy
import math
import re
from collections import Counter
from typing import Any

from .semantic_navigation_benchmark import validate_task_set


MARKERS = [
    ("fire_extinguisher", "red", "red fire extinguisher"),
    ("box", "blue", "blue box"),
    ("cone", "red", "red traffic cone"),
    ("exit_sign", "green", "green exit sign"),
    ("box", "yellow", "yellow box"),
]

V18_PROFILE = "v18_marker_v1"
V19_PROFILE = "v19_local_marker_v2"
RUNTIME_PROFILES = {V18_PROFILE, V19_PROFILE}


def materialize_semantic_navigation_runtime(
    task_set: dict[str, Any],
    *,
    profile: str = V18_PROFILE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_task_set(task_set)
    if profile not in RUNTIME_PROFILES:
        raise ValueError(f"unsupported semantic runtime profile: {profile}")
    tasks: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    for task in task_set["tasks"]:
        runtime_task, scene = _materialize_task(task, profile=profile)
        tasks.append(runtime_task)
        scenes.append(scene)
    return (
        {
            "schema_version": 1,
            "source_benchmark": str(task_set.get("benchmark") or "hierarchical_semantic_navigation_v18"),
            "runtime_profile": profile,
            "instrumented_semantic_markers": True,
            "qualification_evidence": False,
            "tasks": tasks,
        },
        {
            "schema_version": 1,
            "runtime_profile": profile,
            "instrumented_semantic_markers": True,
            "qualification_evidence": False,
            "default_camera": {"front_fov_deg": 90.0, "max_range_m": 12.0},
            "scenes": scenes,
        },
    )


def _materialize_task(task: dict[str, Any], *, profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = str(task["task_id"])
    source_plan = copy.deepcopy(task["oracle_plan"])
    plan = _ensure_approach_before_verify(source_plan) if profile == V19_PROFILE else copy.deepcopy(source_plan)
    objects: list[dict[str, Any]] = []
    marker_for_index: dict[int, str] = {}
    execution_targets: dict[int, str] = {}
    last_marker_index: int | None = None
    marker_index = 0

    for subgoal_index, subgoal in enumerate(plan):
        subgoal_type = str(subgoal["subgoal_type"])
        original_target = str(subgoal["target"])
        if subgoal_type in {"approach", "verify"} and last_marker_index is not None:
            assigned = last_marker_index
        elif subgoal_type == "ask":
            marker_for_index[subgoal_index] = ""
            execution_targets[subgoal_index] = original_target
            continue
        else:
            assigned = marker_index
            marker_index += 1
            last_marker_index = assigned
            if profile == V19_PROFILE:
                marker_class, color, marker_label = MARKERS[assigned % len(MARKERS)]
            else:
                marker_class, color, marker_label = _marker_for_target(original_target, assigned)
            x_pos = 2.3 + assigned * 1.8
            y_pos = (0.0, 0.75, -0.75, 0.35, -0.35)[assigned % 5]
            object_id = f"{task_id}_marker_{assigned:02d}"
            objects.append(
                {
                    "id": object_id,
                    "class": marker_class,
                    "color": color,
                    "pose": [round(x_pos, 3), round(y_pos, 3), _marker_z(marker_class)],
                    "radius": 0.48,
                    "success_condition": "semantic_marker",
                    "semantic_original_target": original_target,
                    "semantic_marker_label": marker_label,
                }
            )
        object_id = f"{task_id}_marker_{assigned:02d}"
        marker_for_index[subgoal_index] = object_id
        marker = next(obj for obj in objects if obj["id"] == object_id)
        marker_label = str(marker["semantic_marker_label"])
        if profile == V19_PROFILE:
            execution_target = f"nearest {marker_label} marker"
            subgoal["relation"] = ""
            subgoal["constraints"] = []
        elif _target_matches_marker(original_target, marker_label):
            execution_target = original_target
        else:
            execution_target = f"{marker_label} marking {original_target}"
        execution_targets[subgoal_index] = execution_target
        subgoal["target"] = execution_target

    if not objects:
        objects.append(
            {
                "id": f"{task_id}_hold_marker",
                "class": "exit_sign",
                "color": "white",
                "pose": [2.0, 0.0, 1.2],
                "radius": 0.45,
                "success_condition": "semantic_marker",
                "semantic_original_target": "operator clarification",
                "semantic_marker_label": "white sign",
            }
        )
    max_x = max(float(obj["pose"][0]) for obj in objects)
    scene_prefix = "v19_semantic" if profile == V19_PROFILE else "v18_semantic"
    scene_type = f"{scene_prefix}_{task_id}"
    scene_id = f"{scene_type}_001"
    final_object_id = next((marker_for_index[index] for index in reversed(range(len(plan))) if marker_for_index.get(index)), objects[-1]["id"])
    runtime_task = {
        "task_id": task_id,
        "task_type": "semantic_navigation",
        "category": str(task["category"]),
        "scene_template": str(task["scene_template"]),
        "scene_type": scene_type,
        "instruction": str(task["instruction"]),
        "target_object": final_object_id,
        "timeout_sec": min(120, int(task.get("timeout_sec", 180))),
        "success": {"distance_to_target_m": 2.0, "target_visible": False, "stop_required": True},
        "judge": copy.deepcopy(task["judge"]),
        "oracle_plan": plan,
        "original_oracle_plan": source_plan,
        "semantic_runtime": {
            "scene_id": scene_id,
            "subgoal_object_ids": {str(index): marker_for_index.get(index, "") for index in range(len(plan))},
            "execution_targets": {str(index): execution_targets[index] for index in range(len(plan))},
            "runtime_profile": profile,
            "judge_completion_policy": "route_progress_v2" if profile == V19_PROFILE else "proximity_v1",
            "pass_reach_distance_m": 1.65,
            "pass_cross_track_max_m": 1.75,
            "enter_reach_distance_m": 1.50,
            "enter_cross_track_max_m": 1.50,
            "instrumented_semantic_markers": True,
            "geometry_judge_is_oracle_only": True,
            "qualification_evidence": False,
        },
    }
    runtime_task["judge"]["required_sequence"] = [str(subgoal["subgoal_type"]) for subgoal in plan]
    scene = {
        "scene_id": scene_id,
        "scene_type": scene_type,
        "robot_start_pose": [0.0, 0.0, 0.0],
        "bounds": [-1.5, -2.4, round(max_x + 2.5, 3), 2.4],
        "semantic_zones": [
            {
                "id": f"{task_id}_semantic_route",
                "class": "semantic_route",
                "center": [round(max_x * 0.5, 3), 0.0],
                "radius": round(max(3.0, max_x * 0.65), 3),
            }
        ],
        "objects": objects,
        "obstacles": _runtime_obstacles(task, max_x),
        "dynamic_obstacles": [],
        "blocked_paths": ["injected_semantic_recovery"] if task["category"] == "recovery" else [],
        "semantic_distractors": [],
        "lighting_seed": 1800 + _stable_number(task_id),
        "texture_seed": 2800 + _stable_number(task_id),
        "instrumented_semantic_markers": True,
        "qualification_evidence": False,
        "runtime_profile": profile,
    }
    return runtime_task, scene


def _ensure_approach_before_verify(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for subgoal in copy.deepcopy(plan):
        if (
            str(subgoal.get("subgoal_type") or "") == "verify"
            and result
            and str(result[-1].get("subgoal_type") or "") == "find"
        ):
            previous = result[-1]
            result.append(
                {
                    "subgoal_type": "approach",
                    "target": str(previous.get("target") or subgoal.get("target") or ""),
                    "relation": str(previous.get("relation") or subgoal.get("relation") or ""),
                    "constraints": list(previous.get("constraints") or []),
                    "completion_evidence": "confirmed target is within task stop distance",
                    "recovery": "scan",
                    "confidence": 1.0,
                }
            )
        result.append(subgoal)
    if result and str(result[-1].get("subgoal_type") or "") not in {"verify", "ask"}:
        if str(result[-1].get("subgoal_type") or "") == "find":
            last = result[-1]
            result.append(
                {
                    "subgoal_type": "approach",
                    "target": str(last.get("target") or ""),
                    "relation": str(last.get("relation") or ""),
                    "constraints": list(last.get("constraints") or []),
                    "completion_evidence": "confirmed target is within task stop distance",
                    "recovery": "scan",
                    "confidence": 1.0,
                }
            )
        last = result[-1]
        result.append(
            {
                "subgoal_type": "verify",
                "target": str(last.get("target") or ""),
                "relation": str(last.get("relation") or ""),
                "constraints": list(last.get("constraints") or []),
                "completion_evidence": "target identity and safe terminal stop are confirmed",
                "recovery": "stop",
                "confidence": 1.0,
            }
        )
    return result


def runtime_audit(tasks_doc: dict[str, Any], scenes_doc: dict[str, Any]) -> dict[str, Any]:
    tasks = list(tasks_doc.get("tasks") or [])
    scenes = list(scenes_doc.get("scenes") or [])
    scene_by_type = {str(scene.get("scene_type")): scene for scene in scenes}
    categories = Counter(str(task.get("category")) for task in tasks)
    errors: list[str] = []
    marker_count = 0
    for task in tasks:
        scene = scene_by_type.get(str(task.get("scene_type")))
        if scene is None:
            errors.append(f"{task.get('task_id')}:missing_scene")
            continue
        objects = {str(obj.get("id")): obj for obj in scene.get("objects", [])}
        marker_count += len(objects)
        runtime = task.get("semantic_runtime") if isinstance(task.get("semantic_runtime"), dict) else {}
        mapping = runtime.get("subgoal_object_ids") if isinstance(runtime.get("subgoal_object_ids"), dict) else {}
        for index, subgoal in enumerate(task.get("oracle_plan", [])):
            object_id = str(mapping.get(str(index)) or "")
            if str(subgoal.get("subgoal_type")) != "ask" and object_id not in objects:
                errors.append(f"{task.get('task_id')}:subgoal_{index}_missing_marker")
            forbidden = set(subgoal) & {"cmd_vel", "waypoint", "primitive", "target_pose", "trajectory"}
            if forbidden:
                errors.append(f"{task.get('task_id')}:subgoal_{index}_motion_fields")
    return {
        "task_count": len(tasks),
        "scene_count": len(scenes),
        "category_counts": dict(categories),
        "marker_count": marker_count,
        "errors": errors,
        "pass": len(tasks) == 30 and len(scenes) == 30 and categories == Counter({"referential": 10, "compositional": 10, "recovery": 10}) and not errors,
        "instrumented_semantic_markers": True,
        "qualification_evidence": False,
    }


def distance_xy(pose: list[float], target_pose: list[float]) -> float:
    return math.hypot(float(target_pose[0]) - float(pose[0]), float(target_pose[1]) - float(pose[1]))


def _marker_for_target(target: str, index: int) -> tuple[str, str, str]:
    lowered = target.lower()
    if "extinguisher" in lowered:
        return "fire_extinguisher", "red", "red fire extinguisher"
    if "cone" in lowered:
        return "cone", "red", "red traffic cone"
    if any(word in lowered for word in ("box", "bin", "cart")):
        color = "yellow" if "yellow" in lowered else "blue"
        return "box", color, f"{color} box"
    if "sign" in lowered:
        color = "green" if "green" in lowered else "red"
        return "exit_sign", color, f"{color} exit sign"
    return MARKERS[index % len(MARKERS)]


def _target_matches_marker(target: str, marker: str) -> bool:
    target_tokens = set(re.findall(r"[a-z]+", target.lower()))
    marker_tokens = set(re.findall(r"[a-z]+", marker.lower()))
    return bool(marker_tokens and marker_tokens.issubset(target_tokens))


def _marker_z(marker_class: str) -> float:
    if marker_class in {"exit_sign", "sign"}:
        return 1.25
    if marker_class == "fire_extinguisher":
        return 0.5
    return 0.45


def _runtime_obstacles(task: dict[str, Any], max_x: float) -> list[dict[str, Any]]:
    if task["category"] != "recovery":
        return []
    return [
        {
            "id": f"{task['task_id']}_recovery_barrier",
            "class": "obstacle",
            "pose": [round(min(3.2, max_x * 0.45), 3), 1.55, 0.55],
            "size": [0.5, 0.35, 1.1],
            "color": "gray",
        }
    ]


def _stable_number(text: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % 900
