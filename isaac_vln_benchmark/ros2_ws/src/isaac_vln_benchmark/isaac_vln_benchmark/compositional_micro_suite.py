from __future__ import annotations

import copy
import math
from typing import Any


BLOCKER_CASES = {
    "comp_01": {"single_index": 0, "chain_start_index": 0, "chain_length": 2},
    "comp_02": {"single_index": 0, "chain_start_index": 0, "chain_length": 2},
    "comp_03": {"single_index": 0, "chain_start_index": 0, "chain_length": 2},
    "comp_04": {"single_index": 4, "chain_start_index": 3, "chain_length": 3},
    "comp_05": {"single_index": 3, "chain_start_index": 2, "chain_length": 3},
    "comp_07": {"single_index": 1, "chain_start_index": 0, "chain_length": 2},
}


def build_v19_blocker_micro_suite(
    tasks_doc: dict[str, Any],
    scenes_doc: dict[str, Any],
    *,
    stage: str,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> tuple[dict[str, Any], dict[str, Any]]:
    if stage not in {"single", "chain"}:
        raise ValueError(f"unsupported micro stage: {stage}")
    tasks = {str(task["task_id"]): task for task in tasks_doc.get("tasks", [])}
    scenes = {str(scene["scene_id"]): scene for scene in scenes_doc.get("scenes", [])}
    micro_tasks: list[dict[str, Any]] = []
    micro_scenes: list[dict[str, Any]] = []
    for seed in seeds:
        for parent_task_id, case in BLOCKER_CASES.items():
            source_task = tasks[parent_task_id]
            source_runtime = source_task["semantic_runtime"]
            source_scene = scenes[str(source_runtime["scene_id"])]
            start_index = int(case["single_index"] if stage == "single" else case["chain_start_index"])
            length = 1 if stage == "single" else int(case["chain_length"])
            task, scene = _slice_case(
                source_task,
                source_scene,
                parent_task_id=parent_task_id,
                start_index=start_index,
                length=length,
                stage=stage,
                seed=int(seed),
            )
            micro_tasks.append(task)
            micro_scenes.append(scene)
    return (
        {
            "schema_version": 1,
            "source_benchmark": "hierarchical_semantic_navigation_v19",
            "runtime_profile": "v19_local_marker_v2",
            "micro_stage": stage,
            "instrumented_semantic_markers": True,
            "qualification_evidence": False,
            "tasks": micro_tasks,
        },
        {
            "schema_version": 1,
            "runtime_profile": "v19_local_marker_v2",
            "micro_stage": stage,
            "instrumented_semantic_markers": True,
            "qualification_evidence": False,
            "scenes": micro_scenes,
        },
    )


def _slice_case(
    source_task: dict[str, Any],
    source_scene: dict[str, Any],
    *,
    parent_task_id: str,
    start_index: int,
    length: int,
    stage: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_plan = list(source_task["oracle_plan"])
    end_index = min(len(source_plan), start_index + length)
    plan = copy.deepcopy(source_plan[start_index:end_index])
    if not plan:
        raise ValueError(f"empty micro plan for {parent_task_id}:{start_index}+{length}")
    source_runtime = source_task["semantic_runtime"]
    source_mapping = source_runtime["subgoal_object_ids"]
    source_objects = {str(obj["id"]): obj for obj in source_scene.get("objects", [])}
    selected_ids = [str(source_mapping[str(index)]) for index in range(start_index, end_index)]
    if stage == "chain" and str(plan[-1].get("subgoal_type") or "") not in {"verify", "ask"}:
        last = plan[-1]
        plan.append(
            {
                "subgoal_type": "verify",
                "target": str(last.get("target") or ""),
                "relation": str(last.get("relation") or ""),
                "constraints": list(last.get("constraints") or []),
                "completion_evidence": "local chain completion and safe stop are confirmed",
                "recovery": "stop",
                "confidence": 1.0,
            }
        )
        selected_ids.append(selected_ids[-1])
    mapping = {str(index): selected_ids[index] for index in range(len(selected_ids))}
    execution_targets = {str(index): str(plan[index]["target"]) for index in range(len(plan))}
    objects = [copy.deepcopy(source_objects[object_id]) for object_id in dict.fromkeys(selected_ids) if object_id]
    start_pose = _micro_start_pose(
        source_scene,
        source_mapping,
        source_objects,
        start_index=start_index,
        first_object_id=selected_ids[0],
        first_subgoal_type=str(plan[0]["subgoal_type"]),
    )
    task_id = f"v19_{parent_task_id}_{stage}_seed{seed}"
    scene_id = f"v19_micro_{parent_task_id}_{stage}_seed{seed}"
    task = copy.deepcopy(source_task)
    task.update(
        {
            "task_id": task_id,
            "category": "compositional",
            "scene_type": scene_id,
            "scene_template": "v19_compositional_marker_micro",
            "instruction": "Complete the current local semantic marker sequence.",
            "timeout_sec": 45,
            "oracle_plan": plan,
            "original_oracle_plan": copy.deepcopy(plan),
            "target_object": selected_ids[-1],
            "seed": seed,
            "judge": copy.deepcopy(source_task.get("judge") or {}),
            "semantic_runtime": {
                "scene_id": scene_id,
                "subgoal_object_ids": mapping,
                "execution_targets": execution_targets,
                "runtime_profile": "v19_local_marker_v2",
                "judge_completion_policy": "route_progress_v2",
                "pass_reach_distance_m": 1.65,
                "pass_cross_track_max_m": 1.75,
                "enter_reach_distance_m": 1.50,
                "enter_cross_track_max_m": 1.50,
                "parent_task_id": parent_task_id,
                "parent_start_index": start_index,
                "micro_stage": stage,
                "instrumented_semantic_markers": True,
                "geometry_judge_is_oracle_only": True,
                "qualification_evidence": False,
            },
        }
    )
    task["judge"]["required_sequence"] = [str(value["subgoal_type"]) for value in plan]
    task["judge"].pop("injected_event", None)
    task["judge"].pop("required_recovery", None)
    scene = copy.deepcopy(source_scene)
    scene.update(
        {
            "scene_id": scene_id,
            "scene_type": scene_id,
            "robot_start_pose": start_pose,
            "objects": objects,
            "obstacles": [],
            "dynamic_obstacles": [],
            "blocked_paths": [],
            "semantic_distractors": [],
            "lighting_seed": int(source_scene.get("lighting_seed", 0)) + seed * 101,
            "texture_seed": int(source_scene.get("texture_seed", 0)) + seed * 103,
            "runtime_profile": "v19_local_marker_v2",
            "qualification_evidence": False,
        }
    )
    return task, scene


def _micro_start_pose(
    scene: dict[str, Any],
    mapping: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    *,
    start_index: int,
    first_object_id: str,
    first_subgoal_type: str,
) -> list[float]:
    target = list(objects[first_object_id]["pose"])
    origin = list(scene.get("robot_start_pose") or [0.0, 0.0, 0.0])
    for previous in reversed(range(start_index)):
        previous_id = str(mapping.get(str(previous)) or "")
        if previous_id and previous_id != first_object_id and previous_id in objects:
            origin = list(objects[previous_id]["pose"])
            break
    dx = float(target[0]) - float(origin[0])
    dy = float(target[1]) - float(origin[1])
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        dx, dy, norm = 1.0, 0.0, 1.0
    ux, uy = dx / norm, dy / norm
    distance = 1.50 if first_subgoal_type in {"verify", "ask"} else min(2.30, norm)
    x = float(target[0]) - ux * distance
    y = float(target[1]) - uy * distance
    yaw = math.atan2(float(target[1]) - y, float(target[0]) - x)
    return [round(x, 4), round(y, 4), round(yaw, 6)]
