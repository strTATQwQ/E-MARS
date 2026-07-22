#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


OBSTACLE_CATEGORIES = (
    "static_clutter",
    "narrow_door",
    "narrow_corridor",
    "corner_dead_end",
    "dynamic_crossing",
)
BASE_SEED = 2026071300


def _jitter(rng: random.Random, magnitude: float = 0.16) -> float:
    return round(rng.uniform(-magnitude, magnitude), 3)


def _obstacle_scene(category: str, index: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed)
    scene_id = f"formal_{category}_{index:02d}"
    scene_type = scene_id
    target_id = f"target_{category}_{index:02d}"
    scene: dict[str, Any] = {
        "scene_id": scene_id,
        "scene_type": scene_type,
        "robot_start_pose": [0.0, 0.0, 0.0],
        "bounds": [-2.0, -3.0, 8.0, 3.0],
        "objects": [
            {
                "id": target_id,
                "class": "exit_sign",
                "pose": [7.0, 0.0, 1.5],
                "color": "red",
                "radius": 0.45,
                "success_condition": "within_distance_visible",
            }
        ],
        "obstacles": [],
        "dynamic_obstacles": [],
        "semantic_zones": [],
        "lighting_seed": seed,
        "texture_seed": seed + 10000,
        "random_seed": seed,
        "blocked_paths": [],
        "semantic_distractors": [],
    }
    if category == "static_clutter":
        scene["bounds"] = [-1.5, -2.0, 8.0, 2.0]
        for number, (x, y) in enumerate(((2.0, 0.45), (3.4, -0.55), (4.8, 0.35), (5.8, -0.65))):
            scene["obstacles"].append(
                {
                    "id": f"clutter_{number}",
                    "class": "box",
                    "pose": [x + _jitter(rng), y + _jitter(rng), 0.5],
                    "size": [0.55, 0.55, 1.0],
                }
            )
        instruction = "Avoid the cluttered boxes and stop at the red exit sign."
    elif category == "narrow_door":
        gap = 0.95 + _jitter(rng, 0.05)
        scene["obstacles"] = [
            {"id": "door_wall_left", "class": "wall", "pose": [3.2, 1.65, 0.8], "size": [0.3, 2.3 - gap / 2, 1.6]},
            {"id": "door_wall_right", "class": "wall", "pose": [3.2, -1.65, 0.8], "size": [0.3, 2.3 - gap / 2, 1.6]},
        ]
        scene["semantic_zones"] = [{"id": "narrow_door", "class": "doorway", "center": [3.2, 0.0], "radius": 0.9}]
        instruction = "Pass carefully through the narrow doorway and stop at the red exit sign."
    elif category == "narrow_corridor":
        half_width = 0.78 + _jitter(rng, 0.04)
        scene["bounds"] = [-1.0, -1.4, 8.0, 1.4]
        scene["obstacles"] = [
            {"id": "corridor_left", "class": "wall", "pose": [3.8, half_width + 0.25, 0.8], "size": [7.0, 0.35, 1.6]},
            {"id": "corridor_right", "class": "wall", "pose": [3.8, -half_width - 0.25, 0.8], "size": [7.0, 0.35, 1.6]},
            {"id": "corridor_box", "class": "box", "pose": [4.0 + _jitter(rng), 0.28, 0.4], "size": [0.35, 0.35, 0.8]},
        ]
        instruction = "Navigate the narrow corridor, avoid the box, and stop at the red exit sign."
    elif category == "corner_dead_end":
        scene["objects"][0]["pose"] = [6.3, 2.0, 1.5]
        scene["obstacles"] = [
            {"id": "dead_end_barrier", "class": "wall", "pose": [3.0, 0.0, 0.75], "size": [0.35, 3.0, 1.5]},
            {"id": "corner_wall", "class": "wall", "pose": [4.6, -1.55, 0.75], "size": [3.0, 0.3, 1.5]},
        ]
        scene["blocked_paths"] = ["straight_ahead"]
        instruction = "The route ahead is a dead end; back up or turn around the corner and stop at the red exit sign."
    elif category == "dynamic_crossing":
        scene["obstacles"] = [
            {"id": "side_barrier", "class": "box", "pose": [5.2, 0.9, 0.5], "size": [0.7, 0.35, 1.0]}
        ]
        scene["dynamic_obstacles"] = [
            {
                "id": "human_crossing",
                "class": "human_dummy",
                "path": [[3.2 + _jitter(rng, 0.1), -1.35], [3.2 + _jitter(rng, 0.1), 1.35]],
                "speed_mps": round(0.28 + 0.02 * (index % 6), 3),
                "radius": 0.45,
            }
        ]
        instruction = "Avoid the moving person crossing the path and stop at the red exit sign."
    else:
        raise KeyError(category)
    task = {
        "task_id": scene_id,
        "episode_key": scene_id,
        "task_type": "obstacle_avoidance",
        "benchmark_group": "obstacle_avoidance",
        "obstacle_category": category,
        "scene_id": scene_id,
        "scene_type": scene_type,
        "instruction": instruction,
        "target_object": target_id,
        "success": {"distance_to_target_m": 1.8, "target_visible": True, "stop_required": True},
        "timeout_sec": 120,
        "random_seed": seed,
        "subgoals": ["avoid obstacles", "reach target", "stop"],
        "expected_step_triggers": [],
    }
    return scene, task


def _semantic_scene(index: int, seed: int, *, complex_task: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed)
    prefix = "complex" if complex_task else "natural"
    scene_id = f"formal_{prefix}_{index:02d}"
    target_id = f"red_chair_{index:02d}"
    door_x = 3.0 + _jitter(rng, 0.12)
    scene = {
        "scene_id": scene_id,
        "scene_type": scene_id,
        "robot_start_pose": [0.0, 0.0, 0.0],
        "bounds": [-1.5, -2.8, 8.0, 2.8],
        "objects": [
            {"id": "door_marker", "class": "door_sign", "pose": [door_x, 0.9, 1.4], "color": "white", "radius": 0.35, "success_condition": "relation_anchor"},
            {"id": target_id, "class": "chair", "pose": [6.2, 1.25, 0.55], "color": "red", "radius": 0.55, "success_condition": "within_distance_visible"},
            {"id": "blue_chair_distractor", "class": "chair", "pose": [5.8, -1.35, 0.55], "color": "blue", "radius": 0.55, "success_condition": "distractor"},
            {"id": "trash_bin", "class": "trash_bin", "pose": [6.6, 1.7, 0.45], "color": "gray", "radius": 0.4, "success_condition": "relation_anchor"},
        ],
        "obstacles": [
            {"id": "front_box", "class": "box", "pose": [2.0 + _jitter(rng), 0.1, 0.5], "size": [0.65, 0.65, 1.0]},
            {"id": "door_wall_left", "class": "wall", "pose": [door_x, 1.85, 0.8], "size": [0.3, 1.25, 1.6]},
            {"id": "door_wall_right", "class": "wall", "pose": [door_x, -1.85, 0.8], "size": [0.3, 1.25, 1.6]},
        ],
        "dynamic_obstacles": [],
        "semantic_zones": [{"id": "doorway", "class": "doorway", "center": [door_x, 0.0], "radius": 0.9}],
        "lighting_seed": seed,
        "texture_seed": seed + 10000,
        "random_seed": seed,
        "blocked_paths": [],
        "semantic_distractors": ["blue_chair_distractor"],
    }
    if complex_task:
        variants = (
            "Go around the box, pass through the doorway, and stop in front of the red chair beside the trash bin.",
            "First avoid the box, then leave the corridor through the door, and stop near the red chair rather than the blue one.",
            "Pass the doorway after clearing the obstacle, find the chair next to the trash bin, and stop there.",
        )
        instruction = variants[index % len(variants)]
        subgoals = ["avoid front box", "pass doorway", "identify red chair by trash bin", "stop"]
        task_type = "complex_semantic_navigation"
        benchmark_group = "complex_semantic"
    else:
        instruction = "Navigate to the red chair next to the gray trash bin and stop."
        subgoals = ["identify red chair by trash bin", "reach target", "stop"]
        task_type = "natural_language_navigation"
        benchmark_group = "natural_language"
    task = {
        "task_id": scene_id,
        "episode_key": scene_id,
        "task_type": task_type,
        "benchmark_group": benchmark_group,
        "obstacle_category": "semantic_navigation",
        "scene_id": scene_id,
        "scene_type": scene_id,
        "instruction": instruction,
        "target_object": target_id,
        "relation_object": "trash_bin",
        "success": {"distance_to_target_m": 1.8, "target_visible": True, "stop_required": True},
        "timeout_sec": 150,
        "random_seed": seed,
        "subgoals": subgoals,
        "expected_step_triggers": ["mission_start", "completion_verification"],
    }
    return scene, task


def build_manifest() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for category_index, category in enumerate(OBSTACLE_CATEGORIES):
        for episode_index in range(20):
            seed = BASE_SEED + category_index * 100 + episode_index
            scene, task = _obstacle_scene(category, episode_index, seed)
            scenes.append(scene)
            tasks.append(task)
    for index in range(10):
        seed = BASE_SEED + 1000 + index
        scene, task = _semantic_scene(index, seed, complex_task=False)
        scenes.append(scene)
        tasks.append(task)
    for index in range(20):
        seed = BASE_SEED + 1100 + index
        scene, task = _semantic_scene(index, seed, complex_task=True)
        scenes.append(scene)
        tasks.append(task)
    smoke_tasks = [next(task for task in tasks if task["obstacle_category"] == category) for category in OBSTACLE_CATEGORIES]
    seeds = {
        "schema_version": 1,
        "generator": "scripts/generate_isaac_backtest_manifest.py",
        "base_seed": BASE_SEED,
        "phases": {
            "smoke": [
                {key: task[key] for key in ("episode_key", "task_id", "scene_id", "benchmark_group", "obstacle_category", "random_seed")}
                for task in smoke_tasks
            ],
            "formal": [
                {key: task[key] for key in ("episode_key", "task_id", "scene_id", "benchmark_group", "obstacle_category", "random_seed")}
                for task in tasks
            ],
        },
    }
    return (
        {"schema_version": 1, "scenes": scenes},
        {"schema_version": 1, "tasks": tasks},
        {"schema_version": 1, "tasks": smoke_tasks},
        seeds,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic OmniNav formal Isaac manifests.")
    parser.add_argument("--output-dir", default="configs/isaac")
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scenes, tasks, smoke, seeds = build_manifest()
    outputs = {
        "scenes.yaml": scenes,
        "tasks.yaml": tasks,
        "smoke_tasks.yaml": smoke,
        "seeds.json": seeds,
    }
    for name, value in outputs.items():
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "scenes": len(scenes["scenes"]), "tasks": len(tasks["tasks"]), "smoke": len(smoke["tasks"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
