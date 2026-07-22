from pathlib import Path

from isaac_vln_benchmark.semantic_navigation_benchmark import load_task_set
from isaac_vln_benchmark.semantic_navigation_runtime import (
    V19_PROFILE,
    materialize_semantic_navigation_runtime,
    runtime_audit,
)


TASK_FILE = Path(__file__).resolve().parents[4] / "configs" / "natural_language_navigation_v18.yaml"


def test_materialized_runtime_has_30_executable_marker_scenes():
    tasks, scenes = materialize_semantic_navigation_runtime(load_task_set(TASK_FILE))
    audit = runtime_audit(tasks, scenes)
    assert audit["pass"] is True
    assert audit["task_count"] == 30
    assert audit["scene_count"] == 30
    assert audit["marker_count"] > 30
    assert audit["qualification_evidence"] is False


def test_runtime_keeps_original_instruction_but_labels_execution_markers():
    source = load_task_set(TASK_FILE)
    tasks, scenes = materialize_semantic_navigation_runtime(source)
    task = tasks["tasks"][3]
    assert task["instruction"] == source["tasks"][3]["instruction"]
    assert task["task_type"] == "semantic_navigation"
    assert task["semantic_runtime"]["instrumented_semantic_markers"] is True
    assert any("marking" in subgoal["target"] for subgoal in task["oracle_plan"])
    assert scenes["scenes"][3]["instrumented_semantic_markers"] is True


def test_runtime_subgoals_still_contain_no_motion_or_geometry_commands():
    tasks, _ = materialize_semantic_navigation_runtime(load_task_set(TASK_FILE))
    forbidden = {"cmd_vel", "waypoint", "primitive", "target_pose", "trajectory"}
    for task in tasks["tasks"]:
        for subgoal in task["oracle_plan"]:
            assert not forbidden.intersection(subgoal)


def test_v19_profile_uses_local_marker_goals_and_repairs_find_verify_gap():
    tasks, scenes = materialize_semantic_navigation_runtime(
        load_task_set(TASK_FILE),
        profile=V19_PROFILE,
    )
    comp_04 = next(task for task in tasks["tasks"] if task["task_id"] == "comp_04")
    sequence = [subgoal["subgoal_type"] for subgoal in comp_04["oracle_plan"]]

    assert sequence == ["enter", "pass", "enter", "find", "approach", "verify"]
    assert comp_04["judge"]["required_sequence"] == sequence
    assert all(subgoal["target"].startswith("nearest ") for subgoal in comp_04["oracle_plan"])
    assert all(subgoal["relation"] == "" for subgoal in comp_04["oracle_plan"])
    assert comp_04["semantic_runtime"]["judge_completion_policy"] == "route_progress_v2"
    assert scenes["runtime_profile"] == V19_PROFILE

    comp_02 = next(task for task in tasks["tasks"] if task["task_id"] == "comp_02")
    assert comp_02["oracle_plan"][-1]["subgoal_type"] == "verify"
    assert [obj["semantic_marker_label"] for obj in next(
        scene for scene in scenes["scenes"] if scene["scene_id"] == comp_02["semantic_runtime"]["scene_id"]
    )["objects"]][:3] == ["red fire extinguisher", "blue box", "red traffic cone"]
