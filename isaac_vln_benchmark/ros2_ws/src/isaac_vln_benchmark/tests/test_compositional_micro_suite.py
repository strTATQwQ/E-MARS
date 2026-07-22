from pathlib import Path

from isaac_vln_benchmark.compositional_micro_suite import build_v19_blocker_micro_suite
from isaac_vln_benchmark.config_loader import load_data


ROOT = Path(__file__).resolve().parents[4]


def generated_runtime():
    tasks = load_data(ROOT / "configs" / "generated" / "natural_language_navigation_v19_tasks.yaml")
    scenes = load_data(ROOT / "configs" / "generated" / "natural_language_navigation_v19_scenes.yaml")
    return tasks, scenes


def test_single_suite_has_six_blockers_and_three_seeds_without_controller_truth():
    tasks, scenes = generated_runtime()
    micro_tasks, micro_scenes = build_v19_blocker_micro_suite(tasks, scenes, stage="single")

    assert len(micro_tasks["tasks"]) == 18
    assert len(micro_scenes["scenes"]) == 18
    assert {task["seed"] for task in micro_tasks["tasks"]} == {0, 1, 2}
    assert all(len(task["oracle_plan"]) == 1 for task in micro_tasks["tasks"])
    first_by_parent = {
        task["semantic_runtime"]["parent_task_id"]: task["oracle_plan"][0]["subgoal_type"]
        for task in micro_tasks["tasks"]
        if task["seed"] == 0
    }
    assert first_by_parent["comp_04"] == "approach"
    assert first_by_parent["comp_05"] == "enter"
    assert first_by_parent["comp_07"] == "pass"
    forbidden = {"target_pose", "waypoint", "cmd_vel", "primitive", "trajectory"}
    assert all(not forbidden.intersection(task["oracle_plan"][0]) for task in micro_tasks["tasks"])


def test_chain_suite_contains_repaired_comp04_find_approach_verify_sequence():
    tasks, scenes = generated_runtime()
    micro_tasks, _ = build_v19_blocker_micro_suite(tasks, scenes, stage="chain", seeds=(0,))
    comp_04 = next(task for task in micro_tasks["tasks"] if task["semantic_runtime"]["parent_task_id"] == "comp_04")

    assert [subgoal["subgoal_type"] for subgoal in comp_04["oracle_plan"]] == ["find", "approach", "verify"]
    assert comp_04["judge"]["required_sequence"] == ["find", "approach", "verify"]
    assert comp_04["semantic_runtime"]["judge_completion_policy"] == "route_progress_v2"

    comp_03 = next(task for task in micro_tasks["tasks"] if task["semantic_runtime"]["parent_task_id"] == "comp_03")
    assert comp_03["oracle_plan"][-1]["subgoal_type"] == "verify"
