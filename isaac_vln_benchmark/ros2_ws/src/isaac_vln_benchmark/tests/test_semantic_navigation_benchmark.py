import json
from collections import Counter
from pathlib import Path

from isaac_vln_benchmark.semantic_navigation_benchmark import (
    evaluate_semantic_navigation_run,
    load_task_set,
    oracle_plan_payload,
    public_task_payload,
)


TASK_FILE = Path(__file__).resolve().parents[4] / "configs" / "natural_language_navigation_v18.yaml"


def test_task_set_has_balanced_30_task_semantic_domain():
    task_set = load_task_set(TASK_FILE)
    counts = Counter(task["category"] for task in task_set["tasks"])
    assert len(task_set["tasks"]) == 30
    assert counts == {"referential": 10, "compositional": 10, "recovery": 10}


def test_live_node_can_load_a_valid_selected_subset(tmp_path):
    full = load_task_set(TASK_FILE)
    subset_path = tmp_path / "selected_tasks.yaml"
    subset_path.write_text(json.dumps({"tasks": full["tasks"][:1]}), encoding="utf-8")

    selected = load_task_set(subset_path, require_full=False)

    assert [task["task_id"] for task in selected["tasks"]] == [full["tasks"][0]["task_id"]]


def test_public_payload_cannot_expose_judge_or_oracle_plan():
    task = load_task_set(TASK_FILE)["tasks"][0]
    public = public_task_payload(task)
    assert set(public) == {"task_id", "category", "scene_template", "instruction", "timeout_sec"}
    assert "judge" not in public
    assert "oracle_plan" not in public
    assert "target_id" not in public


def test_oracle_subgoals_are_semantic_and_motion_free():
    task = load_task_set(TASK_FILE)["tasks"][0]
    subgoals = oracle_plan_payload(task, episode_id="ep-1")
    assert subgoals
    for subgoal in subgoals:
        assert subgoal["source"] == "forced_semantic_oracle"
        assert subgoal["episode_id"] == "ep-1"
        assert not any(key in subgoal for key in ("cmd_vel", "waypoint", "primitive", "target_pose"))


def test_forced_oracle_gate_requires_full_balanced_success_and_zero_safety_errors():
    task_set = load_task_set(TASK_FILE)
    records = []
    for index, task in enumerate(task_set["tasks"]):
        records.append(
            {
                "task_id": task["task_id"],
                "success": index not in {0, 10, 20},
                "clean": index not in {0, 10, 20},
                "collision": 0,
                "runtime_stale": 0,
                "timebase_error": 0,
                "stale_action_executed": 0,
                "real_omninav_calls": 1,
                "omninav_fallback_calls": 0,
            }
        )
    gate = evaluate_semantic_navigation_run(records, task_set, mode="forced_semantic_oracle")
    assert gate["success"] == 27
    assert gate["category_success"] == {"referential": 9, "compositional": 9, "recovery": 9}
    assert gate["pass"] is True
    assert gate["screening_allowed"] is True

    records[0]["collision"] = 1
    failed = evaluate_semantic_navigation_run(records, task_set, mode="forced_semantic_oracle")
    assert failed["pass"] is False
    assert failed["safety_pass"] is False

    records[0]["collision"] = 0
    records[0]["omninav_fallback_calls"] = 1
    invalid_model_evidence = evaluate_semantic_navigation_run(
        records, task_set, mode="forced_semantic_oracle"
    )
    assert invalid_model_evidence["model_evidence_gate"] is False
    assert invalid_model_evidence["pass"] is False


def test_step_mode_requires_one_fresh_real_call_per_episode():
    task_set = load_task_set(TASK_FILE)
    records = [
        {
            "task_id": task["task_id"],
            "success": True,
            "clean": True,
            "step_calls": 1,
            "fresh_multimodal_calls": 1,
        }
        for task in task_set["tasks"]
    ]
    assert evaluate_semantic_navigation_run(records, task_set, mode="step")["model_evidence_gate"] is True
    records[0]["fresh_multimodal_calls"] = 0
    assert evaluate_semantic_navigation_run(records, task_set, mode="step")["model_evidence_gate"] is False
