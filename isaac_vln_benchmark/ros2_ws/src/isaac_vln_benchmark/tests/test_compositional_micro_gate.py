from isaac_vln_benchmark.compositional_micro_gate import (
    evaluate_v19_compositional_full,
    evaluate_v19_compositional_micro,
)


def task_set():
    tasks = []
    for parent in ("comp_01", "comp_02"):
        for seed in range(3):
            tasks.append(
                {
                    "task_id": f"{parent}_{seed}",
                    "semantic_runtime": {"parent_task_id": parent},
                }
            )
    return {"micro_stage": "single", "runtime_profile": "v19_local_marker_v2", "tasks": tasks}


def episode(task_id, success=True):
    return {
        "task_id": task_id,
        "success": success,
        "clean_success": success,
        "semantic_errors": 0,
        "num_collisions": 0,
        "runtime_stale": 0,
        "timebase_error": 0,
        "stale_action_executed": 0,
        "failure_reason": "" if success else "timeout",
    }


def test_micro_gate_requires_rate_parent_coverage_model_evidence_and_safety():
    tasks = task_set()
    metrics = {"episodes": [episode(task["task_id"]) for task in tasks["tasks"]]}
    gate = evaluate_v19_compositional_micro(
        metrics,
        tasks,
        {"pass": True, "real_image_calls": 30, "fallback_calls": 0},
        {"success_rate_min": 0.90, "parent_success_min": 2, "parent_episodes": 3},
    )

    assert gate["pass"] is True
    assert gate["formal_compositional_gate_unlocked"] is False
    assert gate["step_screening_allowed"] is False


def test_micro_gate_fails_when_one_parent_has_fewer_than_two_successes():
    tasks = task_set()
    episodes = [episode(task["task_id"]) for task in tasks["tasks"]]
    episodes[0]["success"] = False
    episodes[1]["success"] = False
    gate = evaluate_v19_compositional_micro(
        {"episodes": episodes},
        tasks,
        {"pass": True, "real_image_calls": 30, "fallback_calls": 0},
        {"success_rate_min": 0.60, "parent_success_min": 2, "parent_episodes": 3},
    )

    assert gate["parent_gate"]["comp_01"] is False
    assert gate["pass"] is False


def test_full_compositional_gate_requires_eight_of_ten_and_model_evidence():
    tasks = {
        "tasks": [
            {
                "task_id": f"comp_{index:02d}",
                "semantic_runtime": {"runtime_profile": "v19_local_marker_v2"},
            }
            for index in range(1, 11)
        ]
    }
    episodes = [episode(task["task_id"], success=index < 8) for index, task in enumerate(tasks["tasks"])]

    gate = evaluate_v19_compositional_full(
        {"episodes": episodes},
        tasks,
        {"pass": True, "real_image_calls": 100, "fallback_calls": 0},
        {"success_min": 8},
    )

    assert gate["pass"] is True
    assert gate["forced_oracle_full30_retry_allowed"] is True
    assert gate["step_screening_allowed"] is False
