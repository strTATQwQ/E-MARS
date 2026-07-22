import json

from isaac_vln_benchmark.step_semantic_quality import (
    PUBLIC_INSTRUCTIONS,
    evaluate_step_semantic_quality,
    materialize_public_marker_tasks,
)


def test_materialized_tasks_replace_hidden_semantics_with_public_marker_language(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": task_id,
                        "instruction": "hidden semantic instruction",
                        "oracle_plan": [{"subgoal_type": "find", "target": "nearest red marker", "recovery": "scan"}],
                    }
                    for task_id in PUBLIC_INSTRUCTIONS
                ]
            }
        ),
        encoding="utf-8",
    )
    result = materialize_public_marker_tasks(source, tmp_path / "out.json")
    assert len(result["tasks"]) == 6
    assert all(task["instruction"] != "hidden semantic instruction" for task in result["tasks"])
    assert all(task["step_quality_scope"]["oracle_plan_is_judge_only"] for task in result["tasks"])


def test_materializer_removes_only_redundant_visible_find_and_reindexes_runtime(tmp_path):
    tasks = []
    for task_id in PUBLIC_INSTRUCTIONS:
        plan = [
            {"subgoal_type": "pass", "target": "red marker", "recovery": "scan"},
            {"subgoal_type": "enter", "target": "blue marker", "recovery": "backtrack"},
            {"subgoal_type": "find", "target": "green marker", "recovery": "scan"},
            {"subgoal_type": "approach", "target": "green marker", "recovery": "scan"},
            {"subgoal_type": "verify", "target": "green marker", "recovery": "stop"},
        ]
        if task_id == "comp_07":
            plan.insert(1, {"subgoal_type": "pass", "target": "blue marker", "recovery": "scan"})
        tasks.append(
            {
                "task_id": task_id,
                "oracle_plan": plan,
                "semantic_runtime": {
                    "subgoal_object_ids": {str(i): f"object-{i}" for i in range(len(plan))},
                    "execution_targets": {str(i): f"target-{i}" for i in range(len(plan))},
                },
            }
        )
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    result = materialize_public_marker_tasks(source, tmp_path / "out.json")
    by_id = {task["task_id"]: task for task in result["tasks"]}
    assert [row["subgoal_type"] for row in by_id["comp_01"]["oracle_plan"]] == [
        "pass", "enter", "approach", "verify"
    ]
    assert by_id["comp_01"]["semantic_runtime"]["subgoal_object_ids"] == {
        "0": "object-0", "1": "object-1", "2": "object-3", "3": "object-4"
    }
    assert [row["subgoal_type"] for row in by_id["comp_07"]["oracle_plan"]] == [
        "pass", "pass", "enter", "approach", "verify"
    ]


def test_semantic_quality_gate_scores_strict_fresh_multimodal_sequence(tmp_path):
    task = {
        "task_id": "ref_01",
        "oracle_plan": [
            {
                "subgoal_type": "find",
                "target": "nearest red fire extinguisher marker",
                "relation": "",
                "recovery": "scan",
            }
        ],
    }
    (tmp_path / "selected_tasks.yaml").write_text(json.dumps({"tasks": [task]}), encoding="utf-8")
    episode = {
        "task_id": "ref_01",
        "mode": "omninav_step_semantic_executive",
        "success": True,
        "num_collisions": 0,
        "runtime_stale": 0,
        "timebase_error": 0,
        "stale_action_executed": 0,
    }
    (tmp_path / "metrics.json").write_text(json.dumps({"episodes": [episode]}), encoding="utf-8")
    episode_dir = tmp_path / "omninav_step_semantic_executive" / "ref_01"
    episode_dir.mkdir(parents=True)
    events = [
        {"event": "step_request", "details": {"instruction": PUBLIC_INSTRUCTIONS["ref_01"]}},
        {
            "event": "metrics_event_jsonl",
            "details": {
                "event_type": "step_http_response",
                "role": "semantic_executive",
                "result": "accepted",
                "multimodal": True,
                "latency_s": 6.5,
                "image_snapshot": {"frame_seq": 8, "age_sec": 0.1, "source": "primary"},
                "output": {
                    "subgoal_type": "find",
                    "target": "red fire extinguisher marker",
                    "relation": "",
                    "recovery": "scan",
                },
            },
        },
    ]
    (episode_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(row) for row in events) + "\n", encoding="utf-8"
    )
    gate = evaluate_step_semantic_quality(tmp_path)
    assert gate["pass"] is True
    assert gate["latency_gate_sec"] == 7.0
    assert gate["type_accuracy"] == 1.0
    assert gate["oracle_leakage_findings"] == []


def test_semantic_quality_gate_rejects_oracle_request_context(tmp_path):
    (tmp_path / "selected_tasks.yaml").write_text(json.dumps({"tasks": []}), encoding="utf-8")
    (tmp_path / "metrics.json").write_text(json.dumps({"episodes": []}), encoding="utf-8")
    assert evaluate_step_semantic_quality(tmp_path)["pass"] is False


def test_semantic_quality_gate_uses_frozen_closed_loop_minimum(tmp_path):
    (tmp_path / "config.yaml").write_text(
        json.dumps({"gates": {"closed_loop_success_min": 3, "step_latency_p95_max_sec": 7.0}}),
        encoding="utf-8",
    )
    (tmp_path / "selected_tasks.yaml").write_text(json.dumps({"tasks": []}), encoding="utf-8")
    episodes = [
        {
            "task_id": f"task-{index}",
            "mode": "omninav_step_semantic_executive",
            "success": index < 2,
        }
        for index in range(3)
    ]
    (tmp_path / "metrics.json").write_text(json.dumps({"episodes": episodes}), encoding="utf-8")

    gate = evaluate_step_semantic_quality(tmp_path)

    assert gate["closed_loop_success"] == 2
    assert gate["closed_loop_success_min"] == 3
    assert gate["gates"]["closed_loop"] is False
