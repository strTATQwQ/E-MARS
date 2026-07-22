import json

from isaac_vln_benchmark.public_semantic_heuristic_node import public_instruction_plan
from isaac_vln_benchmark.semantic_value_screening import SCREEN_TASK_IDS, materialize_v20_screen_tasks


def test_v20_materializer_selects_balanced_explicit_public_tasks(tmp_path):
    tasks = []
    for task_id in SCREEN_TASK_IDS:
        tasks.append(
            {
                "task_id": task_id,
                "task_type": "semantic_navigation",
                "category": "referential" if task_id.startswith("ref") else (
                    "compositional" if task_id.startswith("comp") else "recovery"
                ),
                "instruction": "hidden original wording",
                "oracle_plan": [
                    {
                        "subgoal_type": "find",
                        "target": "nearest red fire extinguisher marker",
                        "recovery": "scan",
                    },
                    {
                        "subgoal_type": "verify",
                        "target": "nearest red fire extinguisher marker",
                        "recovery": "stop",
                    },
                ],
            }
        )
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")

    result = materialize_v20_screen_tasks(source, tmp_path / "screen.json")

    assert len(result["tasks"]) == 15
    assert {task["category"] for task in result["tasks"]} == {"referential", "compositional", "recovery"}
    assert all("numbered stage" in task["instruction"] for task in result["tasks"])
    assert all(task["v20_screen_scope"]["oracle_plan_is_judge_only"] for task in result["tasks"])
    parsed = public_instruction_plan(result["tasks"][0]["instruction"])
    assert [row["subgoal_type"] for row in parsed] == ["find", "verify"]
