import json

from isaac_vln_benchmark.v8_coverage_utils import audit_target_success_judge, canonical_target_id


def test_target_alias_audit_maps_requested_aliases_and_finds_targets(tmp_path):
    tasks = {
        "tasks": [
            {
                "task_id": "semantic_001",
                "task_type": "semantic_target",
                "scene_type": "corridor",
                "target_object": "fire_extinguisher_1",
            }
        ]
    }
    scenes = {
        "scenes": [
            {
                "scene_id": "corridor_001",
                "scene_type": "corridor",
                "objects": [{"id": "fire_extinguisher_1", "pose": [1, 0, 0]}],
            }
        ]
    }
    tasks_path = tmp_path / "tasks.yaml"
    scenes_path = tmp_path / "scenes.yaml"
    tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
    scenes_path.write_text(json.dumps(scenes), encoding="utf-8")

    result = audit_target_success_judge(tasks_path, scenes_path, tmp_path)

    assert result["pass"] is True
    assert canonical_target_id("fire_extinguisher_near_exit") == "fire_extinguisher"
    assert canonical_target_id("red_exit_sign") == "exit_sign"
    assert (tmp_path / "target_success_judge_audit.md").exists()
