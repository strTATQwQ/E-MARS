import json

from isaac_vln_benchmark.v11_value_utils import BASELINE_MODE, ORACLE_MODE, evaluate_v11


def _episodes():
    tasks = []
    for task_type, count in (("simple_navigation", 6), ("turn_choice", 6), ("semantic_target", 3)):
        for index in range(count):
            tasks.append((f"{task_type}_{index}", task_type))
    rows = []
    for mode in (BASELINE_MODE, ORACLE_MODE):
        for index, (task_id, task_type) in enumerate(tasks):
            success = index < 6
            clean = index < 4
            if mode == ORACLE_MODE:
                success = index < 6 or index in {6, 12}
                clean = index < 6
            rows.append(
                {
                    "episode_id": f"{mode}_{task_id}",
                    "mode": mode,
                    "task_id": task_id,
                    "task_type": task_type,
                    "success": success,
                    "clean_success": clean,
                    "mission_time_sec": 10.0,
                    "path_length_m": 2.0,
                    "num_collisions": 0,
                }
            )
    return rows


def test_v11_requires_two_matched_success_and_clean_gains(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"episodes": _episodes()}), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    result = evaluate_v11(tmp_path)
    assert result["pass"] is True
    assert result["success_delta"] == 2
    assert result["clean_delta"] == 2
    assert result["effect_counts"]["helped"] == 2
    assert result["oracle_turn_choice_success"] == 1
    assert result["oracle_semantic_target_success"] == 1


def test_v11_rejects_missing_semantic_gain(tmp_path):
    rows = _episodes()
    semantic = next(row for row in rows if row["mode"] == ORACLE_MODE and row["task_type"] == "semantic_target")
    semantic["success"] = False
    (tmp_path / "metrics.json").write_text(json.dumps({"episodes": rows}), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    result = evaluate_v11(tmp_path)
    assert result["pass"] is False
    assert "semantic_target success <= 0" in result["failures"]
