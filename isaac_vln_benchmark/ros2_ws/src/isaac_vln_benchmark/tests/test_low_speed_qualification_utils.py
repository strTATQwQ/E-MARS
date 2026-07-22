import json

from isaac_vln_benchmark.low_speed_qualification_utils import evaluate_low_speed_probe


def test_low_speed_probe_rejects_commanded_motion_without_actual_progress(tmp_path):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"episodes": [{"success": False, "failure_reason": "timeout", "path_length_m": 0.5, "final_distance_to_target_m": 6.0}]}),
        encoding="utf-8",
    )
    rows = []
    for index in range(10):
        rows.append({"event": "safe_cmd_vel", "t": index + 10, "details": {"linear": {"x": 0.2}}})
        rows.append({"event": "isaac_ground_truth_pose", "t": index + 10, "details": {"pose": [0.01 * index, 0.0], "linear_velocity": [0.01, 0.0]}})
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    config = tmp_path / "scheduler.yaml"
    config.write_text("primitive:\n  max_linear_x: 0.20\n  max_yaw_rate: 0.30\n  accel_limit_mps2: 0.08\n", encoding="utf-8")

    result = evaluate_low_speed_probe(tmp_path, config)

    assert result["pass"] is False
    assert result["blocker"] == "isaac_locomotion_policy_low_speed_stall"
    assert result["safe_cmd_max_abs_mps"] == 0.2


def test_low_speed_probe_accepts_tracking_success(tmp_path):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"episodes": [{"success": True, "failure_reason": None, "path_length_m": 5.0, "final_distance_to_target_m": 1.8}]}),
        encoding="utf-8",
    )
    rows = []
    for index in range(10):
        rows.append({"event": "safe_cmd_vel", "t": index + 10, "details": {"linear": {"x": 0.2}}})
        rows.append({"event": "isaac_ground_truth_pose", "t": index + 10, "details": {"pose": [0.15 * index, 0.0], "linear_velocity": [0.15, 0.0]}})
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    config = tmp_path / "scheduler.yaml"
    config.write_text("primitive:\n  max_linear_x: 0.20\n  max_yaw_rate: 0.30\n  accel_limit_mps2: 0.08\n", encoding="utf-8")

    result = evaluate_low_speed_probe(tmp_path, config)

    assert result["pass"] is True
    assert result["blocker"] == "none"


def test_low_speed_probe_rejects_actual_overspeed(tmp_path):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"episodes": [{"success": True, "failure_reason": None, "path_length_m": 5.0, "final_distance_to_target_m": 1.8}]}),
        encoding="utf-8",
    )
    rows = []
    for index in range(10):
        rows.append({"event": "safe_cmd_vel", "t": index + 10, "details": {"linear": {"x": 0.2}}})
        rows.append({"event": "isaac_ground_truth_pose", "t": index + 10, "details": {"pose": [0.15 * index, 0.0], "linear_velocity": [0.22, 0.0]}})
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    config = tmp_path / "scheduler.yaml"
    config.write_text("primitive:\n  max_linear_x: 0.20\n  max_yaw_rate: 0.30\n  accel_limit_mps2: 0.08\n", encoding="utf-8")

    result = evaluate_low_speed_probe(tmp_path, config)

    assert result["pass"] is False
    assert result["blocker"] == "actual_speed_limit_exceeded"
