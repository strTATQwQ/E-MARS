import json

from isaac_vln_benchmark.sim2real_session_utils import evaluate_sim2real_session, write_sim2real_session


def _write_run(tmp_path, *, actual_speed=0.15, success=True, robot_z=0.33):
    episode_id = "mode_semantic_001_seed0_0000"
    (tmp_path / "metrics.json").write_text(
        json.dumps({"episodes": [{"episode_id": episode_id, "task_type": "semantic_target", "success": success}]}),
        encoding="utf-8",
    )
    rows = [
        {"event": "safe_cmd_vel", "episode_id": episode_id, "t": 10.0, "details": {"linear": {"x": 0.2}}},
        {"event": "isaac_ground_truth_pose", "episode_id": episode_id, "t": 10.1, "details": {"z": robot_z, "linear_velocity": [actual_speed, 0.0], "angular_velocity": [0.0, 0.0, 0.1], "command": {"low_speed_servo_enabled": True, "low_speed_servo_active": True, "vx": 0.45, "low_speed_yaw_servo_enabled": True, "low_speed_yaw_servo_active": True, "wz": 0.30}}},
        {"event": "step_request_json", "episode_id": episode_id, "t": 20.0, "details": {"role": "semantic_stop"}},
        {"event": "metrics_event_jsonl", "episode_id": episode_id, "t": 21.2, "details": {"event_type": "step_http_response", "model": "step_http", "result": "accepted", "role": "semantic_stop", "latency_s": 1.2}},
        {"event": "metrics_event_jsonl", "episode_id": episode_id, "t": 21.21, "details": {"event_type": "primitive_command", "action_type": "stop"}},
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_session_passes_with_exact_real_step_chain_and_bounded_speed(tmp_path):
    _write_run(tmp_path)

    result = evaluate_sim2real_session(tmp_path)

    assert result["pass"] is True
    assert result["semantic_request_to_stop_p95_sec"] == 1.21


def test_session_rejects_actual_overspeed(tmp_path):
    _write_run(tmp_path, actual_speed=0.22)

    result = evaluate_sim2real_session(tmp_path)

    assert result["pass"] is False
    assert "actual speed p95 exceeded 0.20 m/s" in result["failures"]


def test_session_detects_fall_from_ground_truth_body_height(tmp_path):
    _write_run(tmp_path, robot_z=0.057)

    result = evaluate_sim2real_session(tmp_path)

    assert result["pass"] is False
    assert result["fall_count"] == 1


def test_fast_fall_is_counted_and_missing_post_warmup_telemetry_is_reportable(tmp_path):
    episode_id = "mode_semantic_001_seed0_0000"
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_id": episode_id,
                        "task_type": "semantic_target",
                        "success": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "isaac_ground_truth_pose",
                "episode_id": episode_id,
                "t": 4.9,
                "details": {
                    "z": 0.179,
                    "linear_velocity": [0.1, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.1],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_sim2real_session(tmp_path)
    write_sim2real_session(tmp_path, result)

    assert result["fall_count"] == 1
    assert result["actual_speed_p95_mps"] is None
    assert "no post-warmup actual speed telemetry" in result["failures"]
    assert "n/a" in (tmp_path / "session_qualification.md").read_text(encoding="utf-8")
