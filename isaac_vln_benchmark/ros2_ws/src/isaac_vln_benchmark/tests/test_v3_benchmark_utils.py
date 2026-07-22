from isaac_vln_benchmark.v3_benchmark_utils import (
    evaluate_sim2real_readiness,
    judge_route_choice_episode,
    judge_semantic_stop_episode,
)


def test_route_choice_success_judge():
    judged = judge_route_choice_episode(
        {
            "expected_route": "left",
            "first_turn": "left",
            "route_choice_output": "left",
            "yaw_after_5s": 35.0,
            "yaw_after_10s": 70.0,
        }
    )
    assert judged["entered_correct_branch"] is True
    assert judged["failure_reason"] == "none"


def test_semantic_stop_success_judge():
    judged = judge_semantic_stop_episode(
        {
            "stop": True,
            "target_visible": True,
            "distance_at_stop": 2.0,
            "visible_to_stop_latency_sec": 1.2,
            "max_distance_to_target_m": 2.5,
        }
    )
    assert judged["stop_decision_accuracy"] is True
    assert judged["failure_reason"] == "none"


def test_sim2real_readiness_gate_not_ready_on_missing_route():
    result = evaluate_sim2real_readiness(
        {
            "route_choice_correct_branch_rate": 0.50,
            "semantic_stop_accuracy": 0.90,
            "visible_to_stop_latency_sec": 1.0,
            "collision_count": 0,
            "stale_action_executed": 0,
            "parse_error": 0,
            "max_linear_x_mps": 0.20,
            "actions_through_safe_mux": True,
            "real_robot_motion_enabled": False,
        }
    )
    assert result["ready"] is False
    assert "NOT READY" in result["status"]


def test_sim2real_readiness_gate_ready_for_sensor_only():
    result = evaluate_sim2real_readiness(
        {
            "route_choice_correct_branch_rate": 0.80,
            "semantic_stop_accuracy": 0.90,
            "visible_to_stop_latency_sec": 1.0,
            "collision_count": 0,
            "stale_action_executed": 0,
            "parse_error": 0,
            "max_linear_x_mps": 0.20,
            "actions_through_safe_mux": True,
            "real_robot_motion_enabled": False,
        }
    )
    assert result["ready"] is True
    assert "SENSOR-ONLY" in result["status"]


def test_sim2real_readiness_gate_not_ready_on_stale_discards():
    result = evaluate_sim2real_readiness(
        {
            "route_choice_correct_branch_rate": 0.80,
            "semantic_stop_accuracy": 0.90,
            "visible_to_stop_latency_sec": 1.0,
            "collision_count": 0,
            "stale_action_executed": 0,
            "stale_discard_count": 3,
            "parse_error": 0,
            "max_linear_x_mps": 0.20,
            "actions_through_safe_mux": True,
            "real_robot_motion_enabled": False,
        }
    )
    assert result["ready"] is False
    assert any("stale_discard_count" in failure for failure in result["failures"])
