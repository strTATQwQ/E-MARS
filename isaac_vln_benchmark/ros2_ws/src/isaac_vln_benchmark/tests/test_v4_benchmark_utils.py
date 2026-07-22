import math

from isaac_vln_benchmark.v4_benchmark_utils import (
    analyze_stale_events,
    evaluate_sim2real_readiness_v4,
    judge_forced_route_episode,
    judge_forced_stop_episode,
    summarize_stop_rows,
)
from isaac_vln_benchmark.config_loader import normalize_scene_to_robot_origin
from isaac_vln_benchmark.v4_live_probe import (
    _count_model_stale_discards,
    _first_accepted_step_role_response,
    _pose_matches_start,
    _seeded_task,
    _step_supervisor_mode_ack,
)


def test_forced_route_judge_accepts_correct_left_branch():
    judged = judge_forced_route_episode(
        {
            "expected_route": "left",
            "first_turn_action": "left",
            "yaw_after_10s": 0.4,
            "final_y": 0.7,
        }
    )
    assert judged["entered_correct_branch"] is True
    assert judged["failure_reason"] == "none"


def test_forced_stop_judge_latency_and_distance():
    judged = judge_forced_stop_episode(
        {
            "stop_cmd_received": True,
            "target_visible_at_stop": True,
            "distance_at_stop": 2.0,
            "visible_to_stop_latency_sec": 1.1,
        }
    )
    assert judged["stop_decision_accuracy"] is True


def test_stale_event_categorization_counts_roles():
    analysis = analyze_stale_events(
        [
            {"event": "stale_result", "details": {"model": "step", "role": "route_choice"}},
            {"event": "omninav_stale", "details": {"model": "omninav"}},
        ]
    )
    assert analysis["counts"]["stale_step_route_choice"] == 1
    assert analysis["counts"]["stale_omninav_waypoint"] == 1


def test_v4_sim2real_gate_requires_all_small_experiments():
    result = evaluate_sim2real_readiness_v4(
        {
            "forced_route_oracle_correct_branch_rate": 0.80,
            "forced_stop_oracle_accuracy": 0.95,
            "step_route_choice_correct_branch_rate": 0.30,
            "step_semantic_stop_accuracy": 0.80,
            "visible_to_stop_latency_sec": 1.2,
            "collision_count": 0,
            "stale_action_executed": 0,
            "stale_discard_count": 0,
            "parse_error": 0,
            "max_linear_x_mps": 0.20,
            "actions_through_safe_mux": True,
            "real_robot_motion_enabled": False,
        }
    )
    assert result["ready"] is False
    assert any("step_route_choice_correct_branch_rate" in failure for failure in result["failures"])


def test_live_probe_stale_counter_ignores_normal_stale_fields():
    events = [
        {"event": "metrics_event_jsonl", "details": {"event_type": "safe_cmd_mux", "stale_gate_enabled": True}},
        {"event": "metrics_event_jsonl", "details": {"event_type": "step_response_stale", "result": "discarded"}},
    ]

    assert _count_model_stale_discards(events) == 1


def test_seeded_task_preserves_adapter_task_id():
    task = _seeded_task({"task_id": "target_verify_fire_extinguisher"}, 2)

    assert task["task_id"] == "target_verify_fire_extinguisher_seed2"
    assert task["adapter_task_id"] == "target_verify_fire_extinguisher"
    assert task["seed"] == 2


def test_real_step_role_response_requires_strict_accepted_http_result():
    events = [
        {
            "t": 1.25,
            "event": "metrics_event_jsonl",
            "details": {
                "event_type": "step_http_response",
                "role": "semantic_stop",
                "result": "accepted",
                "latency_s": 1.1,
                "output": {"stop": True},
            },
        }
    ]

    response = _first_accepted_step_role_response(events, "semantic_stop")
    assert response == {
        "t": 1.25,
        "result": "accepted",
        "latency_s": 1.1,
        "output": {"stop": True},
        "request_id": "",
    }
    assert _first_accepted_step_role_response(events, "route_choice") is None


def test_reset_pose_handshake_uses_physical_xy_tolerance():
    assert _pose_matches_start([0.1, -0.2, 1.0], [0.0, 0.0, 0.0])
    assert not _pose_matches_start([0.7, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_step_mode_ack_is_scoped_to_supervisor_and_episode():
    events = [
        {
            "details": {
                "event_type": "episode_reset",
                "model": "step",
                "reset_scope": "step_supervisor",
                "episode_id": "episode_7",
            }
        }
    ]

    assert _step_supervisor_mode_ack(events, "episode_7")
    assert not _step_supervisor_mode_ack(events, "episode_8")


def test_stop_summary_gates_on_p95_not_mean_latency():
    rows = [
        {
            "stop_cmd_received": True,
            "target_visible_at_stop": True,
            "distance_at_stop": 2.0,
            "visible_to_stop_latency_sec": latency,
        }
        for latency in [1.0, 1.0, 1.0, 1.0, 1.0, 2.1]
    ]

    summary = summarize_stop_rows(rows, target_acc=0.75, target_latency_sec=2.0)
    assert summary["visible_to_stop_latency_sec"] < 2.0
    assert summary["visible_to_stop_latency_p95_sec"] == 2.1
    assert summary["pass"] is False


def test_scene_normalization_preserves_relative_geometry_at_robot_origin():
    scene = normalize_scene_to_robot_origin(
        {
            "scene_id": "intersection",
            "robot_start_pose": [3.0, 0.0, 0.0],
            "bounds": [0.0, -2.0, 8.0, 2.0],
            "objects": [{"id": "target", "pose": [5.8, 2.0, 0.5]}],
            "obstacles": [{"id": "box", "pose": [4.0, -1.0, 0.4]}],
            "semantic_zones": [{"id": "junction", "center": [4.0, 0.0]}],
            "dynamic_obstacles": [{"id": "person", "path": [[3.2, -1.0], [3.2, 1.0]]}],
        }
    )

    assert scene["robot_start_pose"] == [0.0, 0.0, 0.0]
    assert scene["objects"][0]["pose"] == [2.8, 2.0, 0.5]
    assert scene["obstacles"][0]["pose"] == [1.0, -1.0, 0.4]
    assert scene["semantic_zones"][0]["center"] == [1.0, 0.0]
    assert [[round(value, 3) for value in point] for point in scene["dynamic_obstacles"][0]["path"]] == [
        [0.2, -1.0],
        [0.2, 1.0],
    ]


def test_scene_normalization_rotates_nonzero_start_yaw_into_robot_frame():
    scene = normalize_scene_to_robot_origin(
        {
            "robot_start_pose": [1.0, 2.0, math.pi / 2.0],
            "bounds": [0.0, 0.0, 4.0, 5.0],
            "objects": [{"id": "ahead", "pose": [1.0, 4.0, 0.5]}],
            "obstacles": [],
            "semantic_zones": [{"id": "zone", "center": [0.0, 2.0]}],
            "dynamic_obstacles": [],
        }
    )

    assert scene["robot_start_pose"] == [0.0, 0.0, 0.0]
    assert [round(value, 6) for value in scene["objects"][0]["pose"][:2]] == [2.0, 0.0]
    assert [round(value, 6) for value in scene["semantic_zones"][0]["center"]] == [0.0, 1.0]
    assert [round(value, 6) for value in scene["bounds"]] == [-2.0, -3.0, 3.0, 1.0]
