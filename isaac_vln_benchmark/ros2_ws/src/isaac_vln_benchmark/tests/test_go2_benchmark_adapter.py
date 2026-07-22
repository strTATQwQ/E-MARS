import pytest

from isaac_vln_benchmark.go2_benchmark_adapter_node import (
    accepted_sensor_semantic_step_stop,
    TARGET_HOLD_PHASES,
    accepted_semantic_step_stop,
    branch_pre_entry_reached,
    branch_verify_ready,
    isaac_twist_payload,
    bypass_released,
    bypass_waypoint,
    choose_bypass_side,
    normalize_isaac_pose,
    root_height_unstable,
    simple_goal_stop_should_latch,
    step_payload_matches_episode,
    target_max_yaw_rate,
    target_crawl_max_yaw_error_deg,
    target_phase_timeout_sec,
    target_yaw_polarity_adaptation_enabled,
)


def test_sensor_stop_judge_requires_step_plus_confirmed_actual_sensor_fusion():
    payload = {
        "stop": True,
        "target_visible": True,
        "estimated_distance_ok": True,
        "track": {
            "confirmed": True,
            "fresh": True,
            "visible": True,
            "source": "actual_sensor_spatial_track",
            "fusion_source": "two_frame_grounded_sam_plus_step_confirmation",
        },
    }
    assert accepted_sensor_semantic_step_stop(payload)
    assert not accepted_sensor_semantic_step_stop(payload | {"track": payload["track"] | {"source": "oracle"}})


def test_semantic_coverage_wait_phase_holds_motion_until_track_confirmation():
    assert "await_step_track" in TARGET_HOLD_PHASES
    assert "approach" not in TARGET_HOLD_PHASES


def test_isaac_twist_payload_matches_udp_receiver_schema():
    payload = isaac_twist_payload(0.2, -0.01, 0.15, seq=7, timestamp=123.5)

    assert payload["vx"] == 0.2
    assert payload["vy"] == -0.01
    assert payload["wz"] == 0.15
    assert payload["seq"] == 7
    assert payload["timestamp"] == 123.5
    assert payload["source"] == "go2_benchmark_adapter"


def test_isaac_heading_is_normalized_to_ros_world_yaw():
    pose = normalize_isaac_pose([1.0, -2.0, 0.4])
    assert pose.as_list() == [1.0, -2.0, 0.4]
    assert normalize_isaac_pose([1.0, 2.0, 0.4], heading_sign=-1.0).yaw == -0.4
    with pytest.raises(ValueError):
        normalize_isaac_pose([1.0, 2.0])


def test_root_height_fall_gate():
    assert root_height_unstable(0.057) is True
    assert root_height_unstable(0.30) is False
    assert root_height_unstable(None) is False


def test_step_controller_decisions_are_episode_scoped_and_strict():
    assert step_payload_matches_episode({"episode_id": "ep-1"}, "ep-1")
    assert not step_payload_matches_episode({"episode_id": "ep-old"}, "ep-1")
    assert step_payload_matches_episode({}, "ep-1")
    assert accepted_semantic_step_stop(
        {"stop": True, "target_visible": True, "estimated_distance_ok": True}
    )
    assert not accepted_semantic_step_stop(
        {"stop": True, "target_visible": False, "estimated_distance_ok": True}
    )


def test_step_semantic_stop_can_require_confirmed_visible_track():
    decision = {
        "stop": True,
        "target_visible": True,
        "estimated_distance_ok": True,
        "track": {"confirmed": False, "visible": True},
    }

    assert accepted_semantic_step_stop(decision)
    assert not accepted_semantic_step_stop(decision, require_confirmed_track=True)
    decision["track"]["confirmed"] = True
    assert accepted_semantic_step_stop(decision, require_confirmed_track=True)
    assert not accepted_semantic_step_stop(
        decision,
        require_confirmed_track=True,
        actual_distance_m=2.2,
        max_actual_distance_m=2.0,
    )
    assert accepted_semantic_step_stop(
        decision,
        require_confirmed_track=True,
        actual_distance_m=1.9,
        max_actual_distance_m=2.0,
    )


def test_simple_goal_stop_uses_task_tolerance_and_visibility_contract():
    task = {
        "task_type": "simple_navigation",
        "success": {"distance_to_target_m": 2.0, "target_visible": True},
    }
    assert simple_goal_stop_should_latch(task, {"distance_to_target": 1.97, "target_visible": True})
    assert not simple_goal_stop_should_latch(task, {"distance_to_target": 2.01, "target_visible": True})
    assert not simple_goal_stop_should_latch(task, {"distance_to_target": 1.97, "target_visible": False})
    assert not simple_goal_stop_should_latch(
        {"task_type": "semantic_target", "success": {"distance_to_target_m": 2.0}},
        {"distance_to_target": 1.0, "target_visible": True},
    )


def test_low_speed_target_controller_uses_lower_cold_start_yaw_limit():
    assert target_max_yaw_rate({"target_max_yaw_rate_radps": 0.30}) == 0.30
    assert target_max_yaw_rate(
        {"target_max_yaw_rate_radps": 0.30, "low_speed_primitive_stabilizer": True}
    ) == 0.15
    assert target_phase_timeout_sec({"target_max_phase_duration_sec": 150.0}) == 150.0
    assert target_phase_timeout_sec(
        {"target_max_phase_duration_sec": 150.0, "low_speed_primitive_stabilizer": True}
    ) == 210.0
    assert target_crawl_max_yaw_error_deg({"target_crawl_max_yaw_error_deg": 60.0}) == 60.0
    assert target_crawl_max_yaw_error_deg(
        {"target_crawl_max_yaw_error_deg": 60.0, "low_speed_primitive_stabilizer": True}
    ) == 180.0


def test_target_yaw_polarity_adaptation_is_opt_in():
    assert not target_yaw_polarity_adaptation_enabled({"low_speed_primitive_stabilizer": True})
    assert target_yaw_polarity_adaptation_enabled({"target_yaw_polarity_adaptation": True})


def test_branch_verify_requires_visibility_only_when_task_requests_it():
    assert branch_verify_ready(entered_correct=True, target_visible_required=False, target_visible=False)
    assert branch_verify_ready(entered_correct=True, target_visible_required=True, target_visible=True)
    assert not branch_verify_ready(entered_correct=True, target_visible_required=True, target_visible=False)
    assert not branch_verify_ready(entered_correct=False, target_visible_required=False, target_visible=True)


def test_ros_to_isaac_yaw_contract_uses_standard_signs():
    raw_pose = normalize_isaac_pose([0.0, 0.0, 0.3], heading_sign=1.0)
    udp = isaac_twist_payload(0.2, 0.0, 1.0 * 0.25, seq=1, timestamp=1.0)
    assert raw_pose.yaw == 0.3
    assert udp["wz"] == 0.25


def test_bypass_side_is_deterministic_and_waypoint_can_release():
    assert choose_bypass_side(0.03, 0.0) == 1.0
    assert choose_bypass_side(0.8, 0.0) == -1.0
    waypoint = bypass_waypoint(
        [2.0, 0.0, 0.5],
        [4.2, 0.0, 0.5],
        1.0,
        release_after_x_m=0.8,
        margin_x_m=0.2,
        offset_y_m=1.1,
    )
    assert waypoint == [3.0, 1.1, 0.5]
    assert bypass_released(
        [2.85, 1.05, 0.0],
        [2.0, 0.0, 0.5],
        waypoint,
        1.0,
        release_after_x_m=0.8,
        clearance_y_m=0.8,
        reached_m=0.30,
    ) is True


def test_branch_rotation_waits_for_centerline_injection_window():
    pre_entry = [3.75, 0.0, 0.0]

    assert branch_pre_entry_reached([3.8, 1.0, 0.0], pre_entry, reached_m=0.3, lateral_tolerance_m=0.4) is False
    assert branch_pre_entry_reached([3.8, 0.3, 0.0], pre_entry, reached_m=0.3, lateral_tolerance_m=0.4) is True
    assert branch_pre_entry_reached([3.55, 0.0, 0.0], pre_entry, reached_m=0.3, lateral_tolerance_m=0.4) is True
