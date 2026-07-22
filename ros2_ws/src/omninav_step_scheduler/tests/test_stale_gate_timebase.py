from omninav_step_scheduler.schemas import attach_timebase, parse_omninav_action, parse_step_plan
from omninav_step_scheduler.stale_gate import DEFAULT_CONFIG, evaluate_omninav_action, evaluate_role_decision, evaluate_step_plan


def test_omninav_action_with_v5_timebase_is_valid():
    payload = attach_timebase(
        {
            "request_id": "omni_1",
            "timestamp_request": 100.0,
            "timestamp_response": 100.0,
            "frame_timestamp": 100.0,
            "pose_at_snapshot": [0.0, 0.0, 0.0],
            "primitive": "move_forward",
            "distance_m": 0.3,
            "yaw_deg": 0.0,
            "confidence": 0.9,
            "ttl_sec": 0.5,
        },
        episode_id="ep1",
        request_id="omni_1",
        clock_domain="wall",
        source_stamp=100.0,
        created_ros_time=100.0,
        created_wall_time=100.0,
    )
    action = parse_omninav_action(payload)

    decision = evaluate_omninav_action(action, [0.0, 0.0, 0.0], 100.1, DEFAULT_CONFIG, current_episode_id="ep1")

    assert decision.valid
    assert decision.attribution == "valid"


def test_step_plan_age_stale_gets_specific_attribution():
    payload = attach_timebase(
        {
            "request_id": "step_1",
            "timestamp_request": 10.0,
            "timestamp_response": 10.0,
            "multimodal": False,
            "pose_at_request": [0.0, 0.0, 0.0],
            "navila_or_omninav_instruction": "go",
            "subgoal": "go",
            "success_condition": "arrive",
            "constraints": {},
            "recommended_pending_mode": "stop",
            "confidence": 0.8,
        },
        episode_id="ep1",
        request_id="step_1",
        clock_domain="wall",
        source_stamp=10.0,
        created_ros_time=10.0,
        created_wall_time=10.0,
    )
    plan = parse_step_plan(payload)

    decision = evaluate_step_plan(plan, [0.0, 0.0, 0.0], 20.0, DEFAULT_CONFIG, current_episode_id="ep1")

    assert not decision.valid
    assert decision.attribution == "stale_due_to_age"


def test_role_decision_delay_is_rejected_before_primitive_bridge():
    payload = attach_timebase(
        {"route_choice": "left", "confidence": 1.0, "request_id": "route_1"},
        episode_id="ep1",
        request_id="route_1",
        clock_domain="wall",
        source_stamp=10.0,
        created_ros_time=10.0,
        created_wall_time=10.0,
    )

    decision = evaluate_role_decision(
        payload,
        14.0,
        {"role_decision_ttl_sec": 2.0},
        current_episode_id="ep1",
        current_clock_domain="wall",
    )

    assert decision.discard
    assert decision.attribution == "stale_due_to_age"


def test_role_decision_from_retired_episode_is_reset_cleanup():
    payload = attach_timebase(
        {"stop": True, "request_id": "stop_1"},
        episode_id="retired",
        request_id="stop_1",
        clock_domain="wall",
        source_stamp=20.0,
        created_ros_time=20.0,
        created_wall_time=20.0,
    )

    decision = evaluate_role_decision(
        payload,
        20.1,
        current_episode_id="current",
        current_clock_domain="wall",
        old_response_after_reset=True,
    )

    assert decision.discard
    assert decision.attribution == "old_response_after_reset"


def test_small_cross_host_clock_skew_is_clamped_to_zero_age():
    payload = attach_timebase(
        {
            "request_id": "omni_skew",
            "timestamp_request": 100.0,
            "timestamp_response": 100.05,
            "frame_timestamp": 100.0,
            "pose_at_snapshot": [0.0, 0.0, 0.0],
            "primitive": "move_forward",
            "distance_m": 0.3,
            "yaw_deg": 0.0,
            "confidence": 0.9,
            "ttl_sec": 0.5,
        },
        episode_id="ep1",
        request_id="omni_skew",
        clock_domain="ros_system",
        source_stamp=100.05,
        created_ros_time=100.05,
        created_wall_time=100.05,
    )
    action = parse_omninav_action(payload)

    decision = evaluate_omninav_action(
        action,
        [0.0, 0.0, 0.0],
        100.0,
        DEFAULT_CONFIG | {"max_clock_skew_sec": 0.25},
        current_episode_id="ep1",
        current_clock_domain="ros_system",
    )

    assert decision.valid
    assert decision.attribution == "valid"
    assert decision.age_sec == 0.0
    assert 0.049 <= (decision.clock_skew_sec or 0.0) <= 0.051


def test_large_negative_age_remains_timebase_error():
    payload = attach_timebase(
        {
            "request_id": "omni_bad_skew",
            "timestamp_request": 100.0,
            "timestamp_response": 100.5,
            "frame_timestamp": 100.0,
            "pose_at_snapshot": [0.0, 0.0, 0.0],
            "primitive": "move_forward",
            "distance_m": 0.3,
            "yaw_deg": 0.0,
            "confidence": 0.9,
            "ttl_sec": 0.5,
        },
        episode_id="ep1",
        request_id="omni_bad_skew",
        clock_domain="ros_system",
        source_stamp=100.5,
        created_ros_time=100.5,
        created_wall_time=100.5,
    )
    action = parse_omninav_action(payload)

    decision = evaluate_omninav_action(
        action,
        [0.0, 0.0, 0.0],
        100.0,
        DEFAULT_CONFIG | {"max_clock_skew_sec": 0.25},
        current_episode_id="ep1",
        current_clock_domain="ros_system",
    )

    assert not decision.valid
    assert decision.attribution == "timebase_error"
