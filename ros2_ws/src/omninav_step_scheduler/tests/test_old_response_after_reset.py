from omninav_step_scheduler.schemas import attach_timebase, parse_omninav_action
from omninav_step_scheduler.stale_gate import evaluate_omninav_action


def test_old_omninav_response_after_reset_is_not_valid():
    action = parse_omninav_action(
        attach_timebase(
            {
                "request_id": "old_req",
                "timestamp_request": 1.0,
                "timestamp_response": 1.0,
                "frame_timestamp": 1.0,
                "pose_at_snapshot": [0.0, 0.0, 0.0],
                "primitive": "move_forward",
                "distance_m": 0.3,
                "yaw_deg": 0.0,
                "confidence": 0.9,
            },
            episode_id="ep1",
            request_id="old_req",
            clock_domain="wall",
            source_stamp=1.0,
            created_ros_time=1.0,
            created_wall_time=1.0,
        )
    )

    decision = evaluate_omninav_action(action, [0.0, 0.0, 0.0], 1.1, current_episode_id="ep2", old_response_after_reset=True)

    assert not decision.valid
    assert decision.attribution == "old_response_after_reset"
