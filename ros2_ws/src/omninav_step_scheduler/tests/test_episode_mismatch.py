from omninav_step_scheduler.schemas import attach_timebase, parse_omninav_action
from omninav_step_scheduler.stale_gate import evaluate_omninav_action


def test_episode_mismatch_is_not_counted_as_age_stale():
    action = parse_omninav_action(
        attach_timebase(
            {
                "request_id": "req1",
                "timestamp_request": 1.0,
                "timestamp_response": 1.0,
                "frame_timestamp": 1.0,
                "pose_at_snapshot": [0.0, 0.0, 0.0],
                "primitive": "move_forward",
                "distance_m": 0.3,
                "yaw_deg": 0.0,
                "confidence": 0.9,
            },
            episode_id="ep_old",
            request_id="req1",
            clock_domain="wall",
            source_stamp=1.0,
            created_ros_time=1.0,
            created_wall_time=1.0,
        )
    )

    decision = evaluate_omninav_action(action, [0.0, 0.0, 0.0], 1.1, current_episode_id="ep_new")

    assert not decision.valid
    assert decision.attribution == "episode_mismatch"
