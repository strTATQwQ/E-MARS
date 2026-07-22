from omninav_step_scheduler.schemas import parse_omninav_action
from omninav_step_scheduler.stale_gate import evaluate_omninav_action


def test_missing_v5_timebase_is_attributed_missing_timestamp():
    action = parse_omninav_action(
        {
            "request_id": "legacy_req",
            "timestamp_request": 1.0,
            "timestamp_response": 1.0,
            "frame_timestamp": 1.0,
            "pose_at_snapshot": [0.0, 0.0, 0.0],
            "primitive": "move_forward",
            "distance_m": 0.3,
            "yaw_deg": 0.0,
            "confidence": 0.9,
        }
    )

    decision = evaluate_omninav_action(action, [0.0, 0.0, 0.0], 1.1, current_episode_id="ep1")

    assert not decision.valid
    assert decision.attribution == "missing_timestamp"
