from omninav_step_scheduler.omninav_model_client_node import waypoint_tensor_to_action
from omninav_step_scheduler.schemas import attach_timebase, parse_omninav_action


def test_raw_forward_waypoint_maps_to_move_forward():
    action = waypoint_tensor_to_action([[[0.45, 0.0]]], arrive_pred=0.05)

    assert action["primitive"] == "move_forward"
    assert action["distance_m"] > 0.0
    assert action["raw_waypoint"][0] > 0.0
    assert action["fallback_used"] is False


def test_raw_positive_x_low_yaw_maps_to_move_forward():
    action = waypoint_tensor_to_action([[[0.30, 0.03]]], arrive_pred=0.05)

    assert action["primitive"] == "move_forward"
    assert action["parser_reason"] == "forward_waypoint_low_yaw"


def test_y_axis_forward_waypoint_maps_to_move_forward():
    action = waypoint_tensor_to_action(
        [[[0.05, 0.18]]],
        arrive_pred=0.05,
        config={"waypoint_forward_axis": "y", "forward_yaw_threshold_deg": 20.0},
    )

    assert action["primitive"] == "move_forward"
    assert action["waypoint_forward_axis"] == "y"
    assert abs(action["local_forward_m"] - 0.18) < 1e-9
    assert abs(action["local_lateral_m"] - 0.05) < 1e-9
    assert 0.0 < action["yaw_deg"] < 20.0


def test_raw_left_waypoint_maps_to_turn_left():
    action = waypoint_tensor_to_action([[[0.02, 0.35]]], arrive_pred=0.05)

    assert action["primitive"] == "turn_left"


def test_invalid_waypoint_defaults_to_stop_not_turn_left():
    action = waypoint_tensor_to_action([], arrive_pred=None)

    assert action["primitive"] == "stop"
    assert action["primitive"] != "turn_left"
    assert action["fallback_used"] is True
    assert action["fallback_reason"] == "no_waypoint"


def test_parser_diagnostic_fields_survive_action_schema():
    payload = {
        "request_id": "req1",
        "timestamp_request": 1.0,
        "timestamp_response": 1.1,
        "frame_timestamp": 1.0,
        "pose_at_snapshot": [0.0, 0.0, 0.0],
        "primitive": "move_forward",
        "distance_m": 0.3,
        "yaw_deg": 0.0,
        "confidence": 0.8,
        "raw_text": "wp=(0.3,0.0)",
        "raw_waypoint": [0.3, 0.0],
        "raw_action": "move_forward",
        "parser_reason": "forward_waypoint_low_yaw",
    }
    action = parse_omninav_action(attach_timebase(payload, episode_id="ep1", request_id="req1", clock_domain="wall"))

    assert action.raw_waypoint == [0.3, 0.0]
    assert action.raw_action == "move_forward"
    assert action.parser_reason == "forward_waypoint_low_yaw"
