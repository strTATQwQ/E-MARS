import pytest

from omninav_step_scheduler.schemas import SchemaError
from omninav_step_scheduler.semantic_recovery_bridge_node import recovery_to_primitive


def test_scan_recovery_maps_to_bounded_normal_chain_primitive():
    primitive, duration = recovery_to_primitive(
        {"recovery": "scan", "request_id": "req-1", "reason": "track_lost"},
        {"semantic_recovery": {"scan_duration_sec": 0.8, "scan_yaw_rate_radps": 0.2}},
    )

    assert primitive["primitive"] == "look_around"
    assert primitive["request_id"] == "req-1_primitive"
    assert primitive["source_request_id"] == "req-1"
    assert primitive["ttl_sec"] == duration == 0.8
    assert "cmd_vel" not in primitive
    assert "waypoint" not in primitive


def test_backtrack_recovery_is_bounded_and_does_not_accept_motor_fields():
    primitive, duration = recovery_to_primitive(
        {"recovery": "backtrack", "request_id": "req-2"},
        {"semantic_recovery": {"backtrack_duration_sec": 99.0, "backtrack_distance_m": 0.2}},
    )

    assert primitive["primitive"] == "back_off"
    assert primitive["distance_m"] == 0.2
    assert primitive["ttl_sec"] == duration == 1.5


@pytest.mark.parametrize("recovery", ["ask", "stop"])
def test_non_motion_recovery_maps_to_stop(recovery):
    primitive, duration = recovery_to_primitive({"recovery": recovery, "request_id": "req"})
    assert primitive["primitive"] == "stop"
    assert 0.05 <= duration <= 1.5


def test_unknown_recovery_is_rejected():
    with pytest.raises(SchemaError):
        recovery_to_primitive({"recovery": "move_forward"})
