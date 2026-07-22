import pytest

from omninav_step_scheduler.schemas import (
    SchemaError,
    parse_omninav_action,
    parse_step_plan,
)


def test_invalid_step_json_raises_schema_error():
    with pytest.raises(SchemaError):
        parse_step_plan("{not-json")


def test_omninav_action_requires_confidence():
    with pytest.raises(SchemaError, match="confidence"):
        parse_omninav_action(
            {
                "request_id": "a1",
                "timestamp_request": 99.9,
                "timestamp_response": 100.0,
                "primitive": "move_forward",
                "pose_at_snapshot": [0.0, 0.0, 0.0],
            }
        )


def test_step_plan_parses_minimal_valid_payload():
    plan = parse_step_plan(
        {
            "request_id": "p1",
            "timestamp_request": 99.8,
            "timestamp_response": 100.0,
            "multimodal": False,
            "pose_at_request": [0.0, 0.0, 0.0],
            "navila_or_omninav_instruction": "approach the red door",
            "subgoal": "red door",
            "success_condition": "arrived at red door",
            "constraints": {"max_speed_mps": 0.4},
            "recommended_pending_mode": "stop",
            "confidence": 0.8,
        }
    )

    assert plan.constraints.max_speed_mps == 0.4
    assert plan.subgoal == "red door"
