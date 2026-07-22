import json
from types import SimpleNamespace

import pytest

from omninav_step_scheduler.schemas import SchemaError
from omninav_step_scheduler.semantic_executive import (
    SemanticExecutiveCore,
    build_semantic_executive_prompt,
    oracle_leakage_findings,
    parse_semantic_subgoal_json,
    sanitize_observable_context,
    validate_semantic_goal_payload,
)
from omninav_step_scheduler.step_roles import step_role_for_event
from omninav_step_scheduler.omninav_model_client_node import _instruction_from_request
from omninav_step_scheduler.semantic_executive_node import SemanticExecutiveNode


def valid_subgoal(**overrides):
    payload = {
        "subgoal_type": "find",
        "target": "red fire extinguisher",
        "relation": "beside the second doorway",
        "constraints": ["avoid the kitchen", "use the second doorway"],
        "completion_evidence": "confirmed target track in two fresh frames",
        "recovery": "scan",
        "confidence": 0.88,
    }
    payload.update(overrides)
    return payload


def test_semantic_subgoal_is_strict_and_motion_free():
    parsed = parse_semantic_subgoal_json(valid_subgoal())
    assert parsed.subgoal_type == "find"
    assert parsed.target == "red fire extinguisher"

    for key, value in {
        "cmd_vel": {"x": 0.2},
        "waypoint": [1.0, 2.0],
        "target_pose": [2.0, 3.0, 0.0],
        "oracle_plan": ["left"],
    }.items():
        with pytest.raises(SchemaError):
            parse_semantic_subgoal_json(valid_subgoal(**{key: value}))


def test_semantic_subgoal_accepts_expanded_v5_timebase_metadata():
    parsed = parse_semantic_subgoal_json(
        valid_subgoal(
            episode_id="ep-1",
            request_id="req-1",
            clock_domain="ros_system",
            ros_now_sec=10.0,
            wall_now_sec=20.0,
            clock_msg_sec=10.0,
            header_stamp_sec=10.0,
            source_stamp_sec=10.0,
            created_ros_time_sec=10.0,
            created_wall_time_sec=20.0,
            timestamp_response=10.0,
            timebase={"source_stamp_sec": 10.0},
        )
    )
    assert parsed.metadata["episode_id"] == "ep-1"
    assert parsed.metadata["source_stamp_sec"] == 10.0


def test_semantic_constraint_rejects_geometric_control():
    with pytest.raises(SchemaError, match="semantic"):
        parse_semantic_subgoal_json(valid_subgoal(constraints=["drive at 0.2 m/s"]))


def test_prompt_removes_oracle_and_control_context():
    prompt = build_semantic_executive_prompt(
        instruction="Pass the lobby and enter the second door beside the elevator.",
        active_subgoal={"target": "lobby"},
        observation_history=[
            {
                "sensor_track": {"target_visible": True, "distance_to_target_m": 2.3, "source": "rgbd"},
                "target_pose": [4.0, 1.0, 0.0],
                "correct_branch": "left",
                "primitive": "turn_left",
            }
        ],
        event={"type": "semantic_plan_requested", "oracle_plan": ["left"]},
    )
    user = json.loads(prompt["messages"][1]["content"])
    serialized = json.dumps(user)
    assert "target_pose" not in serialized
    assert "correct_branch" not in serialized
    assert "oracle_plan" not in serialized
    assert "primitive" not in serialized
    assert user["observation_history"][0]["sensor_track"]["target_visible"] is True
    assert "Never output velocity" in prompt["messages"][0]["content"]
    assert "terminal verify subgoal must use recovery stop" in prompt["messages"][0]["content"]
    assert "target must repeat the concrete semantic target" in prompt["messages"][0]["content"]


def test_prompt_strips_transport_timebase_without_losing_sequence_state():
    prompt = build_semantic_executive_prompt(
        instruction="Pass red marker, then enter blue marker.",
        active_subgoal=None,
        observation_history=[],
        event={
            "type": "semantic_subgoal_completed",
            "episode_id": "ep-1",
            "next_subgoal_index": 1,
            "timebase": {"source_stamp_sec": 12.0},
            "source_stamp_sec": 12.0,
        },
    )
    event = json.loads(prompt["messages"][1]["content"])["event"]
    assert event["episode_id"] == "ep-1"
    assert event["next_subgoal_index"] == 1
    assert "timebase" not in event
    assert "source_stamp_sec" not in event


def test_oracle_leakage_audit_requires_sensor_provenance_path():
    unsafe = {
        "target_visible": True,
        "ground_truth": {"correct_branch": "left"},
        "sensor_track": {"target_visible": True, "distance_to_target_m": 1.9, "source": "rgbd"},
    }
    findings = oracle_leakage_findings(unsafe)
    assert "target_visible" in findings
    assert "ground_truth" in findings
    assert "ground_truth.correct_branch" in findings
    assert "sensor_track.target_visible" not in findings
    assert "sensor_track.distance_to_target_m" not in findings
    sanitized = sanitize_observable_context(unsafe)
    assert "target_visible" not in sanitized
    assert "ground_truth" not in sanitized


def test_core_emits_semantic_goal_and_never_motion():
    core = SemanticExecutiveCore(min_confidence=0.55)
    core.reset("ep-1")
    subgoal = parse_semantic_subgoal_json(valid_subgoal(episode_id="ep-1"))
    accepted = core.accept(subgoal)
    assert accepted["status"] == "RUNNING"
    assert accepted["semantic_goal"]["target"] == "red fire extinguisher"
    assert accepted["publishes_motion"] is False
    assert not any(key in accepted["semantic_goal"] for key in ("waypoint", "cmd_vel", "primitive"))

    completed = core.handle_event({"type": "target_track_confirmed", "episode_id": "ep-1"})
    assert completed["status"] == "NEEDS_PLAN"
    assert completed["reason"] == "subgoal_completed"


def test_verify_completion_is_terminal_and_does_not_request_another_plan():
    core = SemanticExecutiveCore(min_confidence=0.55)
    core.reset("ep-verify")
    core.accept(
        parse_semantic_subgoal_json(
            valid_subgoal(
                episode_id="ep-verify",
                subgoal_type="verify",
                recovery="stop",
            )
        )
    )
    completed = core.handle_event({"type": "completion_verified", "episode_id": "ep-verify"})
    assert completed["status"] == "COMPLETE"
    assert completed["reason"] == "mission_verified"


def test_core_recovery_does_not_blindly_advance():
    core = SemanticExecutiveCore()
    core.reset("ep")
    core.accept(parse_semantic_subgoal_json(valid_subgoal(episode_id="ep", recovery="backtrack")))
    result = core.handle_event({"type": "no_progress_timeout"})
    assert result["status"] == "RECOVERY"
    assert result["recovery"] == "backtrack"
    assert result["publishes_motion"] is False


def test_core_recovery_completion_resumes_the_same_subgoal():
    core = SemanticExecutiveCore()
    core.reset("ep")
    core.accept(parse_semantic_subgoal_json(valid_subgoal(episode_id="ep", recovery="scan")))
    assert core.handle_event({"type": "target_track_lost"})["status"] == "RECOVERY"

    completed = core.handle_event({"type": "semantic_recovery_completed", "recovery": "scan"})

    assert completed["status"] == "RUNNING"
    assert completed["reason"] == "semantic_recovery_completed"
    assert completed["recovery"] == "scan"
    assert completed["resumed_subgoal"]["target"] == "red fire extinguisher"
    assert core.active is not None


def test_semantic_executive_events_have_their_own_step_role():
    assert step_role_for_event({"type": "semantic_plan_requested"}) == "semantic_executive"
    assert step_role_for_event({"type": "semantic_subgoal_completed"}) == "semantic_executive"


def test_omninav_receives_semantic_language_not_geometry():
    instruction = _instruction_from_request(
        {
            "subgoal": {
                "semantic_goal": {
                    "subgoal_type": "enter",
                    "target": "second doorway",
                    "relation": "after the reception desk",
                    "constraints": ["avoid the kitchen"],
                    "completion_evidence": "door threshold is crossed",
                }
            }
        }
    )
    assert instruction.startswith("Go through second doorway")
    assert "after the reception desk" in instruction
    assert "avoid the kitchen" in instruction
    assert not any(term in instruction for term in ("cmd_vel", "waypoint", "target_pose"))


def test_omninav_semantic_goal_gate_rejects_motion_fields():
    goal = {
        "subgoal_type": "find",
        "target": "printer",
        "relation": "beyond the blue sign",
        "constraints": ["avoid the kitchen"],
        "completion_evidence": "printer track is confirmed",
    }
    assert validate_semantic_goal_payload(goal)["target"] == "printer"
    with pytest.raises(SchemaError, match="forbidden"):
        validate_semantic_goal_payload(goal | {"waypoint": [1.0, 2.0]})


def test_episode_reset_event_does_not_request_plan_without_instruction():
    node = object.__new__(SemanticExecutiveNode)
    node.enabled = True
    node.active_episode_id = "ep1"
    node.core = SemanticExecutiveCore()
    node.core.reset("ep1")
    states = []
    metrics = []
    triggers = []
    node.publish_state = states.append
    node.publish_metric = lambda event, **fields: metrics.append((event, fields))
    node.publish_trigger = triggers.append

    node.on_event(SimpleNamespace(data=json.dumps({"type": "episode_reset", "episode_id": "ep1"})))

    assert states[-1]["status"] == "NEEDS_PLAN"
    assert metrics[-1][1]["result"] == "observed_without_trigger"
    assert metrics[-1][1]["observed_event_type"] == "episode_reset"
    assert triggers == []


def test_low_confidence_recovery_retries_same_subgoal_after_recovery_completes():
    node = object.__new__(SemanticExecutiveNode)
    node.enabled = True
    node.active_episode_id = "ep1"
    node.core = SemanticExecutiveCore(min_confidence=0.55)
    node.core.reset("ep1")
    node.next_subgoal_index = 2
    rejected = node.core.accept(
        parse_semantic_subgoal_json(valid_subgoal(episode_id="ep1", confidence=0.0))
    )
    assert rejected["status"] == "RECOVERY"
    assert node.core.active is None

    states = []
    metrics = []
    triggers = []
    node.publish_state = states.append
    node.publish_metric = lambda event, **fields: metrics.append((event, fields))
    node.publish_trigger = triggers.append

    node.on_event(
        SimpleNamespace(
            data=json.dumps(
                {
                    "type": "semantic_recovery_completed",
                    "episode_id": "ep1",
                    "recovery": "stop",
                }
            )
        )
    )

    assert states[-1]["status"] == "NEEDS_PLAN"
    assert states[-1]["reason"] == "recovery_retry_required"
    assert metrics[-1][1]["result"] == "retry_triggered"
    assert len(triggers) == 1
    assert triggers[0]["type"] == "semantic_plan_requested"
    assert triggers[0]["next_subgoal_index"] == 2
    assert triggers[0]["reason"] == "recovery_retry_required"
    assert node.next_subgoal_index == 2


def test_completed_semantic_subgoal_requests_exactly_one_next_plan():
    node = object.__new__(SemanticExecutiveNode)
    node.enabled = True
    node.active_episode_id = "ep1"
    node.core = SemanticExecutiveCore()
    node.core.reset("ep1")
    node.next_subgoal_index = 0
    node.core.accept(parse_semantic_subgoal_json(valid_subgoal(episode_id="ep1")))
    states = []
    triggers = []
    clears = []
    holds = []
    node.publish_state = states.append
    node.publish_trigger = triggers.append
    node.clear_omninav_goal = clears.append
    node.hold_omninav_goal = holds.append
    node.publish_recovery = lambda *args: None
    node.publish_metric = lambda *args, **kwargs: None

    node.on_event(SimpleNamespace(data=json.dumps({"type": "target_track_confirmed", "episode_id": "ep1"})))

    assert states[-1]["status"] == "NEEDS_PLAN"
    assert clears == []
    assert holds == ["subgoal_completed"]
    assert len(triggers) == 1
    assert triggers[0]["type"] == "semantic_subgoal_completed"
    assert triggers[0]["completed_subgoal_index"] == 0
    assert triggers[0]["next_subgoal_index"] == 1
    assert node.next_subgoal_index == 1
