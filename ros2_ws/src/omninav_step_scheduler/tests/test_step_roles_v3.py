import pytest

from omninav_step_scheduler.schemas import SchemaError
from omninav_step_scheduler.step_roles import (
    SemanticStopGate,
    TargetTrackState,
    VisibleToStopMonitor,
    build_route_choice_prompt,
    build_semantic_stop_prompt,
    parse_route_choice_json,
    parse_semantic_stop_json,
    route_choice_verifier,
    semantic_stop_verifier,
    semantic_target_query,
)
from omninav_step_scheduler.route_stop_primitive_bridge_node import (
    forced_route_choice_json,
    forced_stop_json,
    route_choice_to_primitive,
    stop_decision_to_primitive,
)


def test_route_choice_json_parse_is_strict_and_motion_free():
    parsed = parse_route_choice_json(
        '{"route_choice":"left","confidence":0.9,"evidence":"red cone is left","visible_in_view":"left"}'
    )
    assert parsed["route_choice"] == "left"
    assert parsed["visible_in_view"] == "left"
    with pytest.raises(SchemaError):
        parse_route_choice_json(
            {
                "route_choice": "left",
                "confidence": 0.9,
                "evidence": "bad motion field",
                "visible_in_view": "left",
                "cmd_vel": {"x": 0.1},
            }
        )


def test_semantic_stop_json_parse_is_strict_and_motion_free():
    parsed = parse_semantic_stop_json(
        {"stop": True, "target_visible": True, "estimated_distance_ok": True, "confidence": 0.8, "reason": "close"}
    )
    assert parsed["stop"] is True
    with pytest.raises(SchemaError):
        parse_semantic_stop_json(
            {
                "stop": True,
                "target_visible": True,
                "estimated_distance_ok": True,
                "confidence": 0.8,
                "reason": "bad motion field",
                "primitive": "turn_left",
            }
        )


def test_route_choice_verifier_uses_instruction_direction():
    result = route_choice_verifier(
        instruction="At the intersection, turn right toward the blue box.",
        semantic_summary={"objects": [{"name": "blue_box", "view": "right"}]},
    )
    assert result == {
        "route_choice": "right",
        "confidence": 0.92,
        "evidence": "instruction says right",
        "visible_in_view": "right",
    }


def test_semantic_stop_verifier_stops_only_when_visible_and_close():
    result = semantic_stop_verifier(target="fire_extinguisher", target_visible=True, distance_m=2.1)
    assert result["stop"] is True
    assert result["estimated_distance_ok"] is True
    far = semantic_stop_verifier(target="fire_extinguisher", target_visible=True, distance_m=3.2)
    assert far["stop"] is False


def test_route_choice_prompt_is_compact_and_motion_free():
    prompt = build_route_choice_prompt(
        instruction="At the intersection turn left toward the red cone.",
        active_subgoal="red cone",
        robot_state={"pose": [1.0, 2.0, 0.0], "large": "x" * 1000},
        semantic_summary={"objects": [{"name": "red_cone", "view": "left"}]},
        safety_status={"safe_mux": "ready", "large": "x" * 1000},
        event={"type": "route_choice_upcoming", "visible_in_view": "left"},
    )
    user = prompt["messages"][1]["content"]
    assert "robot_state" not in user
    assert "safety_status" not in user
    assert "route_choice_upcoming" in user
    assert '"instruction_route_hint": "left"' in user
    assert '"target_visual_attributes": {"category": "cone", "color": "red"}' in user
    assert "Never replace an explicit instruction direction" in prompt["messages"][0]["content"]
    assert len(user) < 700


def test_semantic_stop_prompt_hides_oracle_visibility_and_keeps_geometric_distance():
    prompt = build_semantic_stop_prompt(
        instruction="Stop near the fire extinguisher.",
        active_subgoal="fire extinguisher",
        robot_state={"pose": [0.0, 0.0, 0.0], "large": "x" * 1000},
        semantic_summary={"target_visible": True, "distance_to_target_m": 3.2},
        safety_status={"safe_mux": "ready", "large": "x" * 1000},
        event={
            "type": "target_visible",
            "target": "fire_extinguisher",
            "target_visible": True,
            "distance_to_target_m": 3.2,
        },
    )
    system = prompt["messages"][0]["content"]
    user = prompt["messages"][1]["content"]
    assert "attached front camera image" in system
    assert "image is authoritative" in system
    assert "not evidence that it is present" in system
    assert '"target_visible": true' not in user
    assert '"distance_to_target_m": 3.2' in user
    assert '"within_stop_distance": false' in user
    assert '"required_visual_attributes": {"category": "fire extinguisher"}' in user
    assert "Is an actual fire extinguisher visually present" in user
    assert '"stop": "boolean"' in user
    assert "robot_state" not in user
    assert "safety_status" not in user
    assert len(user) < 800


def test_semantic_stop_prompt_omits_simulator_distance_for_visual_decision():
    prompt = build_semantic_stop_prompt(
        instruction="Find the fire extinguisher.",
        active_subgoal="fire extinguisher",
        robot_state={},
        semantic_summary={"distance_to_target_m": 1.5},
        safety_status={},
        event={"type": "visual_decision_due", "target": "fire extinguisher"},
    )

    user = prompt["messages"][1]["content"]
    assert '"distance_to_target_m"' not in user
    assert '"within_stop_distance"' not in user
    assert '"distance_basis": "image_estimate"' in user


def test_semantic_target_query_removes_action_words_but_keeps_visual_color():
    assert semantic_target_query("Approach the red fire extinguisher and stop.") == "red fire extinguisher"
    prompt = build_semantic_stop_prompt(
        instruction="Approach the red fire extinguisher and stop.",
        active_subgoal="Approach the red fire extinguisher and stop.",
        robot_state={},
        semantic_summary={},
        safety_status={},
        event={"type": "visual_decision_due"},
    )
    user = prompt["messages"][1]["content"]
    assert '"target_query": "red fire extinguisher"' in user
    assert '"required_visual_attributes": {"category": "fire extinguisher", "color": "red"}' in user


def test_semantic_stop_prompt_precomputes_near_distance_boolean():
    prompt = build_semantic_stop_prompt(
        instruction="Stop near the target.",
        active_subgoal="target",
        robot_state={},
        semantic_summary={},
        safety_status={},
        event={"target_visible": True, "distance_to_target_m": 2.352},
    )

    assert '"within_stop_distance": true' in prompt["messages"][1]["content"]


def test_visible_to_stop_latency_monitor():
    monitor = VisibleToStopMonitor()
    monitor.update(timestamp=10.0, target_visible=True, distance_m=3.0)
    summary = monitor.update(timestamp=11.4, target_visible=True, distance_m=2.0, stop_command=True)
    assert summary["visible_to_stop_latency_sec"] == pytest.approx(1.4)
    assert summary["stopped_too_late"] is False


def test_forced_semantic_stop_gate():
    gate = SemanticStopGate(force_stop_if_distance_less_than_m=2.0, force_stop_after_visible_sec=1.5)
    assert gate.update(timestamp=0.0, target_visible=True, distance_m=3.0)["force_stop"] is False
    decision = gate.update(timestamp=0.5, target_visible=True, distance_m=1.9)
    assert decision["force_stop"] is True
    assert decision["actions_through_safe_mux"] is True


def test_forced_route_oracle_maps_to_follow_waypoint_primitive():
    decision = forced_route_choice_json({"route_choice": "right", "confidence": 1.0, "source": "forced_oracle"})
    primitive = route_choice_to_primitive(decision, {"route_choice_bridge": {"turn_yaw_deg": 32.0}})
    assert primitive["primitive"] == "follow_waypoint"
    assert primitive["yaw_deg"] == -32.0
    assert primitive["source"] == "forced_oracle"


def test_forced_stop_oracle_maps_to_stop_primitive():
    decision = forced_stop_json({"stop": True, "source": "forced_oracle", "confidence": 1.0})
    primitive = stop_decision_to_primitive(decision)
    assert primitive["primitive"] == "stop"
    assert primitive["source"] == "forced_oracle"


def test_target_track_is_episode_scoped_and_requires_confidence_or_hits():
    track = TargetTrackState(required_hits=2, high_confidence_single_hit=0.85, min_confidence=0.60)
    first = track.update(
        timestamp=10.0,
        episode_id="ep-1",
        target="fire extinguisher",
        visible=True,
        confidence=0.70,
        visible_in_view="front",
        frame_seq=11,
    )
    second = track.update(
        timestamp=10.4,
        episode_id="ep-1",
        target="fire extinguisher",
        visible=True,
        confidence=0.72,
        visible_in_view="front",
        frame_seq=12,
    )
    reset = track.update(
        timestamp=11.0,
        episode_id="ep-2",
        target="fire extinguisher",
        visible=False,
        confidence=0.9,
        frame_seq=13,
    )

    assert first["confirmed"] is False
    assert second["confirmed"] is True
    assert reset["confirmed"] is False
    assert reset["hits"] == 0


def test_target_track_does_not_count_duplicate_frame_twice():
    track = TargetTrackState(required_hits=2, high_confidence_single_hit=1.01, min_confidence=0.60)
    first = track.update(
        timestamp=1.0,
        episode_id="ep",
        target="fire extinguisher",
        visible=True,
        confidence=0.9,
        frame_seq=7,
    )
    duplicate = track.update(
        timestamp=1.1,
        episode_id="ep",
        target="fire extinguisher",
        visible=True,
        confidence=0.9,
        frame_seq=7,
    )
    second_frame = track.update(
        timestamp=1.2,
        episode_id="ep",
        target="fire extinguisher",
        visible=True,
        confidence=0.9,
        frame_seq=8,
    )

    assert first["hits"] == 1
    assert duplicate["hits"] == 1
    assert duplicate["confirmed"] is False
    assert second_frame["hits"] == 2
    assert second_frame["confirmed"] is True


def test_target_track_requires_consecutive_valid_frames_after_a_miss():
    track = TargetTrackState(required_hits=2, high_confidence_single_hit=1.01, min_confidence=0.60)
    track.update(timestamp=1.0, episode_id="ep", target="extinguisher", visible=True, confidence=0.9, frame_seq=1)
    missed = track.update(
        timestamp=1.1,
        episode_id="ep",
        target="extinguisher",
        visible=False,
        confidence=0.0,
        frame_seq=2,
    )
    next_hit = track.update(
        timestamp=1.2,
        episode_id="ep",
        target="extinguisher",
        visible=True,
        confidence=0.9,
        frame_seq=3,
    )

    assert missed["hits"] == 0
    assert next_hit["hits"] == 1
    assert next_hit["confirmed"] is False


def test_step_stop_requires_confirmed_visible_track_when_configured():
    decision = {
        "stop": True,
        "source": "step_semantic_stop",
        "confidence": 0.9,
        "track": {"confirmed": False, "visible": True},
    }
    config = {"semantic_stop_bridge": {"require_confirmed_track": True}}

    assert stop_decision_to_primitive(decision, config) is None
    decision["track"]["confirmed"] = True
    assert stop_decision_to_primitive(decision, config)["primitive"] == "stop"


def test_route_primitive_can_retain_episode_scope_from_decision():
    decision = {
        "route_choice": "left",
        "source": "step_route_choice",
        "episode_id": "ep-7",
        "request_id": "req-7",
    }
    primitive = route_choice_to_primitive(decision)
    from omninav_step_scheduler.route_stop_primitive_bridge_node import _copy_metadata

    _copy_metadata(decision, primitive)

    assert primitive["episode_id"] == "ep-7"
    assert primitive["request_id"] == "req-7"
