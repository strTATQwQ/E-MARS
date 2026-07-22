from omninav_step_scheduler.schemas import parse_safety_status
from omninav_step_scheduler.step_supervisor_node import (
    forbidden_for_step,
    image_only_role_context,
    instruction_context,
    role_requires_multimodal,
    should_trigger_step,
    trigger_matches_active_episode,
)


def test_mission_start_is_hard_step_trigger():
    triggered = should_trigger_step({"type": "mission_start"}, "IDLE", parse_safety_status({}))

    assert triggered


def test_target_visible_is_hard_step_trigger_when_safe():
    triggered = should_trigger_step(
        {"type": "target_visible", "target_visible": True, "distance_to_target_m": 2.3},
        "RUN_FAST",
        parse_safety_status({"local_costmap_clear": True}),
    )

    assert triggered


def test_single_low_confidence_is_not_hard_trigger():
    triggered = should_trigger_step({"type": "low_confidence_once"}, "RUN_FAST", parse_safety_status({}))

    assert not triggered


def test_repeated_low_confidence_triggers_step():
    triggered = should_trigger_step(
        {"type": "omninav_low_confidence_twice"},
        "RUN_FAST",
        parse_safety_status({}),
        {"low_confidence_streak": 2},
    )

    assert triggered


def test_dynamic_obstacle_forbids_step_call():
    forbidden = forbidden_for_step(parse_safety_status({"dynamic_obstacle": True}), {"last_omninav_age": 0.1})

    assert forbidden


def test_route_choice_allowed_at_clear_intersection():
    triggered = should_trigger_step(
        {"type": "route_choice_upcoming"},
        "RUN_FAST",
        parse_safety_status({"near_intersection": True, "local_costmap_clear": True}),
        {"soft_step_allowed": True, "last_omninav_age": 0.1},
    )

    assert triggered


def test_route_choice_still_blocked_when_local_costmap_blocked():
    triggered = should_trigger_step(
        {"type": "route_choice_upcoming"},
        "RUN_FAST",
        parse_safety_status({"near_intersection": True, "local_costmap_clear": False}),
        {"soft_step_allowed": True, "last_omninav_age": 0.1},
    )

    assert not triggered


def test_scoped_trigger_waits_for_matching_benchmark_mode():
    event = {"type": "mission_start", "mission_id": "episode_1"}

    assert trigger_matches_active_episode(event, "") is False
    assert trigger_matches_active_episode(event, "episode_2") is False
    assert trigger_matches_active_episode(event, "episode_1") is True
    assert trigger_matches_active_episode({"type": "route_choice_upcoming"}, "episode_1") is True


def test_current_event_instruction_wins_over_previous_episode_cache():
    mission, subgoal = instruction_context(
        {"instruction": "turn right", "active_subgoal": "blue box"},
        "turn left",
        "red cone",
    )

    assert mission == "turn right"
    assert subgoal == "blue box"


def test_visual_roles_can_be_forced_multimodal_without_affecting_generic_step():
    config = {"step": {"multimodal_required_roles": ["route_choice", "semantic_stop"]}}

    assert role_requires_multimodal("route_choice", config)
    assert role_requires_multimodal("semantic_stop", config)
    assert not role_requires_multimodal("disabled", config)


def test_image_only_context_removes_oracle_visibility_route_and_distance():
    event, summary = image_only_role_context(
        {
            "type": "target_visible",
            "target": "fire extinguisher",
            "target_visible": True,
            "distance_to_target_m": 1.8,
            "correct_branch": "left",
        },
        {"target_visible": True, "distance_to_target_m": 1.8, "objects": ["red object"]},
    )

    assert event == {
        "type": "visual_decision_due",
        "target": "fire extinguisher",
        "instruction": None,
        "active_subgoal": None,
    }
    assert summary == {"objects": ["red object"]}


def test_image_only_context_keeps_only_confirmed_actual_sensor_track():
    event, _summary = image_only_role_context(
        {
            "type": "candidate_goal_reached",
            "target": "red fire extinguisher",
            "distance_basis": "actual_mask_depth",
            "track": {
                "episode_id": "episode_a",
                "target_id": "fire extinguisher",
                "confirmed": True,
                "fresh": True,
                "visible": True,
                "distance_m": 1.8,
                "bearing_rad": 0.1,
                "frame_seq": 9,
                "hits": 8,
                "source": "actual_sensor_spatial_track",
                "target_pose": [4.0, 0.0, 0.0],
            },
        },
        {},
    )
    assert event["distance_to_target_m"] == 1.8
    assert event["allow_distance_context"] is True
    assert event["sensor_track"]["confirmed"] is True
    assert "target_pose" not in event["sensor_track"]
