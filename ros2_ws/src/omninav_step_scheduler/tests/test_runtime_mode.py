import json
from types import SimpleNamespace

from omninav_step_scheduler.mission_manager_node import (
    MissionManagerNode,
    near_goal_stop_needed,
    periodic_step_only_allowed,
    semantic_step_task_allowed,
    semantic_stop_actual_distance_threshold,
    semantic_stop_release_distance_threshold,
    confirmed_track_is_fresh,
    semantic_stop_retry_config,
    semantic_stop_requires_step,
    step_state_for_event,
    step_trigger_allowed,
)
from omninav_step_scheduler.runtime_mode import apply_benchmark_mode_config
from omninav_step_scheduler.schemas import SchedulerState


def test_apply_benchmark_mode_disables_step_for_omninav_only():
    base = {
        "step": {"enabled": True, "min_interval_sec": 8.0},
        "omninav": {"enabled": True},
        "pending_policy": {"allow_move_while_step": True},
    }
    cfg = apply_benchmark_mode_config(
        base,
        {"mode": "omninav_only", "mode_config": {"use_step": False, "use_omninav": True, "step_policy": "never"}},
    )
    assert cfg["step"]["enabled"] is False
    assert cfg["omninav"]["enabled"] is True
    assert base["step"]["enabled"] is True


def test_apply_benchmark_mode_disables_omninav_for_step_only():
    base = {
        "step": {"enabled": True, "min_interval_sec": 8.0},
        "omninav": {"enabled": True},
        "pending_policy": {"allow_move_while_step": True},
    }
    cfg = apply_benchmark_mode_config(
        base,
        {
            "mode": "step_only",
            "mode_config": {
                "use_step": True,
                "use_omninav": False,
                "step_policy": "every_decision",
                "pending_policy": "stop",
            },
        },
    )
    assert cfg["step"]["enabled"] is True
    assert cfg["omninav"]["enabled"] is False
    assert cfg["step"]["min_interval_sec"] == 1.0
    assert cfg["pending_policy"]["benchmark_policy"] == "stop"
    assert cfg["pending_policy"]["allow_move_while_step"] is False


def test_apply_benchmark_mode_routes_internnav_modes():
    base = {
        "step": {"enabled": True},
        "omninav": {"enabled": True},
        "internnav": {"enabled": False},
        "pending_policy": {"allow_move_while_step": True},
    }
    cfg = apply_benchmark_mode_config(
        base,
        {
            "mode": "step_internnav_event",
            "mode_config": {
                "use_step": True,
                "use_omninav": False,
                "use_internnav": True,
                "pending_policy": "auto",
            },
        },
    )
    assert cfg["step"]["enabled"] is True
    assert cfg["omninav"]["enabled"] is False
    assert cfg["internnav"]["enabled"] is True


def test_apply_benchmark_mode_sets_step_trigger_filter():
    cfg = apply_benchmark_mode_config(
        {"step": {"enabled": True}},
        {
            "mode": "step_omninav_goal_verify_only",
            "mode_config": {
                "use_step": True,
                "use_omninav": True,
                "step_triggers": ["goal_verify_only"],
            },
        },
    )
    assert cfg["runtime_mode"]["step_triggers"] == ["goal_verify_only"]
    assert step_trigger_allowed("candidate_goal_reached", cfg)
    assert step_trigger_allowed("completion_verification", cfg)
    assert not step_trigger_allowed("mission_start", cfg)
    assert not step_trigger_allowed("route_choice_upcoming", cfg)


def test_apply_benchmark_mode_extends_mission_timeout_from_episode_timeout():
    cfg = apply_benchmark_mode_config(
        {"mission": {"max_duration_sec": 90}},
        {
            "mode": "omninav_only",
            "mode_config": {
                "task_timeout_sec": 135,
                "mission_timeout_margin_sec": 15,
            },
        },
    )
    assert cfg["mission"]["max_duration_sec"] == 150
    assert cfg["runtime_mode"]["mission_max_duration_sec"] == 150


def test_apply_benchmark_mode_does_not_shorten_long_mission_timeout():
    cfg = apply_benchmark_mode_config(
        {"mission": {"max_duration_sec": 180}},
        {"mode": "omninav_only", "mode_config": {"task_timeout_sec": 120}},
    )
    assert cfg["mission"]["max_duration_sec"] == 180


def test_apply_benchmark_mode_can_require_external_step_triggers():
    cfg = apply_benchmark_mode_config(
        {},
        {
            "mode": "step_stop_verify_live",
            "mode_config": {"external_step_triggers_only": True},
        },
    )

    assert cfg["runtime_mode"]["external_step_triggers_only"] is True


def test_near_goal_stop_needed_only_for_unheld_stop():
    assert near_goal_stop_needed({"success": False, "done": False, "reason": "never_stopped"})
    assert not near_goal_stop_needed({"success": False, "done": False, "reason": "distance_pending"})
    assert not near_goal_stop_needed({"success": True, "done": True, "reason": None})


def test_semantic_step_evidence_mode_disables_preemptive_goal_stop():
    assert semantic_stop_requires_step(
        {"runtime_mode": {"mode_config": {"semantic_stop_requires_step": True}}}
    )
    assert not semantic_stop_requires_step({"runtime_mode": {"mode_config": {}}})


def test_semantic_retry_is_opt_in_and_mode_scoped():
    base = {"step": {"semantic_stop_retry": {"enabled": False, "max_attempts": 1, "delay_sec": 0.5}}}
    assert semantic_stop_retry_config(base)["enabled"] is False

    configured = dict(base) | {
        "runtime_mode": {
            "mode_config": {
                "semantic_stop_retry_enabled": True,
                "semantic_stop_max_attempts": 3,
                "semantic_stop_retry_delay_sec": 0.25,
            }
        }
    }
    assert semantic_stop_retry_config(configured) == {
        "enabled": True,
        "max_attempts": 3,
        "delay_sec": 0.25,
    }
    assert semantic_stop_actual_distance_threshold(
        {"runtime_mode": {"mode_config": {"target_stop_threshold_m": 1.95}}}
    ) == 1.95
    assert semantic_stop_release_distance_threshold(
        {"runtime_mode": {"mode_config": {"target_stop_threshold_m": 1.95}}}
    ) == 1.95


def test_confirmed_track_release_requires_current_visible_track():
    config = {"model_clients": {"step_http": {"target_track": {"max_age_sec": 12.0}}}}
    track = {"confirmed": True, "visible": True, "last_seen_time": 100.0}

    assert confirmed_track_is_fresh(track, 111.9, config)
    assert not confirmed_track_is_fresh(track, 112.1, config)
    assert not confirmed_track_is_fresh(dict(track) | {"visible": False}, 101.0, config)


def test_v12_semantic_stop_modes_filter_non_semantic_tasks():
    cfg = {"runtime_mode": {"mode": "omninav_step_route_stop_v12_screen", "mode_config": {}}}
    assert semantic_step_task_allowed(cfg, {"task_id": "semantic_001_seed0", "task_type": "semantic_target"})
    assert not semantic_step_task_allowed(cfg, {"task_id": "simple_001_seed0", "task_type": "simple_navigation"})
    assert not semantic_step_task_allowed(cfg, {"task_id": "turn_001_seed0", "task_type": "turn_choice"})


def test_step_state_for_event_uses_move_for_auto_nonterminal_events():
    cfg = {"pending_policy": {"benchmark_policy": "auto", "allow_move_while_step": True}}
    assert step_state_for_event({"type": "no_progress_timeout"}, cfg) == SchedulerState.STEP_THINK_MOVE


def test_step_state_for_event_keeps_terminal_verification_stopped():
    cfg = {"pending_policy": {"benchmark_policy": "auto", "allow_move_while_step": True}}
    assert step_state_for_event({"type": "candidate_goal_reached"}, cfg) == SchedulerState.STEP_THINK_STOP
    assert step_state_for_event({"type": "candidate_goal_reached", "reason": "isaac_status_near_goal"}, cfg) == SchedulerState.STEP_THINK_STOP
    assert step_state_for_event({"type": "completion_verification"}, cfg) == SchedulerState.STEP_THINK_STOP


def test_step_state_for_event_moves_for_periodic_step_in_auto_policy():
    cfg = {"pending_policy": {"benchmark_policy": "auto", "allow_move_while_step": True}}
    event = {"type": "candidate_goal_reached", "reason": "step_only_next_decision"}
    assert step_state_for_event(event, cfg) == SchedulerState.STEP_THINK_MOVE


def test_step_trigger_allowed_stuck_group_maps_progress_events():
    cfg = {"runtime_mode": {"step_triggers": ["stuck_replan_only"]}}
    assert step_trigger_allowed("no_progress", cfg)
    assert step_trigger_allowed("forward_bias", cfg)
    assert step_trigger_allowed("safety_stop_requires_explanation", cfg)
    assert not step_trigger_allowed("mission_start", cfg)


def test_mission_manager_reset_episode_state_clears_failsafe():
    node = object.__new__(MissionManagerNode)
    published = []
    node.publish_state = lambda: published.append(node.state.value)
    node.state = SchedulerState.FAILSAFE
    node.mission_id = "old_mission"
    node.instruction = "old instruction"
    node.mission_started_at = 123.0
    node.active_plan = object()
    node.last_step_only_trigger = 4.0
    node.goal_stop_until = 5.0
    node.last_goal_stop_publish = 6.0
    node.semantic_goal_triggered = True
    node.route_choice_triggered = True
    node.was_near_intersection = True
    node.was_near_doorway = True

    node.reset_episode_state()

    assert node.state == SchedulerState.IDLE
    assert node.mission_id == ""
    assert node.mission_started_at == 0.0
    assert node.active_plan is None
    assert node.semantic_goal_triggered is False
    assert node.route_choice_triggered is False
    assert published == [SchedulerState.IDLE.value]


def test_route_choice_decision_resumes_run_fast_without_emitting_motion():
    node = object.__new__(MissionManagerNode)
    node.active_episode_id = "ep-1"
    node.pending_step_request_ids = {"req-1"}
    node.route_choice_triggered = False
    node.state = SchedulerState.STEP_THINK_STOP
    states = []
    metrics = []
    node.publish_state = lambda: states.append(node.state.value)
    node.publish_metric = lambda event_type, **kwargs: metrics.append((event_type, kwargs))
    message = SimpleNamespace(data=json.dumps({"episode_id": "ep-1", "request_id": "req-1", "route_choice": "left"}))

    node.on_route_choice_decision(message)

    assert node.state == SchedulerState.RUN_FAST
    assert states == [SchedulerState.RUN_FAST.value]
    assert node.pending_step_request_ids == set()
    assert metrics[-1][0] == "route_choice_scheduler_resume"


def test_accepted_semantic_subgoal_releases_pending_stop_after_stale_gate():
    node = object.__new__(MissionManagerNode)
    node.active_episode_id = "ep-1"
    node.pending_step_request_ids = {"req-semantic"}
    node.state = SchedulerState.STEP_THINK_STOP
    states = []
    metrics = []
    node.publish_state = lambda: states.append(node.state.value)
    node.publish_metric = lambda event_type, **kwargs: metrics.append((event_type, kwargs))

    node.on_semantic_subgoal_accepted(
        SimpleNamespace(
            data=json.dumps(
                {
                    "episode_id": "ep-1",
                    "request_id": "req-semantic",
                    "subgoal_index": 1,
                    "subgoal_type": "approach",
                }
            )
        )
    )

    assert node.state == SchedulerState.RUN_FAST
    assert node.pending_step_request_ids == set()
    assert states == [SchedulerState.RUN_FAST.value]
    assert metrics[-1][0] == "semantic_executive_scheduler_resume"


def test_semantic_subgoal_does_not_release_pending_for_old_episode():
    node = object.__new__(MissionManagerNode)
    node.active_episode_id = "ep-current"
    node.pending_step_request_ids = {"req-old"}
    node.state = SchedulerState.STEP_THINK_STOP
    node.publish_state = lambda: (_ for _ in ()).throw(AssertionError("must not resume"))
    metrics = []
    node.publish_metric = lambda event_type, **kwargs: metrics.append((event_type, kwargs))

    node.on_semantic_subgoal_accepted(
        SimpleNamespace(data=json.dumps({"episode_id": "ep-old", "request_id": "req-old"}))
    )

    assert node.state == SchedulerState.STEP_THINK_STOP
    assert node.pending_step_request_ids == {"req-old"}
    assert metrics[-1][1]["result"] == "episode_mismatch"


def test_multimodal_step_request_enters_pending_stop():
    node = object.__new__(MissionManagerNode)
    node.active_episode_id = "ep-1"
    node.pending_step_request_ids = set()
    node.state = SchedulerState.RUN_FAST
    states = []
    metrics = []
    node.publish_state = lambda: states.append(node.state.value)
    node.publish_metric = lambda event_type, **kwargs: metrics.append((event_type, kwargs))

    node.on_step_request(
        SimpleNamespace(
            data=json.dumps(
                {"episode_id": "ep-1", "request_id": "req-1", "multimodal": True, "pending_mode": "stop"}
            )
        )
    )

    assert node.state == SchedulerState.STEP_THINK_STOP
    assert node.pending_step_request_ids == {"req-1"}
    assert states == [SchedulerState.STEP_THINK_STOP.value]
    assert metrics[-1][0] == "step_request_pending_stop"


def test_unconfirmed_semantic_stop_schedules_fresh_frame_retry():
    node = object.__new__(MissionManagerNode)
    node.config = {"step": {"semantic_stop_retry": {"enabled": True, "max_attempts": 3, "delay_sec": 0.5}}}
    node.active_episode_id = "ep-1"
    node.pending_step_request_ids = {"req-1"}
    node.semantic_stop_attempts = 1
    node.semantic_retry_pending = False
    node.semantic_retry_due = 0.0
    node.state = SchedulerState.STEP_THINK_STOP
    states = []
    metrics = []
    node.publish_state = lambda: states.append(node.state.value)
    node.publish_metric = lambda event_type, **kwargs: metrics.append((event_type, kwargs))

    node.on_semantic_stop_decision(
        SimpleNamespace(
            data=json.dumps(
                {
                    "episode_id": "ep-1",
                    "request_id": "req-1",
                    "stop": True,
                    "target_visible": True,
                    "track": {"confirmed": False, "visible": True, "frame_seq": 7},
                }
            )
        )
    )

    assert node.semantic_retry_pending is True
    assert node.pending_step_request_ids == set()
    assert metrics[-1][1]["result"] == "scheduled_fresh_frame_retry"


def test_confirmed_semantic_stop_cancels_retry():
    node = object.__new__(MissionManagerNode)
    node.config = {"step": {"semantic_stop_retry": {"enabled": True, "max_attempts": 3, "delay_sec": 0.5}}}
    node.active_episode_id = "ep-1"
    node.pending_step_request_ids = {"req-2"}
    node.semantic_stop_attempts = 2
    node.semantic_retry_pending = True
    node.semantic_retry_wait_for_distance = False
    node.semantic_last_status = {"distance_to_target": 1.9}
    node.state = SchedulerState.STEP_THINK_STOP
    metrics = []
    node.publish_state = lambda: None
    node.publish_metric = lambda event_type, **kwargs: metrics.append((event_type, kwargs))

    node.on_semantic_stop_decision(
        SimpleNamespace(
            data=json.dumps(
                {
                    "episode_id": "ep-1",
                    "request_id": "req-2",
                    "stop": True,
                    "target_visible": True,
                    "track": {"confirmed": True, "visible": True, "frame_seq": 8},
                }
            )
        )
    )

    assert node.semantic_retry_pending is False
    assert metrics[-1][1]["result"] == "confirmed_stop"


def test_confirmed_visual_stop_waits_until_actual_distance_gate():
    node = object.__new__(MissionManagerNode)
    node.config = {
        "step": {"semantic_stop_retry": {"enabled": True, "max_attempts": 5, "delay_sec": 0.5}},
        "runtime_mode": {"mode_config": {"target_stop_threshold_m": 1.95}},
    }
    node.active_episode_id = "ep-1"
    node.pending_step_request_ids = {"req-3"}
    node.semantic_stop_attempts = 2
    node.semantic_retry_pending = False
    node.semantic_retry_wait_for_distance = False
    node.semantic_last_status = {"distance_to_target": 2.3}
    node.state = SchedulerState.STEP_THINK_STOP
    metrics = []
    node.publish_state = lambda: None
    node.publish_metric = lambda event_type, **kwargs: metrics.append((event_type, kwargs))

    node.on_semantic_stop_decision(
        SimpleNamespace(
            data=json.dumps(
                {
                    "episode_id": "ep-1",
                    "request_id": "req-3",
                    "stop": True,
                    "target_visible": True,
                    "track": {"confirmed": True, "visible": True, "frame_seq": 9},
                }
            )
        )
    )

    assert node.semantic_retry_wait_for_distance is True
    assert node.state == SchedulerState.RUN_FAST
    assert metrics[-1][1]["result"] == "waiting_for_actual_stop_distance"


def test_role_only_instruction_starts_fast_without_generic_step_trigger():
    node = object.__new__(MissionManagerNode)
    node.config = {"step": {"enabled": True, "roles_only": True}}
    node.active_episode_id = "ep-1"
    node.state = SchedulerState.IDLE
    starts = []
    node.start_fast_without_step = lambda reason, step_result: starts.append((reason, step_result))

    node.on_user_instruction(SimpleNamespace(data=json.dumps({"instruction": "find cone", "mission_id": "m-1"})))

    assert starts == [("mission_start", "role_only")]
    assert node.mission_id == "m-1"


def test_role_only_mode_disables_generic_periodic_step_calls():
    assert not periodic_step_only_allowed({"step": {"roles_only": True}})
    assert periodic_step_only_allowed({"step": {"roles_only": False}})


def test_route_choice_decision_rejects_episode_mismatch():
    node = object.__new__(MissionManagerNode)
    node.active_episode_id = "ep-current"
    node.pending_step_request_ids = set()
    node.route_choice_triggered = False
    node.state = SchedulerState.STEP_THINK_STOP
    node.publish_state = lambda: (_ for _ in ()).throw(AssertionError("must not resume"))
    metrics = []
    node.publish_metric = lambda event_type, **kwargs: metrics.append((event_type, kwargs))

    node.on_route_choice_decision(SimpleNamespace(data=json.dumps({"episode_id": "ep-old", "route_choice": "right"})))

    assert node.state == SchedulerState.STEP_THINK_STOP
    assert metrics[-1][1]["result"] == "episode_mismatch"
