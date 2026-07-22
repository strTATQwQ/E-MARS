from __future__ import annotations

import json
import time

from .schemas import (
    SchedulerState,
    attach_timebase,
    clock_domain_from_node,
    deep_get,
    load_yaml_file,
    make_event,
    make_metric,
    new_id,
    node_ros_now_sec,
    now,
    parse_safety_status,
    parse_step_plan,
    parse_user_instruction,
)
from .runtime_mode import apply_benchmark_mode_config, mode_payload_from_json
from .stale_gate import evaluate_step_plan

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover
    rclpy = None
    Node = object
    String = None


class MissionManagerNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run MissionManagerNode")
        super().__init__("mission_manager")
        self.declare_parameter("config_file", "")
        self.base_config = load_yaml_file(self.get_parameter("config_file").value)
        self.config = dict(self.base_config)
        self.state = SchedulerState.IDLE
        self.mission_id = ""
        self.instruction = ""
        self.current_pose = [0.0, 0.0, 0.0]
        self.mission_started_at = 0.0
        self.active_plan = None
        self.last_step_only_trigger = 0.0
        self.goal_stop_until = 0.0
        self.last_goal_stop_publish = 0.0
        self.semantic_goal_triggered = False
        self.semantic_stop_attempts = 0
        self.semantic_retry_pending = False
        self.semantic_retry_due = 0.0
        self.semantic_retry_wait_for_distance = False
        self.semantic_confirmed_decision: dict | None = None
        self.semantic_last_status: dict = {}
        self.route_choice_triggered = False
        self.safety = parse_safety_status({})
        self.was_near_intersection = False
        self.was_near_doorway = False
        self.active_episode_id = ""
        self.pending_step_request_ids: set[str] = set()

        self.state_pub = self.create_publisher(String, "/scheduler/state", 10)
        self.trigger_pub = self.create_publisher(String, "/scheduler/step_trigger_json", 10)
        self.constraint_pub = self.create_publisher(String, "/scheduler/active_constraints_json", 10)
        self.subgoal_pub = self.create_publisher(String, "/scheduler/active_subgoal_json", 10)
        self.omni_req_pub = self.create_publisher(String, "/omninav/request_json", 10)
        self.primitive_pub = self.create_publisher(String, "/primitive/command_json", 10)
        self.event_pub = self.create_publisher(String, "/mission/event_json", 10)
        self.semantic_stop_release_pub = self.create_publisher(String, "/scheduler/semantic_stop_release_json", 10)
        self.reset_pub = self.create_publisher(String, "/episode/reset_lifecycle_json", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)

        self.create_subscription(String, "/benchmark/mode_json", self.on_benchmark_mode, 10)
        self.create_subscription(String, "/user_instruction", self.on_user_instruction, 10)
        self.create_subscription(String, "/step/request_json", self.on_step_request, 10)
        self.create_subscription(String, "/step/response_json", self.on_step_response, 10)
        self.create_subscription(String, "/step/route_choice_json", self.on_route_choice_decision, 10)
        self.create_subscription(String, "/step/semantic_stop_json", self.on_semantic_stop_decision, 10)
        self.create_subscription(
            String,
            "/semantic_executive/accepted_subgoal_json",
            self.on_semantic_subgoal_accepted,
            10,
        )
        self.create_subscription(String, "/robot_state_json", self.on_robot_state, 10)
        self.create_subscription(String, "/safety/local_status_json", self.on_safety, 10)
        self.create_subscription(String, "/mission/event_json", self.on_event, 10)
        self.create_subscription(String, "/isaac/episode_status", self.on_isaac_status, 10)
        self.create_timer(0.5, self.tick)
        self.publish_state()

    def on_benchmark_mode(self, msg):
        payload = mode_payload_from_json(msg.data)
        episode_id = str(payload.get("episode_id", ""))
        if episode_id and episode_id != self.active_episode_id:
            self.active_episode_id = episode_id
            self.reset_episode_state()
        self.config = apply_benchmark_mode_config(self.base_config, payload)
        self.publish_metric(
            "benchmark_mode_applied",
            mode=deep_get(self.config, "runtime_mode.mode", ""),
            step_enabled=self.step_enabled(),
            omninav_enabled=self.omninav_enabled(),
        )

    def reset_episode_state(self) -> None:
        self.state = SchedulerState.IDLE
        self.mission_id = ""
        self.instruction = ""
        self.mission_started_at = 0.0
        self.active_plan = None
        self.last_step_only_trigger = 0.0
        self.goal_stop_until = 0.0
        self.last_goal_stop_publish = 0.0
        self.semantic_goal_triggered = False
        self.semantic_stop_attempts = 0
        self.semantic_retry_pending = False
        self.semantic_retry_due = 0.0
        self.semantic_retry_wait_for_distance = False
        self.semantic_confirmed_decision = None
        self.semantic_last_status = {}
        self.route_choice_triggered = False
        self.was_near_intersection = False
        self.was_near_doorway = False
        if not hasattr(self, "pending_step_request_ids"):
            self.pending_step_request_ids = set()
        self.pending_step_request_ids.clear()
        active_episode_id = getattr(self, "active_episode_id", "")
        if hasattr(self, "reset_pub"):
            self.publish_json(
                self.reset_pub,
                attach_timebase(
                    {
                        "type": "episode_reset",
                        "episode_id": active_episode_id,
                        "reset_scope": "mission_manager",
                        "cleared": ["step_pending", "active_plan", "goal_stop", "stale_cache"],
                    },
                    node=self,
                    episode_id=active_episode_id,
                ),
            )
        if hasattr(self, "metric_pub"):
            self.publish_metric("episode_reset", episode_id=active_episode_id, reset_scope="mission_manager", result="cleared")
        self.publish_state()

    def step_enabled(self) -> bool:
        return bool(deep_get(self.config, "step.enabled", True))

    def omninav_enabled(self) -> bool:
        return bool(deep_get(self.config, "omninav.enabled", True))

    def internnav_enabled(self) -> bool:
        return bool(deep_get(self.config, "internnav.enabled", False))

    def on_user_instruction(self, msg):
        try:
            instruction = parse_user_instruction(msg.data)
        except Exception as exc:
            self.publish_metric("instruction_parse_error", result="parse_error", raw=msg.data, error=repr(exc))
            return
        self.mission_id = instruction.mission_id or new_id("mission")
        self.instruction = instruction.instruction
        self.mission_started_at = instruction.timestamp
        self.last_step_only_trigger = 0.0
        self.goal_stop_until = 0.0
        self.last_goal_stop_publish = 0.0
        self.semantic_goal_triggered = False
        self.semantic_stop_attempts = 0
        self.semantic_retry_pending = False
        self.semantic_retry_due = 0.0
        self.semantic_retry_wait_for_distance = False
        self.semantic_confirmed_decision = None
        self.semantic_last_status = {}
        self.route_choice_triggered = False
        self.was_near_intersection = False
        self.was_near_doorway = False
        if not self.step_enabled():
            self.start_fast_without_step("mission_start", step_result="step_disabled")
            return
        if bool(deep_get(self.config, "step.roles_only", False)):
            self.start_fast_without_step("mission_start", step_result="role_only")
            return
        if not step_trigger_allowed("mission_start", self.config):
            self.start_fast_without_step("mission_start", step_result="trigger_filtered")
            return
        self.state = SchedulerState.STEP_THINK_STOP
        self.publish_state()
        self.publish_json(self.subgoal_pub, {"subgoal": self.instruction, "success_condition": "mission target reached"})
        event = attach_timebase(
            make_event(
            "mission_start",
            mission_id=self.mission_id,
            episode_id=self.active_episode_id,
            reason="mission_start",
            instruction=self.instruction,
            current_subgoal=self.instruction,
            multimodal=False,
            ),
            node=self,
            episode_id=self.active_episode_id,
            mission_id=self.mission_id,
        )
        self.publish_json(self.trigger_pub, event)
        self.publish_metric("mission_start", mission_id=self.mission_id, state=self.state.value, result="step_triggered")

    def on_step_request(self, msg):
        payload = _safe_json(msg.data)
        request_id = str(payload.get("request_id") or "")
        if request_id:
            self.pending_step_request_ids.add(request_id)
        if str(payload.get("role") or "") == "semantic_stop":
            self.semantic_stop_attempts += 1
        pending_stop = bool(payload.get("multimodal")) or str(payload.get("pending_mode") or "") == "stop"
        if pending_stop and self.state != SchedulerState.FAILSAFE:
            self.state = SchedulerState.STEP_THINK_STOP
            self.publish_state()
            self.publish_metric(
                "step_request_pending_stop",
                request_id=request_id,
                episode_id=self.active_episode_id,
                result="hold_position",
            )

    def on_semantic_stop_decision(self, msg):
        payload = _safe_json(msg.data)
        episode_id = str(payload.get("episode_id") or "")
        if episode_id and self.active_episode_id and episode_id != self.active_episode_id:
            self.publish_metric(
                "semantic_stop_scheduler_retry",
                result="episode_mismatch",
                decision_episode_id=episode_id,
                active_episode_id=self.active_episode_id,
            )
            return
        self.semantic_goal_triggered = True
        request_id = str(payload.get("request_id") or "")
        if request_id:
            self.pending_step_request_ids.discard(request_id)
        track = payload.get("track") if isinstance(payload.get("track"), dict) else {}
        visual_stop_confirmed = bool(payload.get("stop")) and bool(track.get("confirmed")) and bool(track.get("visible"))
        actual_distance = _optional_float(getattr(self, "semantic_last_status", {}).get("distance_to_target"))
        stop_threshold = semantic_stop_release_distance_threshold(self.config)
        sensor_only = bool(deep_get(self.config, "runtime_mode.mode_config.sensor_only_planning", False))
        if sensor_only:
            actual_distance = _optional_float(track.get("distance_m"))
        distance_ready = sensor_only or (actual_distance is not None and actual_distance <= stop_threshold)
        if visual_stop_confirmed and distance_ready:
            self.semantic_retry_pending = False
            self.semantic_retry_wait_for_distance = False
            self.semantic_confirmed_decision = None
            self.state = SchedulerState.STEP_THINK_STOP
            self.publish_state()
            self.publish_metric(
                "semantic_stop_scheduler_retry",
                result="confirmed_stop",
                request_id=request_id,
                attempts=self.semantic_stop_attempts,
                frame_seq=track.get("frame_seq"),
                actual_distance_m=actual_distance,
                stop_threshold_m=stop_threshold,
            )
            return
        if visual_stop_confirmed and not distance_ready:
            self.semantic_retry_pending = False
            self.semantic_retry_wait_for_distance = True
            self.semantic_confirmed_decision = dict(payload)
            self.state = SchedulerState.RUN_FAST
            self.publish_state()
            self.publish_metric(
                "semantic_stop_scheduler_retry",
                result="waiting_for_actual_stop_distance",
                request_id=request_id,
                attempts=self.semantic_stop_attempts,
                frame_seq=track.get("frame_seq"),
                actual_distance_m=actual_distance,
                stop_threshold_m=stop_threshold,
            )
            return
        retry_cfg = semantic_stop_retry_config(self.config)
        if retry_cfg["enabled"] and self.semantic_stop_attempts < retry_cfg["max_attempts"]:
            self.semantic_retry_pending = True
            self.semantic_retry_due = now() + retry_cfg["delay_sec"]
            result = "scheduled_fresh_frame_retry"
        else:
            self.semantic_retry_pending = False
            result = "attempts_exhausted"
        self.state = SchedulerState.STEP_THINK_STOP
        self.publish_state()
        self.publish_metric(
            "semantic_stop_scheduler_retry",
            result=result,
            request_id=request_id,
            attempts=self.semantic_stop_attempts,
            max_attempts=retry_cfg["max_attempts"],
            frame_seq=track.get("frame_seq"),
            track_confirmed=bool(track.get("confirmed")),
            target_visible=bool(payload.get("target_visible")),
        )

    def on_route_choice_decision(self, msg):
        payload = _safe_json(msg.data)
        episode_id = str(payload.get("episode_id") or "")
        if episode_id and self.active_episode_id and episode_id != self.active_episode_id:
            self.publish_metric(
                "route_choice_scheduler_resume",
                result="episode_mismatch",
                decision_episode_id=episode_id,
                active_episode_id=self.active_episode_id,
            )
            return
        route_choice = str(payload.get("route_choice") or "").lower()
        if route_choice not in {"left", "right", "front", "scan", "stop"}:
            self.publish_metric("route_choice_scheduler_resume", result="invalid_route_choice", route_choice=route_choice)
            return
        request_id = str(payload.get("request_id") or "")
        if request_id:
            self.pending_step_request_ids.discard(request_id)
        self.route_choice_triggered = True
        self.state = SchedulerState.RUN_FAST
        self.publish_state()
        self.publish_metric(
            "route_choice_scheduler_resume",
            result="run_fast",
            route_choice=route_choice,
            request_id=request_id,
            episode_id=self.active_episode_id,
        )

    def on_semantic_subgoal_accepted(self, msg):
        payload = _safe_json(msg.data)
        episode_id = str(payload.get("episode_id") or "")
        if episode_id and self.active_episode_id and episode_id != self.active_episode_id:
            self.publish_metric(
                "semantic_executive_scheduler_resume",
                result="episode_mismatch",
                decision_episode_id=episode_id,
                active_episode_id=self.active_episode_id,
            )
            return
        request_id = str(payload.get("request_id") or "")
        if request_id:
            self.pending_step_request_ids.discard(request_id)
        if self.state != SchedulerState.FAILSAFE:
            self.state = SchedulerState.RUN_FAST
            self.publish_state()
        self.publish_metric(
            "semantic_executive_scheduler_resume",
            result="run_fast",
            request_id=request_id,
            subgoal_index=payload.get("subgoal_index"),
            subgoal_type=payload.get("subgoal_type"),
            episode_id=self.active_episode_id,
        )

    def on_step_response(self, msg):
        try:
            plan = parse_step_plan(msg.data)
        except Exception as exc:
            self.state = SchedulerState.FAILSAFE
            self.publish_state()
            self.publish_json(self.event_pub, make_event("plan_parse_error", mission_id=self.mission_id, error=repr(exc), raw=msg.data))
            self.publish_metric("step_response_parse_error", mission_id=self.mission_id, result="parse_error", raw=msg.data, error=repr(exc))
            return
        stale_cfg = self.config.get("stale_gate", {})
        old_after_reset = bool(self.active_episode_id and plan.request_id not in self.pending_step_request_ids)
        decision = evaluate_step_plan(
            plan,
            self.current_pose,
            node_ros_now_sec(self),
            stale_cfg,
            current_episode_id=self.active_episode_id,
            current_clock_domain=clock_domain_from_node(self),
            strict_timebase=True,
            old_response_after_reset=old_after_reset,
        )
        if not decision.valid:
            self.state = SchedulerState.STEP_THINK_STOP
            self.publish_state()
            details = decision.to_metric_fields()
            details.setdefault("mission_id", self.mission_id)
            details.setdefault("request_id", plan.request_id)
            self.publish_json(self.event_pub, make_event("stale_result", model="step", **details))
            self.publish_metric("step_response_stale", result="discarded", **details)
            return
        self.pending_step_request_ids.discard(plan.request_id)
        self.active_plan = plan
        if self.goal_stop_until and now() <= self.goal_stop_until:
            self.publish_json(self.constraint_pub, plan.constraints.__dict__)
            self.publish_json(self.subgoal_pub, {"subgoal": plan.subgoal, "success_condition": plan.success_condition})
            self.publish_metric("step_response_during_goal_stop", mission_id=self.mission_id, request_id=plan.request_id, result="held")
            return
        self.state = SchedulerState.RUN_FAST
        self.publish_state()
        self.publish_json(self.constraint_pub, plan.constraints.__dict__)
        self.publish_json(self.subgoal_pub, {"subgoal": plan.subgoal, "success_condition": plan.success_condition})
        if self.internnav_enabled() and not self.omninav_enabled():
            self.publish_metric("step_response_to_internnav", mission_id=self.mission_id, request_id=plan.request_id, result="accepted")
            return
        if not self.omninav_enabled():
            self.publish_json(
                self.primitive_pub,
                attach_timebase(
                {
                    "primitive": "move_forward",
                    "distance_m": 0.35,
                    "yaw_deg": 0.0,
                    "confidence": plan.confidence,
                    "source": "step_direct",
                    "request_id": plan.request_id,
                    "timestamp": node_ros_now_sec(self),
                },
                node=self,
                episode_id=self.active_episode_id,
                mission_id=self.mission_id,
                request_id=plan.request_id,
                source_stamp=plan.timebase.get("source_stamp_sec", plan.timestamp_response) if isinstance(plan.timebase, dict) else plan.timestamp_response,
                ),
            )
            self.publish_metric("step_direct_primitive", mission_id=self.mission_id, request_id=plan.request_id, result="accepted")
            return
        self.publish_json(
            self.omni_req_pub,
            attach_timebase(
            {
                "request_id": new_id("omni_req"),
                "episode_id": self.active_episode_id,
                "mission_id": self.mission_id,
                "timestamp": node_ros_now_sec(self),
                "instruction": plan.navila_or_omninav_instruction,
                "subgoal": plan.subgoal,
                "constraints": plan.constraints.__dict__,
            },
            node=self,
            episode_id=self.active_episode_id,
            mission_id=self.mission_id,
            ),
        )
        self.publish_metric("step_response_accepted", mission_id=self.mission_id, request_id=plan.request_id, stale=False, result="accepted")

    def on_robot_state(self, msg):
        payload = _safe_json(msg.data)
        self.current_pose = list(payload.get("pose") or self.current_pose)

    def on_safety(self, msg):
        self.safety = parse_safety_status(msg.data)
        if bool(deep_get(self.config, "runtime_mode.mode_config.sensor_only_planning", False)):
            self.was_near_intersection = False
            self.was_near_doorway = False
            return
        if not self.mission_id or not self.step_enabled():
            self.was_near_intersection = self.safety.near_intersection
            self.was_near_doorway = self.safety.near_doorway
            return
        if self.safety.near_intersection and not self.was_near_intersection and not self.route_choice_triggered:
            self.route_choice_triggered = True
            self.publish_route_context_event("route_choice_upcoming", reason="near_intersection")
        if self.safety.near_doorway and not self.was_near_doorway:
            self.publish_route_context_event("doorway_choice", reason="near_doorway")
        self.was_near_intersection = self.safety.near_intersection
        self.was_near_doorway = self.safety.near_doorway

    def on_isaac_status(self, msg):
        status = _safe_json(msg.data)
        self.semantic_last_status = dict(status)
        if bool(deep_get(self.config, "runtime_mode.mode_config.sensor_only_planning", False)):
            return
        if self.semantic_retry_wait_for_distance:
            distance = _optional_float(status.get("distance_to_target"))
            if distance is not None and distance <= semantic_stop_release_distance_threshold(self.config):
                self.semantic_retry_wait_for_distance = False
                decision = self.semantic_confirmed_decision or {}
                track = decision.get("track") if isinstance(decision.get("track"), dict) else {}
                if confirmed_track_is_fresh(track, node_ros_now_sec(self), self.config):
                    release_request_id = new_id("semantic_release")
                    release = attach_timebase(
                        dict(decision)
                        | {
                            "request_id": release_request_id,
                            "origin_request_id": str(decision.get("request_id") or ""),
                            "source": "step_confirmed_track_distance_gate",
                            "reason": "confirmed_track_entered_actual_stop_distance",
                            "actual_distance_m": distance,
                            "stop_threshold_m": semantic_stop_release_distance_threshold(self.config),
                        },
                        node=self,
                        episode_id=self.active_episode_id,
                        mission_id=self.mission_id,
                        request_id=release_request_id,
                        source_stamp=node_ros_now_sec(self),
                        created_ros_time=node_ros_now_sec(self),
                    )
                    self.semantic_confirmed_decision = None
                    self.state = SchedulerState.STEP_THINK_STOP
                    self.publish_state()
                    self.publish_json(self.semantic_stop_release_pub, release)
                    self.publish_metric(
                        "semantic_stop_track_release",
                        result="released_confirmed_fresh_track",
                        request_id=release_request_id,
                        origin_request_id=release.get("origin_request_id"),
                        actual_distance_m=distance,
                        stop_threshold_m=semantic_stop_release_distance_threshold(self.config),
                        track_frame_seq=track.get("frame_seq"),
                    )
                else:
                    self.semantic_confirmed_decision = None
                    self.semantic_retry_pending = True
                    self.semantic_retry_due = now()
                    self.publish_metric(
                        "semantic_stop_track_release",
                        result="track_stale_retry_required",
                        actual_distance_m=distance,
                    )
        if not near_goal_stop_needed(status):
            return
        if self.state not in {SchedulerState.RUN_FAST, SchedulerState.STEP_THINK_MOVE, SchedulerState.STEP_THINK_STOP}:
            return
        if not semantic_step_task_allowed(self.config, status):
            if self.semantic_goal_triggered:
                return
            self.semantic_goal_triggered = True
            hold_sec = max(1.2, float(deep_get(self.config, "mission.success_hold_sec", 2.0)))
            self.goal_stop_until = now() + hold_sec + 0.4
            self.state = SchedulerState.STEP_THINK_STOP
            self.publish_state()
            self.publish_near_goal_stop(status, reason="non_semantic_task_near_goal")
            self.publish_metric(
                "semantic_goal_step_filtered",
                mission_id=self.mission_id,
                episode_id=self.active_episode_id,
                task_id=status.get("task_id"),
                task_type=status.get("task_type"),
                result="non_semantic_task",
            )
            return
        step_stop_required = semantic_stop_requires_step(self.config)
        if step_stop_required and self.semantic_goal_triggered:
            return
        self.semantic_goal_triggered = True
        self.state = SchedulerState.STEP_THINK_STOP
        self.publish_state()
        if not step_stop_required:
            hold_sec = max(1.2, float(deep_get(self.config, "mission.success_hold_sec", 2.0)))
            self.goal_stop_until = now() + hold_sec + 0.4
            self.publish_near_goal_stop(status, reason="isaac_status_near_goal")
        self.publish_json(
            self.event_pub,
            make_event(
                "candidate_goal_reached",
                mission_id=self.mission_id,
                episode_id=self.active_episode_id,
                reason="isaac_status_near_goal",
                instruction=self.instruction,
                current_subgoal=(self.active_plan.subgoal if self.active_plan else self.instruction),
                distance_to_target=status.get("distance_to_target"),
                target_visible=status.get("target_visible"),
                multimodal=False,
            ),
        )
        self.publish_metric(
            "semantic_goal_step_trigger",
            mission_id=self.mission_id,
            episode_id=self.active_episode_id,
            result="step_only" if step_stop_required else "pre_stop_then_step",
            distance_to_target=status.get("distance_to_target"),
            target_visible=status.get("target_visible"),
        )

    def on_event(self, msg):
        event = _safe_json(msg.data)
        event_type = event.get("type")
        if event_type in {"mission_start", "stale_result"}:
            return
        if not self.step_enabled():
            return
        if event_type in {
            "candidate_goal_reached",
            "completion_verification",
            "semantic_ambiguity",
            "route_choice_upcoming",
            "doorway_choice",
            "blocked_path_detected",
            "no_progress",
            "forward_bias",
            "omninav_low_confidence_twice",
            "omninav_failed_twice",
            "no_progress_timeout",
            "target_not_seen_after_m",
            "instruction_changed_by_user",
            "safety_stop_requires_explanation",
            "map_semantic_conflict",
        }:
            if not step_trigger_allowed(event_type, self.config):
                self.publish_metric("mission_event_step_filtered", mission_id=self.mission_id, event=event_type, result="trigger_filtered")
                return
            self.state = step_state_for_event(event, self.config)
            self.publish_state()
            event.setdefault("mission_id", self.mission_id)
            event.setdefault("instruction", self.instruction)
            event.setdefault("current_subgoal", self.active_plan.subgoal if self.active_plan else self.instruction)
            self.publish_json(self.trigger_pub, event)
            self.publish_metric("mission_event_step_trigger", mission_id=self.mission_id, event=event_type, result="step_triggered")

    def tick(self):
        if self.state == SchedulerState.IDLE or not self.mission_started_at:
            return
        if self.semantic_retry_pending and now() >= self.semantic_retry_due:
            self.semantic_retry_pending = False
            status = self.semantic_last_status
            event = attach_timebase(
                make_event(
                    "candidate_goal_reached",
                    mission_id=self.mission_id,
                    episode_id=self.active_episode_id,
                    reason="semantic_track_fresh_frame_retry",
                    instruction=self.instruction,
                    current_subgoal=(self.active_plan.subgoal if self.active_plan else self.instruction),
                    distance_to_target=status.get("distance_to_target"),
                    target_visible=status.get("target_visible"),
                    multimodal=False,
                ),
                node=self,
                episode_id=self.active_episode_id,
                mission_id=self.mission_id,
            )
            self.state = SchedulerState.STEP_THINK_STOP
            self.publish_state()
            self.publish_json(self.trigger_pub, event)
            self.publish_metric(
                "semantic_stop_scheduler_retry",
                result="fresh_frame_retry_triggered",
                attempts=self.semantic_stop_attempts,
            )
            return
        if self.goal_stop_until:
            if now() <= self.goal_stop_until:
                if now() - self.last_goal_stop_publish >= 0.4:
                    self.publish_near_goal_stop({}, reason="goal_stop_hold")
                return
            self.goal_stop_until = 0.0
            if self.state == SchedulerState.STEP_THINK_STOP:
                self.state = SchedulerState.RUN_FAST
                self.publish_state()
        if now() - self.mission_started_at > float(deep_get(self.config, "mission.max_duration_sec", 180)):
            self.state = SchedulerState.FAILSAFE
            self.publish_state()
            self.publish_metric("mission_timeout", mission_id=self.mission_id, result="stopped")
            return
        if (
            self.step_enabled()
            and not self.omninav_enabled()
            and self.state == SchedulerState.RUN_FAST
            and not bool(deep_get(self.config, "runtime_mode.external_step_triggers_only", False))
            and periodic_step_only_allowed(self.config)
        ):
            period = float(deep_get(self.config, "runtime_mode.step_only_period_sec", 1.5))
            if now() - self.last_step_only_trigger >= period:
                self.last_step_only_trigger = now()
                event = attach_timebase(
                    make_event(
                    "candidate_goal_reached",
                    mission_id=self.mission_id,
                    episode_id=self.active_episode_id,
                    reason="step_only_next_decision",
                    instruction=self.instruction,
                    current_subgoal=(self.active_plan.subgoal if self.active_plan else self.instruction),
                    multimodal=False,
                    ),
                    node=self,
                    episode_id=self.active_episode_id,
                    mission_id=self.mission_id,
                )
                self.state = step_state_for_event(event, self.config)
                self.publish_state()
                self.publish_json(self.trigger_pub, event)
                self.publish_metric("step_only_next_decision", mission_id=self.mission_id, result="step_triggered")

    def start_fast_without_step(self, reason: str, *, step_result: str):
        self.state = SchedulerState.RUN_FAST
        self.publish_state()
        self.publish_json(self.subgoal_pub, {"subgoal": self.instruction, "success_condition": "mission target reached"})
        if self.omninav_enabled():
            self.publish_initial_omninav_request()
            result = "omninav_only"
        elif self.internnav_enabled():
            result = "internnav_only"
        else:
            result = "no_fast_executor"
        self.publish_metric(
            "mission_start",
            mission_id=self.mission_id,
            state=self.state.value,
            result=result,
            step_result=step_result,
            reason=reason,
        )

    def publish_route_context_event(self, event_type: str, *, reason: str):
        if not step_trigger_allowed(event_type, self.config):
            self.publish_metric("route_context_step_filtered", mission_id=self.mission_id, event=event_type, result="trigger_filtered")
            return
        event = make_event(
            event_type,
            mission_id=self.mission_id,
            reason=reason,
            instruction=self.instruction,
            current_subgoal=(self.active_plan.subgoal if self.active_plan else self.instruction),
            multimodal=False,
        )
        self.state = step_state_for_event(event, self.config)
        self.publish_state()
        self.publish_json(self.trigger_pub, event)
        self.publish_metric("route_context_step_trigger", mission_id=self.mission_id, event=event_type, result="step_triggered")

    def publish_initial_omninav_request(self):
        self.publish_json(
            self.omni_req_pub,
            attach_timebase(
            {
                "request_id": new_id("omni_req"),
                "episode_id": self.active_episode_id,
                "mission_id": self.mission_id,
                "timestamp": node_ros_now_sec(self),
                "instruction": self.instruction,
                "subgoal": self.instruction,
                "constraints": {},
            },
            node=self,
            episode_id=self.active_episode_id,
            mission_id=self.mission_id,
            ),
        )

    def publish_near_goal_stop(self, status: dict, *, reason: str):
        if not bool(deep_get(self.config, "pending_policy.stop_if_near_goal", True)):
            return
        self.last_goal_stop_publish = now()
        request_id = new_id("success_stop")
        self.publish_json(
            self.primitive_pub,
            attach_timebase(
            {
                "primitive": "stop",
                "distance_m": 0.0,
                "yaw_deg": 0.0,
                "confidence": 1.0,
                "source": "isaac_success_bridge",
                "request_id": request_id,
                "timestamp": node_ros_now_sec(self),
                "reason": reason,
                "distance_to_target": status.get("distance_to_target"),
                "target_visible": status.get("target_visible"),
            },
            node=self,
            episode_id=self.active_episode_id,
            mission_id=self.mission_id,
            request_id=request_id,
            ),
        )
        self.publish_metric(
            "near_goal_stop",
            mission_id=self.mission_id,
            request_id=request_id,
            result="stop_hold",
            reason=reason,
            distance_to_target=status.get("distance_to_target"),
            target_visible=status.get("target_visible"),
        )

    def publish_state(self):
        self.publish_json(self.state_pub, self.state.value)

    def publish_metric(self, event_type: str, **kwargs):
        self.publish_json(self.metric_pub, make_metric(event_type, model="none", **kwargs))

    @staticmethod
    def publish_json(pub, payload):
        msg = String()
        msg.data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        pub.publish(msg)


def _safe_json(raw: str) -> dict:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def near_goal_stop_needed(status: dict) -> bool:
    if not isinstance(status, dict):
        return False
    if status.get("success"):
        return False
    if status.get("done") and status.get("reason") not in {"never_stopped", "stopped_too_early"}:
        return False
    return str(status.get("reason") or "") == "never_stopped"


def semantic_stop_requires_step(config: dict) -> bool:
    return bool(deep_get(config, "runtime_mode.mode_config.semantic_stop_requires_step", False))


def periodic_step_only_allowed(config: dict) -> bool:
    return not bool(deep_get(config, "step.roles_only", False))


def semantic_stop_retry_config(config: dict) -> dict[str, object]:
    mode_cfg = deep_get(config, "runtime_mode.mode_config", {}) or {}
    base_cfg = deep_get(config, "step.semantic_stop_retry", {}) or {}
    return {
        "enabled": bool(mode_cfg.get("semantic_stop_retry_enabled", base_cfg.get("enabled", False))),
        "max_attempts": max(1, int(mode_cfg.get("semantic_stop_max_attempts", base_cfg.get("max_attempts", 1)))),
        "delay_sec": max(0.0, float(mode_cfg.get("semantic_stop_retry_delay_sec", base_cfg.get("delay_sec", 0.5)))),
    }


def semantic_stop_actual_distance_threshold(config: dict) -> float:
    mode_cfg = deep_get(config, "runtime_mode.mode_config", {}) or {}
    return max(0.1, float(mode_cfg.get("semantic_stop_actual_distance_m", mode_cfg.get("target_stop_threshold_m", 2.0))))


def semantic_stop_release_distance_threshold(config: dict) -> float:
    mode_cfg = deep_get(config, "runtime_mode.mode_config", {}) or {}
    return max(0.1, float(mode_cfg.get("target_stop_threshold_m", 2.0)))


def confirmed_track_is_fresh(track: dict, timestamp: float, config: dict) -> bool:
    if not bool(track.get("confirmed")) or not bool(track.get("visible")):
        return False
    last_seen = _optional_float(track.get("last_seen_time"))
    if last_seen is None:
        return False
    max_age = float(deep_get(config, "model_clients.step_http.target_track.max_age_sec", 3.0))
    return max(0.0, float(timestamp) - last_seen) <= max_age


def _optional_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def semantic_step_task_allowed(config: dict, status: dict) -> bool:
    mode = str(deep_get(config, "runtime_mode.mode", ""))
    task_only = bool(deep_get(config, "runtime_mode.mode_config.semantic_stop_task_only", False)) or mode in {
        "omninav_step_stop_only_v12_screen",
        "omninav_step_route_stop_v12_screen",
        "omninav_step_route_stop_sim2real_low_speed",
        "step_only_v12_sanity",
    }
    if not task_only:
        return True
    task_type = str(status.get("task_type") or "")
    task_id = str(status.get("task_id") or "")
    return task_type == "semantic_target" or task_id.startswith("semantic_")


TRIGGER_GROUPS = {
    "none": set(),
    "mission_start": {"mission_start"},
    "mission_start_only": {"mission_start"},
    "stuck_replan": {
        "no_progress",
        "forward_bias",
        "no_progress_timeout",
        "omninav_failed_twice",
        "target_not_seen_after_m",
        "safety_stop_requires_explanation",
        "blocked_path_detected",
    },
    "stuck_replan_only": {
        "no_progress",
        "forward_bias",
        "no_progress_timeout",
        "omninav_failed_twice",
        "target_not_seen_after_m",
        "safety_stop_requires_explanation",
        "blocked_path_detected",
    },
    "goal_verify": {"candidate_goal_reached", "completion_verification"},
    "goal_verify_only": {"candidate_goal_reached", "completion_verification"},
    "semantic_ambiguity": {"semantic_ambiguity", "map_semantic_conflict", "ambiguous_target", "semantic_grounding"},
    "semantic_ambiguity_only": {"semantic_ambiguity", "map_semantic_conflict", "ambiguous_target", "semantic_grounding"},
    "route_choice": {"route_choice_upcoming", "doorway_choice", "blocked_path_detected"},
    "route_choice_only": {"route_choice_upcoming", "doorway_choice", "blocked_path_detected"},
    "all_events_current": None,
    "all": None,
}


def step_trigger_allowed(event_type: str, config: dict | None = None) -> bool:
    raw = deep_get(config or {}, "runtime_mode.step_triggers", None)
    if raw is None or raw == "":
        return True
    values = raw if isinstance(raw, list) else [raw]
    allowed: set[str] = set()
    for value in values:
        key = str(value)
        group = TRIGGER_GROUPS.get(key)
        if group is None and key in TRIGGER_GROUPS:
            return True
        if group is not None:
            allowed.update(group)
        else:
            allowed.add(key)
    return str(event_type) in allowed


def step_state_for_event(event: dict, config: dict | None = None) -> SchedulerState:
    pending_mode = str(event.get("pending_mode") or "")
    if pending_mode == "move_slow":
        return SchedulerState.STEP_THINK_MOVE
    if pending_mode == "safe_scan":
        return SchedulerState.STEP_THINK_SCAN
    if pending_mode == "stop":
        return SchedulerState.STEP_THINK_STOP
    event_type = str(event.get("type") or "")
    event_reason = str(event.get("reason") or "")
    if event_type == "candidate_goal_reached" and event_reason == "step_only_next_decision":
        policy = str(deep_get(config or {}, "pending_policy.benchmark_policy", "auto"))
        if policy in {"auto", "move_slow"} and bool(deep_get(config or {}, "pending_policy.allow_move_while_step", True)):
            return SchedulerState.STEP_THINK_MOVE
        return SchedulerState.STEP_THINK_STOP
    hard_stop_events = {
        "candidate_goal_reached",
        "completion_verification",
        "safety_stop_requires_explanation",
        "map_semantic_conflict",
        "semantic_ambiguity",
        "instruction_changed_by_user",
    }
    if event_type in hard_stop_events:
        return SchedulerState.STEP_THINK_STOP
    policy = str(deep_get(config or {}, "pending_policy.benchmark_policy", "auto"))
    if policy == "safe_scan":
        return SchedulerState.STEP_THINK_SCAN
    if policy in {"auto", "move_slow"} and bool(deep_get(config or {}, "pending_policy.allow_move_while_step", True)):
        return SchedulerState.STEP_THINK_MOVE
    return SchedulerState.STEP_THINK_STOP


def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
