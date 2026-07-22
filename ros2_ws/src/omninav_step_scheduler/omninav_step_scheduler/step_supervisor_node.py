from __future__ import annotations

import json
import time
from typing import Any

from .schemas import (
    SchedulerState,
    attach_timebase,
    deep_get,
    load_yaml_file,
    make_metric,
    new_id,
    node_ros_now_sec,
    now,
    parse_safety_status,
    parse_user_instruction,
)
from .runtime_mode import apply_benchmark_mode_config, mode_payload_from_json
from .semantic_executive import build_semantic_executive_prompt, sanitize_observable_context
from .step_roles import build_route_choice_prompt, build_semantic_stop_prompt, step_role_for_event

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover - exercised on ROS hosts
    rclpy = None
    Node = object
    String = None


HARD_TRIGGERS = {
    "mission_start",
    "candidate_goal_reached",
    "completion_verification",
    "target_visible",
    "distance_to_target_below_threshold",
    "semantic_ambiguity",
    "omninav_failed_twice",
    "omninav_low_confidence_twice",
    "no_progress_timeout",
    "target_not_seen_after_m",
    "instruction_changed_by_user",
    "safety_stop_requires_explanation",
    "map_semantic_conflict",
    "semantic_plan_requested",
    "semantic_subgoal_completed",
    "semantic_recovery_required",
}

SOFT_TRIGGERS = {
    "periodic_summary_every_m",
    "periodic_summary_every_sec",
    "low_confidence_once",
    "approaching_known_semantic_region",
    "object_candidate_visible",
    "route_choice_upcoming",
}


def forbidden_for_step(safety, timers: dict[str, Any] | None = None) -> bool:
    timers = timers or {}
    if not safety.local_costmap_clear:
        return True
    if safety.dynamic_obstacle or safety.near_intersection or safety.near_doorway:
        return True
    if safety.on_slope_or_stairs:
        return True
    if safety.human_distance_m is not None and safety.human_distance_m < 2.0:
        return True
    if bool(timers.get("robot_turning_fast", False)):
        return True
    if float(timers.get("last_omninav_age", 0.0)) > 0.5:
        return True
    return False


def should_trigger_step(event: dict[str, Any], state: str | SchedulerState, safety, timers: dict[str, Any] | None = None) -> bool:
    event_type = str(event.get("type") or event.get("reason") or "")
    if event_type == "route_choice_upcoming" and safety.near_intersection:
        safe_for_route_choice = (
            safety.local_costmap_clear
            and not safety.dynamic_obstacle
            and not safety.estop
            and not safety.robot_fallen_or_unstable
            and (safety.human_distance_m is None or safety.human_distance_m >= 2.0)
        )
        return bool(safe_for_route_choice and (timers or {}).get("soft_step_allowed", False))
    if forbidden_for_step(safety, timers):
        return False
    if event_type in HARD_TRIGGERS:
        return True
    if event_type == "low_confidence_twice":
        return True
    if event_type in SOFT_TRIGGERS:
        return bool((timers or {}).get("soft_step_allowed", False))
    return False


def trigger_matches_active_episode(event: dict[str, Any], active_episode_id: str) -> bool:
    scoped_id = str(event.get("episode_id") or event.get("mission_id") or "")
    return not scoped_id or bool(active_episode_id and scoped_id == active_episode_id)


def instruction_context(event: dict[str, Any], cached_mission: str, cached_subgoal: str) -> tuple[str, str]:
    mission = (str(event.get("instruction") or event.get("mission") or "") or cached_mission).strip()
    subgoal = (
        str(event.get("active_subgoal") or event.get("current_subgoal") or event.get("subgoal") or "")
        or cached_subgoal
        or mission
    ).strip()
    return mission, subgoal


def choose_step_mode(event: dict[str, Any], safety, multimodal: bool, config: dict[str, Any] | None = None) -> str:
    if safety.estop or safety.robot_fallen_or_unstable or not safety.battery_ok:
        return "stop"
    forced_policy = str(deep_get(config, "pending_policy.benchmark_policy", "auto"))
    if forced_policy in {"stop", "move_slow", "safe_scan"}:
        return forced_policy
    if multimodal and deep_get(config, "pending_policy.stop_if_step_multimodal", True):
        return "stop"
    if safety.dynamic_obstacle or safety.near_intersection or safety.near_doorway or safety.on_slope_or_stairs:
        return "stop"
    obstacle_limit = float(deep_get(config, "pending_policy.stop_if_obstacle_within_m", 0.8))
    human_limit = float(deep_get(config, "pending_policy.stop_if_human_within_m", 2.0))
    if safety.obstacle_distance_m is not None and safety.obstacle_distance_m < obstacle_limit:
        return "stop"
    if safety.human_distance_m is not None and safety.human_distance_m < human_limit:
        return "stop"
    if str(event.get("type", "")) in {"completion_verification", "object_candidate_visible"}:
        return "safe_scan"
    if deep_get(config, "pending_policy.allow_move_while_step", True):
        return "move_slow"
    return "stop"


def role_requires_multimodal(role: str, config: dict[str, Any]) -> bool:
    roles = deep_get(config, "step.multimodal_required_roles", []) or []
    return str(role or "") in {str(item) for item in roles}


def image_only_role_context(
    event: dict[str, Any], semantic_summary: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove simulator truth before constructing a multimodal role request."""
    safe_event = {
        "type": "visual_decision_due",
        "target": event.get("target"),
        "instruction": event.get("instruction"),
        "active_subgoal": event.get("active_subgoal"),
    }
    sensor_track = event.get("track") if isinstance(event.get("track"), dict) else {}
    if (
        str(event.get("distance_basis") or "") == "actual_mask_depth"
        and str(sensor_track.get("source") or "") == "actual_sensor_spatial_track"
        and bool(sensor_track.get("confirmed"))
        and bool(sensor_track.get("fresh"))
    ):
        safe_track_keys = {
            "episode_id",
            "target_id",
            "hits",
            "misses",
            "confirmed",
            "visible",
            "confidence",
            "distance_m",
            "bearing_rad",
            "frame_seq",
            "age_sec",
            "fresh",
            "source",
            "source_frames",
        }
        safe_event.update(
            {
                "sensor_track": {key: sensor_track[key] for key in safe_track_keys if key in sensor_track},
                "distance_to_target_m": sensor_track.get("distance_m"),
                "allow_distance_context": True,
                "distance_basis": "actual_mask_depth",
            }
        )
    forbidden = {
        "target_visible",
        "distance_to_target_m",
        "distance_to_target",
        "expected_route",
        "correct_branch",
        "oracle_visibility",
        "target_present",
        "ground_truth",
    }
    safe_summary = {key: value for key, value in semantic_summary.items() if str(key).lower() not in forbidden}
    return safe_event, safe_summary


def build_step_prompt(
    mission: str,
    current_subgoal: str,
    robot_state: dict[str, Any],
    semantic_summary: dict[str, Any],
    omninav_status: dict[str, Any],
    safety_status: dict[str, Any],
    event: dict[str, Any],
) -> dict[str, Any]:
    system = "You are a semantic supervisor for a quadruped robot. You do not output motor commands. Return JSON only."
    user = {
        "Mission": mission,
        "Current subgoal": current_subgoal,
        "Robot state": robot_state,
        "Visible objects": semantic_summary,
        "OmniNav status": omninav_status,
        "Safety status": safety_status,
        "Event reason": event,
        "Return JSON": {
            "omninav_instruction": "...",
            "subgoal": "...",
            "success_condition": "...",
            "constraints": {
                "max_speed_mps": 0.2,
                "avoid_people": True,
                "stop_if_uncertain": True,
                "forbidden_zones": [],
            },
            "replan_triggers": [],
            "recommended_pending_mode": "stop|move_slow|safe_scan",
            "confidence": 0.0,
        },
    }
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]
    }


class StepSupervisorNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run StepSupervisorNode")
        super().__init__("step_supervisor")
        self.declare_parameter("config_file", "")
        self.base_config = load_yaml_file(self.get_parameter("config_file").value)
        self.config = dict(self.base_config)
        self.safety = parse_safety_status({})
        self.state = SchedulerState.IDLE.value
        self.mission = ""
        self.subgoal = ""
        self.robot_state: dict[str, Any] = {}
        self.semantic_summary: dict[str, Any] = {}
        self.omninav_status: dict[str, Any] = {}
        self.last_call_time = 0.0
        self.call_times: list[float] = []
        self.active_episode_id = ""

        self.step_pub = self.create_publisher(String, "/step/request_json", 10)
        self.mode_pub = self.create_publisher(String, "/scheduler/step_pending_mode", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/benchmark/mode_json", self.on_benchmark_mode, 10)
        self.create_subscription(String, "/scheduler/step_trigger_json", self.on_trigger, 10)
        self.create_subscription(String, "/safety/local_status_json", self.on_safety, 10)
        self.create_subscription(String, "/scheduler/state", self.on_state, 10)
        self.create_subscription(String, "/user_instruction", self.on_instruction, 10)
        self.create_subscription(String, "/scheduler/active_subgoal_json", self.on_active_subgoal, 10)
        self.create_subscription(String, "/robot_state_json", self.on_robot_state, 10)
        self.create_subscription(String, "/semantic_summary_json", self.on_semantic_summary, 10)
        self.create_subscription(String, "/omninav/status_json", self.on_omninav_status, 10)

    def on_benchmark_mode(self, msg):
        payload = mode_payload_from_json(msg.data)
        episode_id = str(payload.get("episode_id", ""))
        if episode_id and episode_id != self.active_episode_id:
            self.active_episode_id = episode_id
            self.last_call_time = 0.0
            self.call_times = []
            self.mission = ""
            self.subgoal = ""
            self.publish_metric("episode_reset", episode_id=episode_id, reset_scope="step_supervisor", result="cleared")
        self.config = apply_benchmark_mode_config(self.base_config, payload)
        self.publish_metric(
            "benchmark_mode_applied",
            mode=deep_get(self.config, "runtime_mode.mode", ""),
            step_enabled=bool(deep_get(self.config, "step.enabled", True)),
            pending_policy=deep_get(self.config, "pending_policy.benchmark_policy", "auto"),
        )

    def on_safety(self, msg):
        self.safety = parse_safety_status(msg.data)

    def on_state(self, msg):
        self.state = msg.data

    def on_instruction(self, msg):
        try:
            self.mission = parse_user_instruction(msg.data).instruction
        except Exception:
            self.mission = msg.data

    def on_active_subgoal(self, msg):
        payload = _safe_json(msg.data)
        subgoal = str(payload.get("subgoal") or "").strip()
        if subgoal:
            self.subgoal = subgoal

    def on_robot_state(self, msg):
        self.robot_state = _safe_json(msg.data)

    def on_semantic_summary(self, msg):
        self.semantic_summary = _safe_json(msg.data)

    def on_omninav_status(self, msg):
        self.omninav_status = _safe_json(msg.data)

    def on_trigger(self, msg):
        event = _safe_json(msg.data)
        if not trigger_matches_active_episode(event, self.active_episode_id):
            self.publish_metric("step_trigger_pre_mode_ignored", event=event, result="ignored_pre_mode")
            return
        if not bool(deep_get(self.config, "step.enabled", True)):
            self.publish_metric("step_disabled", event=event, result="disabled")
            return
        if bool(deep_get(self.config, "runtime_mode.external_step_triggers_only", False)) and not str(
            event.get("source") or ""
        ).startswith("v4_step_"):
            self.publish_metric("step_external_trigger_filtered", event=event, result="filtered")
            return
        if bool(deep_get(self.config, "runtime_mode.external_step_triggers_only", False)) and self.last_call_time > 0.0:
            self.publish_metric("step_external_trigger_duplicate", event=event, result="filtered")
            return
        role = step_role_for_event(event)
        if bool(deep_get(self.config, "step.roles_only", False)) and role == "disabled":
            self.publish_metric("step_role_filtered", event=event, role=role, result="disabled_role")
            return
        timestamp = node_ros_now_sec(self)
        min_interval = float(deep_get(self.config, "step.min_interval_sec", 8.0))
        max_per_min = int(deep_get(self.config, "step.max_calls_per_minute", 6))
        self.call_times = [t for t in self.call_times if timestamp - t < 60.0]
        timers = {"soft_step_allowed": timestamp - self.last_call_time >= min_interval, "last_omninav_age": event.get("last_omninav_age", 0.0)}
        if len(self.call_times) >= max_per_min:
            self.publish_metric("step_rate_limited", event=event)
            return
        if not should_trigger_step(event, self.state, self.safety, timers):
            mode_msg = String()
            mode_msg.data = "stop" if forbidden_for_step(self.safety, timers) else "none"
            self.mode_pub.publish(mode_msg)
            self.publish_metric("step_not_triggered", event=event, result="forbidden_or_soft")
            return

        multimodal = bool(
            event.get("multimodal", deep_get(self.config, "step.multimodal_policy", "event_only") == "always")
            or role_requires_multimodal(role, self.config)
        )
        mode = choose_step_mode(event, self.safety, multimodal, self.config)
        mode_msg = String()
        mode_msg.data = mode
        self.mode_pub.publish(mode_msg)
        req = self.build_request(event, multimodal, mode)
        out = String()
        out.data = json.dumps(req, ensure_ascii=False)
        self.step_pub.publish(out)
        self.last_call_time = timestamp
        self.call_times.append(timestamp)
        self.publish_metric("step_request", event=event, request_id=req["request_id"], pending_mode=mode)

    def build_request(self, event: dict[str, Any], multimodal: bool, mode: str) -> dict[str, Any]:
        request_id = new_id("step_req")
        ts = node_ros_now_sec(self)
        mission, subgoal = instruction_context(event, self.mission, self.subgoal)
        role = step_role_for_event(event)
        allow_oracle_context = bool(deep_get(self.config, "step.allow_oracle_visibility_context", False))
        prompt_event = event
        prompt_summary = self.semantic_summary
        if role in {"route_choice", "semantic_stop"} and not allow_oracle_context:
            prompt_event, prompt_summary = image_only_role_context(event, self.semantic_summary)
        elif role == "semantic_executive" and not allow_oracle_context:
            prompt_event = sanitize_observable_context(event)
            prompt_summary = sanitize_observable_context(self.semantic_summary)
        prompt = build_step_prompt(
            mission,
            subgoal,
            self.robot_state,
            prompt_summary,
            self.omninav_status,
            self.safety.__dict__,
            prompt_event,
        )
        if role == "route_choice":
            prompt = build_route_choice_prompt(
                instruction=mission,
                active_subgoal=subgoal,
                robot_state=self.robot_state,
                semantic_summary=prompt_summary,
                safety_status=self.safety.__dict__,
                event=prompt_event,
            )
        elif role == "semantic_stop":
            prompt = build_semantic_stop_prompt(
                instruction=mission,
                active_subgoal=subgoal,
                robot_state=self.robot_state,
                semantic_summary=prompt_summary,
                safety_status=self.safety.__dict__,
                event=prompt_event,
            )
        elif role == "semantic_executive":
            observation_history = prompt_summary.get("observation_history", [])
            if not isinstance(observation_history, list):
                observation_history = [prompt_summary]
            prompt = build_semantic_executive_prompt(
                instruction=mission,
                active_subgoal={"text": subgoal} if subgoal else None,
                observation_history=observation_history,
                event=sanitize_observable_context(prompt_event),
            )
        payload = {
            "request_id": request_id,
            "episode_id": self.active_episode_id,
            "mission_id": str(event.get("mission_id") or ""),
            "timestamp_request": ts,
            "reason": prompt_event.get("type", prompt_event.get("reason", "unknown")),
            "role": role,
            "multimodal": multimodal,
            "pose_at_request": self.robot_state.get("pose", [0.0, 0.0, 0.0]),
            "pending_mode": mode,
            "instruction": mission,
            "active_subgoal": subgoal,
            "semantic_summary": prompt_summary,
            "target": event.get("target") or subgoal,
            "sensor_track": prompt_event.get("sensor_track", {}),
            "visual_ground_truth_hidden": not allow_oracle_context,
            "endpoint": deep_get(self.config, "step.endpoint", "http://127.0.0.1:8080/v1/chat/completions"),
            "model": deep_get(self.config, "step.model", "Step-3.7-Flash"),
            "max_tokens": deep_get(
                self.config,
                "semantic_executive.max_tokens" if role == "semantic_executive" else "step.max_tokens",
                160 if role == "semantic_executive" else 80,
            ),
            "temperature": deep_get(self.config, "step.temperature", 0.0),
            "prompt": prompt,
        }
        if allow_oracle_context:
            payload["target_visible"] = event.get(
                "target_visible", self.semantic_summary.get("target_visible", False)
            )
            payload["distance_m"] = event.get(
                "distance_to_target_m",
                event.get("distance_to_target", self.semantic_summary.get("distance_to_target_m")),
            )
        return attach_timebase(
            payload,
            node=self,
            episode_id=self.active_episode_id,
            mission_id=str(event.get("mission_id") or ""),
            request_id=request_id,
            source_stamp=ts,
            created_ros_time=ts,
        )

    def publish_metric(self, event_type: str, **kwargs):
        msg = String()
        msg.data = json.dumps(make_metric(event_type, model="step", **kwargs), ensure_ascii=False)
        self.metric_pub.publish(msg)


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def main(args=None):
    rclpy.init(args=args)
    node = StepSupervisorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
