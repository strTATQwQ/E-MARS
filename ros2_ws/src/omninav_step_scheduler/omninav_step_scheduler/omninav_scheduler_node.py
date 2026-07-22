from __future__ import annotations

import json
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
    parse_omninav_action,
)
from .runtime_mode import apply_benchmark_mode_config, mode_payload_from_json
from .semantic_executive import validate_semantic_goal_payload
from .stale_gate import evaluate_omninav_action

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover
    rclpy = None
    Node = object
    String = None


class OmniNavSchedulerNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run OmniNavSchedulerNode")
        super().__init__("omninav_scheduler")
        self.declare_parameter("config_file", "")
        self.base_config = load_yaml_file(self.get_parameter("config_file").value)
        self.config = dict(self.base_config)
        self.state = SchedulerState.IDLE.value
        self.pending_mode = "none"
        self.current_pose = [0.0, 0.0, 0.0]
        self.active_subgoal: dict = {}
        self.inflight = False
        self.last_request = 0.0
        self.last_valid_action = now()
        self.low_confidence_count = 0
        self.active_episode_id = ""
        self.pending_request_ids: set[str] = set()
        self.oracle_override_active = False
        self.semantic_goal_active = False
        self.semantic_hold_active = False

        self.request_pub = self.create_publisher(String, "/omninav/request_json", 10)
        self.primitive_pub = self.create_publisher(String, "/primitive/command_json", 10)
        self.event_pub = self.create_publisher(String, "/mission/event_json", 10)
        self.status_pub = self.create_publisher(String, "/omninav/status_json", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/benchmark/mode_json", self.on_benchmark_mode, 10)
        self.create_subscription(String, "/scheduler/state", self.on_state, 10)
        self.create_subscription(String, "/scheduler/step_pending_mode", self.on_pending_mode, 10)
        self.create_subscription(String, "/scheduler/active_subgoal_json", self.on_subgoal, 10)
        self.create_subscription(String, "/omninav/semantic_goal_json", self.on_semantic_goal, 10)
        self.create_subscription(String, "/robot_state_json", self.on_robot_state, 10)
        self.create_subscription(String, "/omninav/request_json", self.on_omninav_request_seen, 10)
        self.create_subscription(String, "/omninav/action_candidate_json", self.on_action_candidate, 10)
        self.create_subscription(String, "/oracle/control_override_json", self.on_oracle_override, 10)
        self.create_timer(1.0 / float(deep_get(self.config, "omninav.target_rate_hz", 6.0)), self.tick)

    def on_benchmark_mode(self, msg):
        payload = mode_payload_from_json(msg.data)
        episode_id = str(payload.get("episode_id", ""))
        if episode_id and episode_id != self.active_episode_id:
            self.active_episode_id = episode_id
            self.inflight = False
            self.low_confidence_count = 0
            self.last_request = 0.0
            self.last_valid_action = now()
            self.pending_request_ids.clear()
            self.oracle_override_active = False
            self.semantic_goal_active = False
            self.semantic_hold_active = False
            self.active_subgoal = {}
            self.publish_json(self.event_pub, make_event("episode_reset", episode_id=episode_id, reset_scope="omninav_scheduler"))
            self.publish_metric("episode_reset", episode_id=episode_id, reset_scope="omninav_scheduler", result="cleared")
        self.config = apply_benchmark_mode_config(self.base_config, payload)
        self.publish_metric(
            "benchmark_mode_applied",
            mode=deep_get(self.config, "runtime_mode.mode", ""),
            omninav_enabled=bool(deep_get(self.config, "omninav.enabled", True)),
        )

    def on_oracle_override(self, msg):
        payload = _safe_json(msg.data)
        episode_id = str(payload.get("episode_id") or "")
        if self.active_episode_id and episode_id and episode_id != self.active_episode_id:
            return
        self.oracle_override_active = bool(payload.get("active", False))
        self.publish_metric(
            "oracle_control_override",
            active=self.oracle_override_active,
            task_id=payload.get("task_id"),
            phase=payload.get("phase"),
        )

    def on_state(self, msg):
        self.state = msg.data
        if self.state == SchedulerState.RUN_FAST.value:
            self.pending_mode = "none"

    def on_pending_mode(self, msg):
        self.pending_mode = msg.data

    def on_subgoal(self, msg):
        if not self.semantic_goal_active:
            self.active_subgoal = _safe_json(msg.data)

    def on_semantic_goal(self, msg):
        payload = _safe_json(msg.data)
        episode_id = str(payload.get("episode_id") or "")
        if self.active_episode_id and episode_id and episode_id != self.active_episode_id:
            self.publish_metric("semantic_goal_episode_mismatch", result="discarded", episode_id=episode_id)
            return
        if bool(payload.get("clear", False)):
            hold = bool(payload.get("hold", False))
            self.semantic_goal_active = hold
            self.semantic_hold_active = hold
            self.active_subgoal = {}
            self.inflight = False
            self.pending_request_ids.clear()
            if hold:
                self._publish_semantic_hold(str(payload.get("reason") or "between_semantic_subgoals"))
            self.publish_metric(
                "semantic_goal_cleared",
                result="held" if hold else "cleared",
                reason=payload.get("reason"),
                hold=hold,
            )
            return
        try:
            semantic_goal = validate_semantic_goal_payload(payload)
        except Exception as exc:
            self.publish_metric(
                "semantic_goal_invalid",
                result="discarded",
                error=repr(exc),
            )
            return
        self.active_subgoal = {
            "semantic_goal": semantic_goal,
            "source": "semantic_executive",
        }
        superseded_requests = len(self.pending_request_ids)
        self.inflight = False
        self.pending_request_ids.clear()
        self.last_request = 0.0
        self.semantic_goal_active = True
        self.semantic_hold_active = semantic_goal["subgoal_type"] in {"verify", "ask"}
        if self.semantic_hold_active:
            self._publish_semantic_hold(semantic_goal["subgoal_type"])
        self.publish_metric(
            "semantic_goal_accepted",
            result="accepted",
            subgoal_type=payload.get("subgoal_type"),
            target=payload.get("target"),
            publishes_motion=False,
            superseded_requests=superseded_requests,
        )

    def _publish_semantic_hold(self, reason: str) -> None:
        hold = attach_timebase(
            {
                "primitive": "stop",
                "ttl_sec": 0.3,
                "source": "omninav_semantic_hold",
                "reason": reason,
            },
            node=self,
            episode_id=self.active_episode_id,
            request_id=new_id("omninav_semantic_hold"),
        )
        self.publish_json(self.primitive_pub, hold)

    def on_robot_state(self, msg):
        payload = _safe_json(msg.data)
        self.current_pose = list(payload.get("pose") or self.current_pose)

    def on_omninav_request_seen(self, msg):
        payload = _safe_json(msg.data)
        request_id = str(payload.get("request_id") or "")
        episode_id = str(payload.get("episode_id") or "")
        if not request_id:
            return
        if self.active_episode_id and episode_id and episode_id != self.active_episode_id:
            return
        self.pending_request_ids.add(request_id)

    def tick(self):
        if not deep_get(self.config, "omninav.enabled", True):
            return
        if self.state != SchedulerState.RUN_FAST.value:
            return
        if self.pending_mode in {"stop", "move_slow", "safe_scan"}:
            return
        if self.semantic_hold_active:
            return
        if self.inflight and int(deep_get(self.config, "omninav.max_inflight_requests", 1)) <= 1:
            return
        if now() - self.last_valid_action > float(deep_get(self.config, "triggers.no_progress_timeout_sec", 15.0)):
            self.publish_json(self.event_pub, make_event("no_progress_timeout", last_valid_action_age=now() - self.last_valid_action))
            self.last_valid_action = now()
        self.inflight = True
        self.last_request = node_ros_now_sec(self)
        request_id = new_id("omni_req")
        req = {
            "request_id": request_id,
            "episode_id": self.active_episode_id,
            "timestamp": self.last_request,
            "timestamp_request": self.last_request,
            "pose": self.current_pose,
            "subgoal": self.active_subgoal,
            "frame_count": deep_get(self.config, "omninav.frame_count", 3),
            "use_left_right": deep_get(self.config, "omninav.use_left_right", True),
            "use_history": deep_get(self.config, "omninav.use_history", False),
        }
        req = attach_timebase(req, node=self, episode_id=self.active_episode_id, request_id=request_id, source_stamp=self.last_request, created_ros_time=self.last_request)
        self.pending_request_ids.add(request_id)
        self.publish_json(self.request_pub, req)
        self.publish_metric("omninav_request", request_id=req["request_id"])

    def on_action_candidate(self, msg):
        self.inflight = False
        receive_ros_time = node_ros_now_sec(self)
        receive_clock_domain = clock_domain_from_node(self)
        try:
            action = parse_omninav_action(msg.data)
        except Exception as exc:
            self.publish_metric("omninav_parse_error", result="parse_error", raw=msg.data, error=repr(exc))
            return
        cfg = dict(self.config.get("stale_gate", {}))
        cfg["max_action_age_sec"] = deep_get(self.config, "omninav.max_action_age_sec", 0.5)
        old_after_reset = bool(self.active_episode_id and action.request_id not in self.pending_request_ids)
        decision = evaluate_omninav_action(
            action,
            self.current_pose,
            receive_ros_time,
            cfg,
            current_episode_id=self.active_episode_id,
            current_clock_domain=receive_clock_domain,
            strict_timebase=True,
            old_response_after_reset=old_after_reset,
        )
        if not decision.valid:
            details = decision.to_metric_fields()
            details.setdefault("request_id", action.request_id)
            self.publish_json(self.event_pub, make_event("stale_result", model="omninav", **details))
            self.publish_metric("omninav_stale", result="discarded", **details)
            return
        self.pending_request_ids.discard(action.request_id)
        if action.confidence < float(deep_get(self.config, "omninav.min_confidence", 0.50)):
            self.low_confidence_count += 1
            self.publish_json(self.event_pub, make_event("low_confidence_once", model="omninav", confidence=action.confidence))
            if self.low_confidence_count >= int(deep_get(self.config, "omninav.low_confidence_count_for_step", 2)):
                self.publish_json(self.event_pub, make_event("omninav_low_confidence_twice", model="omninav", confidence=action.confidence))
                self.low_confidence_count = 0
        else:
            self.low_confidence_count = 0
        self.last_valid_action = now()
        primitive_payload = dict(action.__dict__)
        primitive_payload["model_timebase"] = dict(action.timebase) if isinstance(action.timebase, dict) else {}
        primitive_payload["model_clock_domain"] = action.clock_domain
        primitive_payload["model_source_stamp_sec"] = primitive_payload["model_timebase"].get("source_stamp_sec")
        primitive_payload["model_header_stamp_sec"] = primitive_payload["model_timebase"].get("header_stamp_sec")
        primitive = attach_timebase(
            primitive_payload,
            node=self,
            episode_id=self.active_episode_id,
            mission_id=action.mission_id,
            request_id=action.request_id,
            clock_domain=receive_clock_domain,
            header_stamp=receive_ros_time,
            source_stamp=receive_ros_time,
            created_ros_time=receive_ros_time,
        )
        if self.oracle_override_active:
            self.publish_json(
                self.status_pub,
                attach_timebase(
                    {
                        "timestamp": node_ros_now_sec(self),
                        "last_action_confidence": action.confidence,
                        "inflight": False,
                        "control_suppressed_by_oracle": True,
                    },
                    node=self,
                    episode_id=self.active_episode_id,
                ),
            )
            self.publish_metric(
                "omninav_action",
                request_id=action.request_id,
                confidence=action.confidence,
                result="control_suppressed_by_oracle",
            )
            return
        self.publish_json(self.primitive_pub, primitive)
        self.publish_json(self.status_pub, attach_timebase({"timestamp": node_ros_now_sec(self), "last_action_confidence": action.confidence, "inflight": False}, node=self, episode_id=self.active_episode_id))
        self.publish_metric("omninav_action", request_id=action.request_id, confidence=action.confidence, result="accepted")

    def publish_metric(self, event_type: str, **kwargs):
        self.publish_json(self.metric_pub, make_metric(event_type, model="omninav", state=self.state, **kwargs))

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


def main(args=None):
    rclpy.init(args=args)
    node = OmniNavSchedulerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
