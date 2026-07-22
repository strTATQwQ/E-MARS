from __future__ import annotations

import json
import time
from typing import Any

from .internnav_bridge import InternNavProgressMonitor, normalize_action_name, primitive_motion
from .runtime_mode import apply_benchmark_mode_config, mode_payload_from_json
from .schemas import SchedulerState, attach_timebase, clock_domain_from_node, deep_get, load_yaml_file, make_event, make_metric, node_ros_now_sec, now
from .stale_gate import pose_delta_m, yaw_delta_deg

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover
    rclpy = None
    Node = object
    String = None


class InternNavSchedulerNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run InternNavSchedulerNode")
        super().__init__("internnav_scheduler")
        self.declare_parameter("config_file", "")
        self.base_config = load_yaml_file(self.get_parameter("config_file").value)
        self.config = dict(self.base_config)
        self.state = SchedulerState.IDLE.value
        self.current_pose = [0.0, 0.0, 0.0]
        self.active_episode_id = ""
        pm_cfg = deep_get(self.config, "progress_monitor", {}) or {}
        self.monitor = InternNavProgressMonitor(
            window_sec=float(pm_cfg.get("window_sec", 8.0)),
            min_progress_m=float(pm_cfg.get("min_progress_m", 0.5)),
            max_forward_without_goal_update=int(pm_cfg.get("max_forward_without_goal_update", 6)),
            max_same_action_ratio=float(pm_cfg.get("max_same_action_ratio", 0.85)),
        )

        self.primitive_pub = self.create_publisher(String, "/primitive/command_json", 10)
        self.event_pub = self.create_publisher(String, "/mission/event_json", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/benchmark/mode_json", self.on_benchmark_mode, 10)
        self.create_subscription(String, "/scheduler/state", self.on_state, 10)
        self.create_subscription(String, "/robot_state_json", self.on_robot_state, 10)
        self.create_subscription(String, "/internnav/action_json", self.on_action, 20)

    def on_benchmark_mode(self, msg):
        payload = mode_payload_from_json(msg.data)
        episode_id = str(payload.get("episode_id", ""))
        if episode_id and episode_id != self.active_episode_id:
            self.active_episode_id = episode_id
            self.monitor.reset()
        self.config = apply_benchmark_mode_config(self.base_config, payload)
        self.publish_metric(
            "benchmark_mode_applied",
            mode=deep_get(self.config, "runtime_mode.mode", ""),
            internnav_enabled=self.internnav_enabled(),
        )

    def on_state(self, msg):
        self.state = msg.data

    def on_robot_state(self, msg):
        payload = _safe_json(msg.data)
        pose = payload.get("pose")
        if isinstance(pose, list) and len(pose) >= 3:
            self.current_pose = [float(pose[0]), float(pose[1]), float(pose[2])]

    def internnav_enabled(self) -> bool:
        return bool(deep_get(self.config, "internnav.enabled", False))

    def on_action(self, msg):
        action = _safe_json(msg.data)
        if not self.internnav_enabled() or self.state not in {
            SchedulerState.RUN_FAST.value,
            SchedulerState.STEP_THINK_MOVE.value,
        }:
            return
        if not self.action_is_fresh(action):
            attribution = "episode_mismatch" if self.active_episode_id and str(action.get("episode_id") or "") and str(action.get("episode_id")) != self.active_episode_id else "stale_due_to_age"
            self.publish_json(self.event_pub, make_event("stale_result", model="internnav", request_id=action.get("request_id"), attribution=attribution, discard_reason=attribution))
            self.publish_metric("internnav_stale", request_id=action.get("request_id"), attribution=attribution, result="discarded")
            return

        model_action = normalize_action_name(action.get("model_action", "unknown"))
        applied_action = normalize_action_name(action.get("applied_action", model_action))
        target_distance = _float_or_none(action.get("target_distance_m"))
        decision = self.monitor.update(
            timestamp=float(action.get("timestamp_response", now()) or now()),
            pose=list(action.get("pose_at_snapshot") or self.current_pose),
            target_distance_m=target_distance,
            model_action=model_action,
            applied_action=applied_action,
        )

        recovery_override = bool(action.get("recovery_override", False))
        recovery_reason = action.get("recovery_reason")
        primitive = primitive_motion(applied_action)
        mode = str(deep_get(self.config, "runtime_mode.mode", ""))
        if decision.trigger:
            if mode_uses_internnav_recovery(mode):
                recovery_override = True
                recovery_reason = decision.trigger
                primitive = self.recovery_primitive(decision.recovery_primitive)
                if mode.startswith("step_internnav"):
                    self.publish_json(
                        self.event_pub,
                        make_event(decision.trigger, reason=decision.reason or decision.trigger, model="internnav"),
                    )
            else:
                self.publish_json(
                    self.event_pub,
                    make_event(decision.trigger, reason=decision.reason or decision.trigger, model="internnav"),
                )

        primitive.update(
            {
                "source": "internnav",
                "episode_id": self.active_episode_id,
                "request_id": action.get("request_id"),
                "timestamp_request": action.get("timestamp_request"),
                "timestamp_response": action.get("timestamp_response"),
                "pose_at_snapshot": action.get("pose_at_snapshot") or self.current_pose,
                "confidence": action.get("confidence"),
                "model_action": model_action,
                "applied_action": applied_action,
                "recovery_override": recovery_override,
                "recovery_reason": recovery_reason,
                "ttl_sec": action.get("ttl_sec", 0.5),
            }
        )
        primitive = attach_timebase(
            primitive,
            node=self,
            episode_id=self.active_episode_id,
            request_id=str(action.get("request_id") or ""),
            clock_domain=str(action.get("clock_domain") or clock_domain_from_node(self)),
            source_stamp=action.get("source_stamp_sec") or action.get("timestamp_response") or node_ros_now_sec(self),
            created_ros_time=node_ros_now_sec(self),
        )
        self.publish_json(self.primitive_pub, primitive)
        self.publish_metric(
            "internnav_primitive",
            request_id=action.get("request_id"),
            result="accepted",
            model_action=model_action,
            applied_action=applied_action,
            primitive=primitive.get("primitive"),
            forward_ratio=decision.forward_ratio,
            same_action_ratio=decision.same_action_ratio,
            action_entropy=decision.action_entropy,
            recovery_override=recovery_override,
            recovery_reason=recovery_reason,
        )

    def action_is_fresh(self, action: dict[str, Any]) -> bool:
        episode_id = str(action.get("episode_id") or "")
        if self.active_episode_id and episode_id and episode_id != self.active_episode_id:
            return False
        try:
            timestamp_response = float(action["timestamp_response"])
        except Exception:
            return False
        max_age = min(
            float(action.get("ttl_sec", deep_get(self.config, "internnav.max_action_age_sec", 0.5))),
            float(deep_get(self.config, "internnav.max_action_age_sec", 0.5)),
        )
        if time.time() - timestamp_response > max_age:
            return False
        pose = action.get("pose_at_snapshot")
        if isinstance(pose, list) and len(pose) >= 3:
            if pose_delta_m(pose, self.current_pose) >= float(deep_get(self.config, "stale_gate.omninav_pose_delta_m", 0.30)):
                return False
            if yaw_delta_deg(pose[2], self.current_pose[2]) >= float(deep_get(self.config, "stale_gate.omninav_yaw_delta_deg", 15.0)):
                return False
        return True

    @staticmethod
    def recovery_primitive(kind: str | None) -> dict[str, Any]:
        if kind == "move_forward":
            return {"primitive": "move_forward", "distance_m": 0.50, "yaw_deg": 0.0}
        if kind == "turn_left":
            return {"primitive": "turn_left", "distance_m": 0.0, "yaw_deg": 15.0}
        if kind == "stop":
            return {"primitive": "stop", "distance_m": 0.0, "yaw_deg": 0.0}
        return {"primitive": "look_around", "distance_m": 0.0, "yaw_deg": 15.0}

    def publish_metric(self, event_type: str, **kwargs):
        self.publish_json(self.metric_pub, make_metric(event_type, model="internnav", state=self.state, **kwargs))

    @staticmethod
    def publish_json(pub, payload):
        msg = String()
        msg.data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        pub.publish(msg)


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def mode_uses_internnav_recovery(mode: str) -> bool:
    return mode == "internnav_only" or mode.startswith("step_internnav")


def main(args=None):
    rclpy.init(args=args)
    node = InternNavSchedulerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
