from __future__ import annotations

import json
from typing import Any

from .runtime_mode import mode_payload_from_json
from .schemas import attach_timebase, load_yaml_file, make_metric, now
from .step_roles import route_choice_verifier, step_role_for_event

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover - exercised on ROS hosts
    rclpy = None
    Node = object
    String = None


class RouteChoiceVerifierNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run RouteChoiceVerifierNode")
        super().__init__("route_choice_verifier")
        self.declare_parameter("config_file", "")
        self.config = load_yaml_file(self.get_parameter("config_file").value)
        self.instruction = ""
        self.active_subgoal = ""
        self.semantic_summary: dict[str, Any] = {}
        self.safety_status: dict[str, Any] = {}
        self.active_episode_id = ""
        self.active_mode_config: dict[str, Any] = {}

        self.route_pub = self.create_publisher(String, "/step/route_choice_json", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/benchmark/mode_json", self.on_benchmark_mode, 10)
        self.create_subscription(String, "/user_instruction", self.on_instruction, 10)
        self.create_subscription(String, "/scheduler/active_subgoal_json", self.on_active_subgoal, 10)
        self.create_subscription(String, "/semantic_summary_json", self.on_semantic_summary, 10)
        self.create_subscription(String, "/safety/local_status_json", self.on_safety, 10)
        self.create_subscription(String, "/step/route_choice_trigger_json", self.on_trigger, 10)
        self.create_subscription(String, "/mission/event_json", self.on_trigger, 10)

    def on_benchmark_mode(self, msg):
        payload = mode_payload_from_json(msg.data)
        mode_config = payload.get("mode_config") if isinstance(payload.get("mode_config"), dict) else {}
        self.active_mode_config = dict(mode_config)
        episode_id = str(payload.get("episode_id", ""))
        if episode_id and episode_id != self.active_episode_id:
            self.active_episode_id = episode_id
            self.publish_metric("episode_reset", episode_id=episode_id, reset_scope="route_choice_verifier", result="cleared")

    def on_instruction(self, msg):
        payload = _safe_json(msg.data)
        self.instruction = str(payload.get("instruction") or msg.data or "")

    def on_active_subgoal(self, msg):
        payload = _safe_json(msg.data)
        self.active_subgoal = str(payload.get("subgoal") or payload.get("active_subgoal") or "")

    def on_semantic_summary(self, msg):
        self.semantic_summary = _safe_json(msg.data)

    def on_safety(self, msg):
        self.safety_status = _safe_json(msg.data)

    def on_trigger(self, msg):
        if self.active_mode_config and not bool(self.active_mode_config.get("use_step", False)):
            return
        if bool(self.active_mode_config.get("real_step_required", False)):
            return
        event = _safe_json(msg.data)
        if step_role_for_event(event) != "route_choice":
            return
        result = route_choice_verifier(
            instruction=str(event.get("instruction") or self.instruction),
            active_subgoal=str(event.get("active_subgoal") or event.get("subgoal") or self.active_subgoal),
            semantic_summary=self.semantic_summary,
            intersection_context={"visible_in_view": event.get("visible_in_view", "none"), "safety": self.safety_status},
        )
        result = attach_timebase(
            dict(result) | {"request_id": str(event.get("request_id") or "")},
            node=self,
            episode_id=str(event.get("episode_id") or self.active_episode_id),
            mission_id=str(event.get("mission_id") or ""),
            request_id=str(event.get("request_id") or ""),
        )
        self.publish_json(self.route_pub, result)
        self.publish_metric("step_route_choice_json", event=event, output=result, result="published")

    def publish_metric(self, event_type: str, **kwargs):
        self.publish_json(self.metric_pub, make_metric(event_type, model="step_route_choice_verifier", **kwargs))

    @staticmethod
    def publish_json(pub, payload):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        pub.publish(msg)


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw, "timestamp": now()}


def main(args=None):
    rclpy.init(args=args)
    node = RouteChoiceVerifierNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
