from __future__ import annotations

import json
from typing import Any

from .schemas import make_metric
from .step_roles import step_role_for_event

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover - exercised on ROS hosts
    rclpy = None
    Node = object
    String = None


class StepRoleRouterNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run StepRoleRouterNode")
        super().__init__("step_role_router")
        self.route_pub = self.create_publisher(String, "/step/route_choice_trigger_json", 10)
        self.stop_pub = self.create_publisher(String, "/step/semantic_stop_trigger_json", 10)
        self.semantic_pub = self.create_publisher(String, "/step/semantic_executive_trigger_json", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/scheduler/step_trigger_json", self.on_trigger, 10)
        self.create_subscription(String, "/mission/event_json", self.on_trigger, 10)

    def on_trigger(self, msg):
        event = _safe_json(msg.data)
        role = step_role_for_event(event)
        if role == "route_choice":
            self.publish_json(self.route_pub, event)
        elif role == "semantic_stop":
            self.publish_json(self.stop_pub, event)
        elif role == "semantic_executive":
            self.publish_json(self.semantic_pub, event)
        else:
            self.publish_json(self.metric_pub, make_metric("step_role_router_disabled", event=event, role=role))
            return
        self.publish_json(self.metric_pub, make_metric("step_role_router", event=event, role=role, result="routed"))

    @staticmethod
    def publish_json(pub, payload):
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        pub.publish(out)


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def main(args=None):
    rclpy.init(args=args)
    node = StepRoleRouterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
