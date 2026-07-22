from __future__ import annotations

import json
from typing import Any

from .episode_overlay_logger import EpisodeOverlayLogger

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover - exercised on ROS hosts
    rclpy = None
    Node = object
    String = None


class VisualOverlayNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run VisualOverlayNode")
        super().__init__("visual_overlay")
        self.declare_parameter("output_dir", "runs/visual_overlay_live")
        self.logger = EpisodeOverlayLogger(self.get_parameter("output_dir").value)
        self.create_subscription(String, "/visual_debug/overlay_state_json", self.on_overlay, 100)
        self.create_subscription(String, "/mission/event_json", self.on_event, 100)

    def on_overlay(self, msg):
        self.logger.log_overlay(_safe_json(msg.data))

    def on_event(self, msg):
        self.logger.log_event(_safe_json(msg.data))


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": raw}


def main(args=None):
    rclpy.init(args=args)
    node = VisualOverlayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
