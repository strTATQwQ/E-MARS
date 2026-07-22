from __future__ import annotations

import json
from typing import Any

from .schemas import load_yaml_file, make_metric, now
from .step_roles import VisibleToStopMonitor

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover - exercised on ROS hosts
    rclpy = None
    Node = object
    String = None


class VisibleToStopMonitorNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run VisibleToStopMonitorNode")
        super().__init__("visible_to_stop_monitor")
        self.declare_parameter("config_file", "")
        self.config = load_yaml_file(self.get_parameter("config_file").value)
        gate_cfg = self.config.get("semantic_stop_gate", {}) if isinstance(self.config.get("semantic_stop_gate"), dict) else {}
        self.monitor = VisibleToStopMonitor(
            target_visible_required_frames=int(gate_cfg.get("target_visible_required_frames", 2)),
            max_distance_to_target_m=float(gate_cfg.get("max_distance_to_target_m", 2.5)),
        )
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.state_pub = self.create_publisher(String, "/step/visible_to_stop_monitor_json", 10)
        self.create_subscription(String, "/benchmark/mode_json", self.on_benchmark_mode, 10)
        self.create_subscription(String, "/isaac/episode_status_json", self.on_status, 10)
        self.create_subscription(String, "/primitive/command_json", self.on_primitive, 10)

    def on_benchmark_mode(self, msg):
        self.monitor = VisibleToStopMonitor(
            target_visible_required_frames=self.monitor.target_visible_required_frames,
            max_distance_to_target_m=self.monitor.max_distance_to_target_m,
        )

    def on_status(self, msg):
        status = _safe_json(msg.data)
        summary = self.monitor.update(
            timestamp=now(),
            target_visible=bool(status.get("target_visible", False)),
            distance_m=_float(status.get("distance_to_target_m", status.get("distance_to_target"))),
        )
        self.publish_json(self.state_pub, summary)

    def on_primitive(self, msg):
        primitive = _safe_json(msg.data)
        if str(primitive.get("primitive") or "") != "stop":
            return
        summary = self.monitor.update(
            timestamp=now(),
            target_visible=True,
            distance_m=_float(primitive.get("distance_to_target_m")),
            stop_command=True,
        )
        self.publish_json(self.state_pub, summary)
        self.publish_json(self.metric_pub, make_metric("visible_to_stop_monitor", **summary))

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


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main(args=None):
    rclpy.init(args=args)
    node = VisibleToStopMonitorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
