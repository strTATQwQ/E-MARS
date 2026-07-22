from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from std_msgs.msg import String

    class BenchmarkLogger(Node):
        def __init__(self) -> None:
            super().__init__("isaac_vln_benchmark_logger")
            self.declare_parameter("output_dir", "runs/live_episode")
            self.output_dir = Path(str(self.get_parameter("output_dir").value))
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.events_path = self.output_dir / "events.jsonl"
            topics = [
                "/scheduler/state",
                "/step/request_json",
                "/step/response_json",
                "/omninav/request_json",
                "/omninav/action_candidate_json",
                "/primitive/command_json",
                "/isaac/collision_event",
                "/isaac/episode_status",
                "/mission/event_json",
            ]
            self.subs = [self.create_subscription(String, topic, lambda msg, t=topic: self.on_string(t, msg), 10) for topic in topics]
            self.cmd_sub = self.create_subscription(Twist, "/safe_cmd_vel", self.on_cmd, 10)

        def now(self) -> float:
            return self.get_clock().now().nanoseconds / 1e9

        def write(self, record: dict) -> None:
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")

        def on_string(self, topic: str, msg: String) -> None:
            try:
                details = json.loads(msg.data)
            except json.JSONDecodeError:
                details = {"raw": msg.data}
            self.write({"t": self.now(), "event": topic.strip("/").replace("/", "_"), "details": details})

        def on_cmd(self, msg: Twist) -> None:
            self.write(
                {
                    "t": self.now(),
                    "event": "cmd_vel",
                    "details": {"linear_x": msg.linear.x, "angular_z": msg.angular.z},
                }
            )

    rclpy.init()
    node = BenchmarkLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
