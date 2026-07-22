#!/usr/bin/env python3
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


WATCH_TOPICS = [
    "/scheduler/state",
    "/step/request_json",
    "/step/response_json",
    "/omninav/request_json",
    "/primitive/command_json",
    "/metrics/event_jsonl",
]


class Smoke(Node):
    def __init__(self):
        super().__init__("smoke_test_scheduler")
        self.seen = {topic: 0 for topic in WATCH_TOPICS}
        self.pub = self.create_publisher(String, "/user_instruction", 10)
        for topic in WATCH_TOPICS:
            self.create_subscription(String, topic, lambda msg, t=topic: self.on_msg(t, msg), 50)
        self.create_timer(0.5, self.publish_once)
        self.sent = False

    def publish_once(self):
        if self.sent:
            return
        msg = String()
        msg.data = "Go to the red exit sign and stop near the fire extinguisher."
        self.pub.publish(msg)
        self.sent = True
        self.get_logger().info("sent smoke test instruction")

    def on_msg(self, topic, msg):
        self.seen[topic] += 1
        preview = msg.data[:180].replace("\n", " ")
        self.get_logger().info(f"{topic}: {preview}")


def main():
    rclpy.init()
    node = Smoke()
    deadline = time.time() + 8.0
    while rclpy.ok() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    print(json.dumps(node.seen, indent=2))
    missing = [topic for topic, count in node.seen.items() if count == 0]
    node.destroy_node()
    rclpy.shutdown()
    if missing:
        raise SystemExit(f"missing expected topics: {missing}")


if __name__ == "__main__":
    main()
