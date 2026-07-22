#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Replay(Node):
    def __init__(self, path: Path, speed: float):
        super().__init__("replay_episode")
        self.events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.speed = speed
        self.publishers = {}

    def pub(self, topic: str):
        if topic not in self.publishers:
            self.publishers[topic] = self.create_publisher(String, topic, 10)
        return self.publishers[topic]

    def run(self):
        last_ts = None
        for event in self.events:
            topic = event.get("topic")
            payload = event.get("payload")
            if not topic:
                continue
            ts = float(event.get("timestamp", time.time()))
            if last_ts is not None:
                time.sleep(max(0.0, (ts - last_ts) / self.speed))
            last_ts = ts
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.pub(topic).publish(msg)
            rclpy.spin_once(self, timeout_sec=0.01)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    rclpy.init()
    node = Replay(args.jsonl, args.speed)
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
