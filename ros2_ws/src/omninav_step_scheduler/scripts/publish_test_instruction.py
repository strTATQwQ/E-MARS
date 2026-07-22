#!/usr/bin/env python3
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Publisher(Node):
    def __init__(self, instruction: str):
        super().__init__("publish_test_instruction")
        self.pub = self.create_publisher(String, "/user_instruction", 10)
        self.instruction = instruction
        self.create_timer(0.5, self.tick)
        self.sent = False

    def tick(self):
        if self.sent:
            return
        msg = String()
        msg.data = self.instruction
        self.pub.publish(msg)
        self.get_logger().info(f"published /user_instruction: {self.instruction}")
        self.sent = True


def main():
    rclpy.init()
    instruction = " ".join(sys.argv[1:]) or "Go to the red exit sign and stop near the fire extinguisher."
    node = Publisher(instruction)
    deadline = time.time() + 2.0
    while rclpy.ok() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
