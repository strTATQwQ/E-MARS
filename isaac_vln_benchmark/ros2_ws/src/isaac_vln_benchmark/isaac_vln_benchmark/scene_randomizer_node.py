from __future__ import annotations

import json
import random

from .config_loader import load_data


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class SceneRandomizer(Node):
        def __init__(self) -> None:
            super().__init__("isaac_vln_scene_randomizer")
            self.declare_parameter("scenes_path", "configs/scenes.yaml")
            self.declare_parameter("seed", 42)
            self.scenes = load_data(self.get_parameter("scenes_path").value).get("scenes", [])
            self.rng = random.Random(int(self.get_parameter("seed").value))
            self.pub = self.create_publisher(String, "/isaac/set_scene", 10)
            self.timer = self.create_timer(0.5, self.publish_once)
            self.sent = False

        def publish_once(self) -> None:
            if self.sent or not self.scenes:
                return
            scene = dict(self.rng.choice(self.scenes))
            scene["lighting_seed"] = self.rng.randint(0, 100000)
            scene["texture_seed"] = self.rng.randint(0, 100000)
            self.pub.publish(String(data=json.dumps(scene)))
            self.sent = True

    rclpy.init()
    node = SceneRandomizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
