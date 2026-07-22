from __future__ import annotations

import json


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class ObstacleController(Node):
        def __init__(self) -> None:
            super().__init__("isaac_vln_obstacle_controller")
            self.scene = {}
            self.pub = self.create_publisher(String, "/safety/local_status_json", 10)
            self.obs_pub = self.create_publisher(String, "/isaac/objects", 10)
            self.scene_sub = self.create_subscription(String, "/isaac/set_scene", self.on_scene, 10)
            self.timer = self.create_timer(0.2, self.tick)

        def on_scene(self, msg: String) -> None:
            try:
                self.scene = json.loads(msg.data)
            except json.JSONDecodeError:
                self.scene = {}

        def tick(self) -> None:
            dynamic = bool(self.scene.get("dynamic_obstacles"))
            status = {
                "local_costmap_clear": not dynamic,
                "obstacle_distance_m": 0.7 if dynamic else 2.5,
                "human_distance_m": 0.8 if dynamic else None,
                "on_slope_or_stairs": False,
                "near_doorway": any(z.get("class") == "doorway" for z in self.scene.get("semantic_zones", [])),
                "near_intersection": any(z.get("class") == "intersection" for z in self.scene.get("semantic_zones", [])),
                "dynamic_obstacle": dynamic,
                "estop": False,
                "deadman": True,
                "robot_fallen_or_unstable": False,
                "battery_ok": True,
            }
            self.pub.publish(String(data=json.dumps(status)))
            if self.scene:
                self.obs_pub.publish(
                    String(
                        data=json.dumps(
                            {
                                "objects": self.scene.get("objects", []),
                                "obstacles": self.scene.get("obstacles", []),
                                "dynamic_obstacles": self.scene.get("dynamic_obstacles", []),
                            }
                        )
                    )
                )

    rclpy.init()
    node = ObstacleController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
