from __future__ import annotations

import json
from pathlib import Path

from .config_loader import load_data, scene_by_type


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class EpisodeManager(Node):
        def __init__(self) -> None:
            super().__init__("isaac_vln_episode_manager")
            self.declare_parameter("tasks_path", "configs/tasks.yaml")
            self.declare_parameter("scenes_path", "configs/scenes.yaml")
            self.declare_parameter("mode", "step_omninav_event")
            self.declare_parameter("start_paused", False)
            self.tasks = load_data(self.get_parameter("tasks_path").value).get("tasks", [])
            self.scenes_doc = load_data(self.get_parameter("scenes_path").value)
            self.index = -1
            self.waiting = bool(self.get_parameter("start_paused").value)
            self.user_pub = self.create_publisher(String, "/user_instruction", 10)
            self.reset_pub = self.create_publisher(String, "/isaac/reset_episode", 10)
            self.scene_pub = self.create_publisher(String, "/isaac/set_scene", 10)
            self.event_pub = self.create_publisher(String, "/mission/event_json", 10)
            self.status_sub = self.create_subscription(String, "/isaac/episode_status", self.on_status, 10)
            self.timer = self.create_timer(1.0, self.tick)

        def tick(self) -> None:
            if self.waiting:
                return
            self.index += 1
            if self.index >= len(self.tasks):
                self.get_logger().info("All benchmark tasks dispatched")
                self.waiting = True
                return
            task = self.tasks[self.index]
            scene = scene_by_type(self.scenes_doc, task["scene_type"])
            self.scene_pub.publish(String(data=json.dumps(scene)))
            self.reset_pub.publish(String(data=json.dumps({"task_id": task["task_id"], "scene_id": scene["scene_id"]})))
            self.user_pub.publish(String(data=task["instruction"]))
            self.event_pub.publish(
                String(data=json.dumps({"event": "episode_start", "task_id": task["task_id"], "scene_id": scene["scene_id"]}))
            )
            self.get_logger().info(f"Started {task['task_id']}: {task['instruction']}")
            self.waiting = True

        def on_status(self, msg: String) -> None:
            try:
                status = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            if status.get("done"):
                self.event_pub.publish(String(data=json.dumps({"event": "episode_done", "status": status})))
                self.waiting = False

    rclpy.init()
    node = EpisodeManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
