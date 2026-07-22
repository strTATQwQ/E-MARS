from __future__ import annotations

import json

from .config_loader import load_data, object_by_id, scene_by_type


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class GoalOracle(Node):
        def __init__(self) -> None:
            super().__init__("isaac_vln_goal_oracle")
            self.declare_parameter("tasks_path", "configs/tasks.yaml")
            self.declare_parameter("scenes_path", "configs/scenes.yaml")
            self.tasks = {t["instruction"]: t for t in load_data(self.get_parameter("tasks_path").value).get("tasks", [])}
            self.scenes_doc = load_data(self.get_parameter("scenes_path").value)
            self.pub = self.create_publisher(String, "/mission/event_json", 10)
            self.sub = self.create_subscription(String, "/user_instruction", self.on_instruction, 10)

        def on_instruction(self, msg: String) -> None:
            task = self.tasks.get(msg.data)
            if not task:
                return
            scene = scene_by_type(self.scenes_doc, task["scene_type"])
            target = object_by_id(scene, task["target_object"])
            self.pub.publish(String(data=json.dumps({"event": "goal_oracle", "task_id": task["task_id"], "target": target})))

    rclpy.init()
    node = GoalOracle()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
