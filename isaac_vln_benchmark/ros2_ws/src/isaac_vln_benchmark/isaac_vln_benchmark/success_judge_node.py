from __future__ import annotations

import json

from .config_loader import load_data, object_by_id, scene_by_type
from .metrics import SuccessJudgeCore


def main() -> None:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from std_msgs.msg import String

    class SuccessJudge(Node):
        def __init__(self) -> None:
            super().__init__("isaac_vln_success_judge")
            self.declare_parameter("tasks_path", "configs/tasks.yaml")
            self.declare_parameter("scenes_path", "configs/scenes.yaml")
            self.declare_parameter("task_id", "")
            self.tasks_doc = load_data(self.get_parameter("tasks_path").value)
            self.scenes_doc = load_data(self.get_parameter("scenes_path").value)
            self.task = self._select_task()
            self.scene = scene_by_type(self.scenes_doc, self.task["scene_type"])
            self.target = object_by_id(self.scene, self.task["target_object"])
            self.judge = SuccessJudgeCore()
            self.pose = self.scene.get("robot_start_pose", [0.0, 0.0, 0.0])
            self.cmd = {"linear_x": 0.0, "angular_z": 0.0}
            self.collision = False
            self.start_time = self.get_clock().now()
            self.pub = self.create_publisher(String, "/isaac/episode_status", 10)
            self.pose_sub = self.create_subscription(String, "/isaac/ground_truth_pose", self.on_pose, 10)
            self.cmd_sub = self.create_subscription(Twist, "/safe_cmd_vel", self.on_cmd, 10)
            self.collision_sub = self.create_subscription(String, "/isaac/collision_event", self.on_collision, 10)
            self.timer = self.create_timer(0.2, self.tick)

        def _select_task(self) -> dict:
            wanted = str(self.get_parameter("task_id").value)
            tasks = self.tasks_doc.get("tasks", [])
            if wanted:
                for task in tasks:
                    if task["task_id"] == wanted:
                        return task
            return tasks[0]

        def on_pose(self, msg: String) -> None:
            try:
                data = json.loads(msg.data)
                self.pose = data.get("pose", data.get("robot_pose", self.pose))
            except json.JSONDecodeError:
                pass

        def on_cmd(self, msg: Twist) -> None:
            self.cmd = {"linear_x": float(msg.linear.x), "angular_z": float(msg.angular.z)}

        def on_collision(self, msg: String) -> None:
            self.collision = True

        def tick(self) -> None:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            if elapsed > float(self.task.get("timeout_sec", 120)):
                status = {"success": False, "done": True, "reason": "timeout", "time_sec": elapsed}
            else:
                status = self.judge.evaluate(
                    self.task,
                    self.scene,
                    self.pose,
                    self.target,
                    now_sec=elapsed,
                    cmd_vel=self.cmd,
                    collision=self.collision,
                )
            self.pub.publish(String(data=json.dumps(status)))

    rclpy.init()
    node = SuccessJudge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
