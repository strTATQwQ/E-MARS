from __future__ import annotations

import json
from typing import Any


def publish_reset(node: Any, publisher: Any, task_id: str, scene_id: str, robot_start_pose: list[float] | None = None) -> None:
    from std_msgs.msg import String

    payload = {"task_id": task_id, "scene_id": scene_id, "robot_start_pose": robot_start_pose}
    publisher.publish(String(data=json.dumps(payload)))
    node.get_logger().info(f"Requested Isaac reset for {task_id} in {scene_id}")
