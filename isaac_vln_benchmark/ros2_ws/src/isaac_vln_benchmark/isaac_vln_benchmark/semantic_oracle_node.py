from __future__ import annotations

import json

from .metrics import bearing_deg, distance_xy, object_visible


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class SemanticOracle(Node):
        def __init__(self) -> None:
            super().__init__("isaac_vln_semantic_oracle")
            self.declare_parameter("semantic_noise_prob", 0.0)
            self.pose = [0.0, 0.0, 0.0]
            self.objects = []
            self.obstacles = []
            self.pub = self.create_publisher(String, "/semantic_summary_json", 10)
            self.obj_sub = self.create_subscription(String, "/isaac/objects", self.on_objects, 10)
            self.pose_sub = self.create_subscription(String, "/isaac/ground_truth_pose", self.on_pose, 10)
            self.timer = self.create_timer(0.5, self.tick)

        def on_objects(self, msg: String) -> None:
            try:
                data = json.loads(msg.data)
                self.objects = data.get("objects", data if isinstance(data, list) else [])
                self.obstacles = data.get("obstacles", [])
            except json.JSONDecodeError:
                pass

        def on_pose(self, msg: String) -> None:
            try:
                data = json.loads(msg.data)
                self.pose = data.get("pose", self.pose)
            except json.JSONDecodeError:
                pass

        def tick(self) -> None:
            visible = []
            for obj in self.objects:
                is_visible = object_visible(self.pose, obj, blockers=self.obstacles)
                if not is_visible:
                    continue
                visible.append(
                    {
                        "id": obj.get("id"),
                        "class": obj.get("class"),
                        "bearing_deg": round(bearing_deg(self.pose, obj.get("pose", [0, 0, 0])), 1),
                        "distance_m": round(distance_xy(self.pose, obj.get("pose", [0, 0, 0])), 2),
                        "visible": True,
                        "confidence": 1.0,
                    }
                )
            summary = {
                "visible_objects": visible,
                "nearby_landmarks": visible[:3],
                "frontier_summary": "oracle primitive scene",
                "route_context": "intersection_ahead" if any(abs(v["bearing_deg"]) < 30 for v in visible) else "corridor",
            }
            self.pub.publish(String(data=json.dumps(summary)))

    rclpy.init()
    node = SemanticOracle()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
