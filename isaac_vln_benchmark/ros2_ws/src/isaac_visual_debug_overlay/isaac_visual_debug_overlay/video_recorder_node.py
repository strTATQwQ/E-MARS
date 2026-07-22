from __future__ import annotations

try:
    import rclpy
    from rclpy.node import Node
except Exception:  # pragma: no cover - exercised on ROS hosts
    rclpy = None
    Node = object


class VideoRecorderNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is required to run VideoRecorderNode")
        super().__init__("video_recorder")
        self.get_logger().info("PNG/MP4 rendering is handled by scripts/render_episode_video.py")


def main(args=None):
    rclpy.init(args=args)
    node = VideoRecorderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
