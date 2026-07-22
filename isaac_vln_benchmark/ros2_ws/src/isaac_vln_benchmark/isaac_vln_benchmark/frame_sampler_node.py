from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    class FrameSampler(Node):
        def __init__(self) -> None:
            super().__init__("isaac_vln_frame_sampler")
            self.declare_parameter("output_dir", "runs/live_frames")
            self.declare_parameter("history_size", 8)
            self.output_dir = Path(str(self.get_parameter("output_dir").value))
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.history = []
            self.pub = self.create_publisher(String, "/benchmark/frame_bundle_json", 10)
            self.sub = self.create_subscription(Image, "/camera/front/image", self.on_image, 10)

        def on_image(self, msg: Image) -> None:
            seq = len(self.history)
            # Avoid stuffing image bytes into String. Store metadata; image persistence can be enabled later via cv_bridge.
            record = {
                "view": "front",
                "seq": seq,
                "stamp_sec": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9,
                "width": int(msg.width),
                "height": int(msg.height),
                "encoding": msg.encoding,
                "file_path": None,
            }
            self.history.append(record)
            self.history = self.history[-int(self.get_parameter("history_size").value) :]
            self.pub.publish(String(data=json.dumps({"current_front": record, "rolling_history_front_N": self.history})))

    rclpy.init()
    node = FrameSampler()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
