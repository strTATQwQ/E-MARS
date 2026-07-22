#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from PIL import Image as PilImage
from rclpy.node import Node
from sensor_msgs.msg import Image


def image_to_rgb(msg: Image) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    channels = 4 if encoding in {"rgba8", "bgra8"} else 3
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape((int(msg.height), int(msg.step)))
    image = rows[:, : int(msg.width) * channels].reshape((int(msg.height), int(msg.width), channels))
    if encoding in {"bgr8", "bgra8"}:
        image = image[..., :3][..., ::-1]
    else:
        image = image[..., :3]
    return image.copy()


class CaptureNode(Node):
    def __init__(self, topic: str):
        super().__init__("capture_ros_image_once")
        self.message: Image | None = None
        self.create_subscription(Image, topic, self.on_image, 2)

    def on_image(self, msg: Image) -> None:
        if self.message is None:
            self.message = msg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/camera/front/image")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = CaptureNode(args.topic)
    deadline = time.monotonic() + args.timeout_sec
    while node.message is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    msg = node.message
    node.destroy_node()
    rclpy.shutdown()
    if msg is None:
        raise SystemExit(f"no image received from {args.topic}")

    image = image_to_rgb(msg)
    variants = {
        "raw": image,
        "horizontal": image[:, ::-1, :],
        "vertical": image[::-1, :, :],
        "both": image[::-1, ::-1, :],
    }
    for name, pixels in variants.items():
        PilImage.fromarray(pixels).save(output / f"{name}.jpg", quality=95)
    gray = image.astype(np.float32).mean(axis=2)
    metadata = {
        "topic": args.topic,
        "width": int(msg.width),
        "height": int(msg.height),
        "encoding": str(msg.encoding),
        "step": int(msg.step),
        "frame_id": str(msg.header.frame_id),
        "pixel_mean_rgb": [round(float(value), 4) for value in image.mean(axis=(0, 1))],
        "pixel_std_rgb": [round(float(value), 4) for value in image.std(axis=(0, 1))],
        "black_pixel_fraction": round(float((gray < 5.0).mean()), 6),
        "bright_pixel_fraction": round(float((gray > 245.0).mean()), 6),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
