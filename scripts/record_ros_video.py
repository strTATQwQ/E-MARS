#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


def image_message_to_rgb(message: Any) -> bytes:
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    encoding = str(message.encoding).lower()
    if width <= 0 or height <= 0 or step <= 0:
        raise ValueError("invalid image dimensions")
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding {encoding!r}")
    raw = bytes(message.data)
    if len(raw) < step * height:
        raise ValueError("image data shorter than step*height")
    output = bytearray(width * height * 3)
    output_offset = 0
    for row in range(height):
        source = raw[row * step : row * step + width * channels]
        for column in range(width):
            offset = column * channels
            if encoding.startswith("bgr"):
                red, green, blue = source[offset + 2], source[offset + 1], source[offset]
            else:
                red, green, blue = source[offset], source[offset + 1], source[offset + 2]
            output[output_offset : output_offset + 3] = bytes((red, green, blue))
            output_offset += 3
    return bytes(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a bounded representative ROS Image stream to MP4.")
    parser.add_argument("--topic", default="/camera/front/isaac_image")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-duration-sec", type=float, default=300.0)
    args = parser.parse_args(argv)

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    output = Path(args.output)
    metadata_path = Path(args.metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"stop": False}
    signal.signal(signal.SIGTERM, lambda *_: state.__setitem__("stop", True))
    signal.signal(signal.SIGINT, lambda *_: state.__setitem__("stop", True))

    rclpy.init(args=None)
    node = Node("omninav_representative_video_recorder")
    process: subprocess.Popen[bytes] | None = None
    frame_count = 0
    dropped = 0
    first_sha = ""
    last_sha = ""
    dimensions: tuple[int, int] | None = None
    started = time.time()

    def on_image(message: Any) -> None:
        nonlocal process, frame_count, dropped, first_sha, last_sha, dimensions
        try:
            rgb = image_message_to_rgb(message)
            current_dimensions = (int(message.width), int(message.height))
            if dimensions is None:
                dimensions = current_dimensions
                command = [
                    "ffmpeg", "-y", "-loglevel", "warning", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{dimensions[0]}x{dimensions[1]}", "-r", str(args.fps), "-i", "-",
                    "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(output),
                ]
                process = subprocess.Popen(command, stdin=subprocess.PIPE)
            if current_dimensions != dimensions or process is None or process.stdin is None:
                dropped += 1
                return
            digest = hashlib.sha256(rgb).hexdigest()
            first_sha = first_sha or digest
            last_sha = digest
            process.stdin.write(rgb)
            frame_count += 1
        except Exception:
            dropped += 1

    node.create_subscription(Image, args.topic, on_image, 2)
    try:
        while rclpy.ok() and not state["stop"] and time.time() - started < args.max_duration_sec:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=30.0)
    metadata = {
        "topic": args.topic,
        "source": "real_isaac_render_product_udp_15012",
        "started_at": started,
        "ended_at": time.time(),
        "fps": args.fps,
        "frames": frame_count,
        "dropped_frames": dropped,
        "width": dimensions[0] if dimensions else 0,
        "height": dimensions[1] if dimensions else 0,
        "first_frame_sha256": first_sha,
        "last_frame_sha256": last_sha,
        "output": str(output),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if frame_count <= 0 or not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("no representative video frames were recorded")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
