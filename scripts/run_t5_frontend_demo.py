#!/usr/bin/env python3
"""Serve a local visual-only mock of the T5 read-only dashboard."""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "slow_planner_frontend" / "static" / "index.html"
VIEWS = (
    "front_left",
    "front",
    "front_right",
    "rear",
    "go2_front",
    "d435_color",
    "d435_depth",
)


def _camera_jpeg(view: str) -> bytes:
    width, height = (640, 360) if view == "go2_front" else (640, 480)
    palettes = {
        "front_left": ((12, 36, 40), (45, 132, 111)),
        "front": ((13, 31, 34), (72, 143, 117)),
        "front_right": ((16, 29, 37), (49, 112, 124)),
        "rear": ((30, 25, 35), (106, 73, 108)),
        "go2_front": ((14, 24, 28), (91, 116, 91)),
        "d435_color": ((26, 32, 28), (118, 131, 88)),
        "d435_depth": ((17, 16, 45), (18, 157, 169)),
    }
    low, high = palettes[view]
    image = Image.new("RGB", (width, height), low)
    pixels = image.load()
    for y in range(height):
        ratio = y / max(height - 1, 1)
        for x in range(width):
            vignette = 0.82 + 0.18 * math.sin(math.pi * x / width)
            pixels[x, y] = tuple(
                int((low[index] * (1 - ratio) + high[index] * ratio) * vignette)
                for index in range(3)
            )
    draw = ImageDraw.Draw(image, "RGBA")
    horizon = int(height * 0.45)
    draw.rectangle((0, horizon, width, height), fill=(4, 12, 12, 62))
    for offset in range(-4, 5):
        x = width // 2 + offset * width // 10
        draw.line((x, height, width // 2 + offset * 16, horizon), fill=(164, 236, 207, 48), width=2)
    draw.rounded_rectangle((24, 24, 235, 83), radius=12, fill=(4, 12, 12, 165), outline=(118, 227, 187, 100), width=1)
    draw.text((42, 38), view.replace("_", " ").upper(), fill=(232, 248, 242, 255), font=ImageFont.load_default(size=18))
    draw.text((42, 62), "LOCAL VISUAL MOCK", fill=(118, 227, 187, 230), font=ImageFont.load_default(size=11))
    if view == "d435_depth":
        for index, color in enumerate(((72, 35, 116, 180), (18, 157, 169, 150), (247, 212, 63, 130))):
            radius = 42 + index * 38
            draw.ellipse((width * 0.64 - radius, height * 0.57 - radius, width * 0.64 + radius, height * 0.57 + radius), fill=color)
        image = image.filter(ImageFilter.GaussianBlur(2))
    output = io.BytesIO()
    image.save(output, "JPEG", quality=84, optimize=True)
    return output.getvalue()


CAMERAS = {view: _camera_jpeg(view) for view in VIEWS}


def _state() -> dict:
    now = time.time()
    rev_c = [
        {
            "view_id": view,
            "available": True,
            "url": f"/api/v1/cameras/{view}.jpg",
            "sim_stamp_s": 248.125,
            "age_s": 0.08,
        }
        for view in VIEWS[:4]
    ]
    live = [
        {
            "view_id": view,
            "available": True,
            "url": f"/api/v1/cameras/{view}.jpg",
            "stamp_s": now - 0.12,
            "age_s": 0.12,
        }
        for view in VIEWS[4:]
    ]
    topic = lambda name, count: {
        "topic": name,
        "received": True,
        "fresh": True,
        "age_s": 0.06,
        "count": count,
    }
    return {
        "schema_version": 1,
        "lane_id": "b",
        "readonly": True,
        "version": int(now),
        "updated_wall_time_s": now,
        "health": {
            "ready": True,
            "status": "demo",
            "model_variant": "Step3-VL-10B BF16",
            "revision": "5026053b…",
            "precision_mode": "bf16",
        },
        "gpu": {
            "unified_memory_available_mib": 81742,
            "unified_memory_total_mib": 122880,
            "system_swap_used_mib": 0,
            "gpu_util_percent": 63,
        },
        "latency": {"end_to_end_ms": 184.3, "peak_memory_mib": 37648.0},
        "decision": {
            "decision": {
                "mode": "bounded_advisor",
                "intent": "frontier_advice",
                "confidence": 0.84,
                "source_decision": "Step3 structured JSON",
                "scene_summary": "Open corridor ahead; doorway visible on the right.",
                "target_evidence": ["bright doorway", "clear forward path"],
                "blocked_directions": ["rear"],
                "recommended_frontier": 7,
                "target_found": False,
                "abstain": False,
                "fallback_used": False,
            }
        },
        "snapshot": {"snapshot_id": "b::demo-01::0::42", "episode_id": "demo-01", "reset_id": 0},
        "cameras": rev_c,
        "ros": {
            "ready": True,
            "status": "live mock",
            "topics": {
                "low_state": topic("/lowstate", 28421),
                "sport_mode_state": topic("/sportmodestate", 17013),
                "lidar_state": topic("/utlidar/lidar_state", 921),
                "d435_color": topic("/check/d435/color/image_raw", 904),
                "d435_depth": topic("/check/d435/depth/image_rect_raw", 904),
            },
        },
        "robot": {
            "battery": {"soc": 61, "cycle": 2, "power_v": 29.1, "power_a": 0.14},
            "imu": {"rpy": [-0.012, -0.084, 0.120]},
            "motion": {
                "position": [-0.074, -0.011, 0.054],
                "velocity": [0.0, 0.0, 0.0],
                "yaw_speed": 0.003,
            },
            "lidar": {"error_state": 0, "cloud_frequency": 15.4, "cloud_packet_loss_rate": 0.0},
        },
        "live_cameras": live,
        "ingest_warnings": [],
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
            return
        if path in {"/api/v1/state", "/api/v1/health"}:
            state = _state()
            payload = state if path.endswith("state") else {
                key: state[key] for key in ("schema_version", "lane_id", "readonly", "health", "gpu", "ros")
            }
            self._send(200, "application/json", json.dumps(payload).encode())
            return
        prefix = "/api/v1/cameras/"
        if path.startswith(prefix) and path.endswith(".jpg"):
            view = path[len(prefix) : -4]
            body = CAMERAS.get(view)
            if body is not None:
                self._send(200, "image/jpeg", body)
                return
        self._send(404, "application/json", b'{"detail":"not found"}')

    def log_message(self, message: str, *args: object) -> None:
        print(f"[frontend-demo] {message % args}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"T5 frontend visual mock: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
