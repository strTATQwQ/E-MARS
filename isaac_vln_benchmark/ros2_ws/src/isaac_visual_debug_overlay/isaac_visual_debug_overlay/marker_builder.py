from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import Any


COLORS = {
    "white": (245, 245, 245),
    "black": (20, 20, 20),
    "gray": (170, 170, 170),
    "dark_gray": (70, 70, 70),
    "red": (220, 60, 60),
    "blue": (60, 110, 220),
    "green": (60, 160, 90),
    "yellow": (235, 195, 70),
    "orange": (230, 140, 40),
}


class MarkerBuilder:
    def __init__(self, width: int = 960, height: int = 540, world_scale: float = 60.0):
        self.width = width
        self.height = height
        self.world_scale = world_scale
        self.origin = (90, height // 2)

    def build_overlay_state(
        self,
        *,
        t: float,
        robot_pose: list[float],
        trajectory: list[dict[str, Any]],
        scene: dict[str, Any],
        mode: str,
        active_subgoal: str,
        primitive: str,
        step_json: dict[str, Any] | None,
        stale_status: str,
        safety_status: str,
        distance_to_target: float,
        target_visible: bool,
        visible_to_stop_latency: float | None,
        entered_correct_branch: bool | None,
        stop_decision: bool | None,
    ) -> dict[str, Any]:
        return {
            "t": round(t, 3),
            "robot_pose": robot_pose,
            "trajectory_len": len(trajectory),
            "target": scene.get("target"),
            "success_radius_m": scene.get("success_radius_m", 2.0),
            "mode": mode,
            "active_subgoal": active_subgoal,
            "current_primitive": primitive,
            "current_waypoint": _waypoint_ahead(robot_pose),
            "step_trigger_reason": "route_or_stop_event" if step_json else "none",
            "step_json": step_json or {},
            "stale_gate_status": stale_status,
            "safety_mux_status": safety_status,
            "distance_to_target": round(distance_to_target, 3),
            "target_visible": bool(target_visible),
            "visible_to_stop_latency": visible_to_stop_latency,
            "entered_correct_branch": entered_correct_branch,
            "stop_decision": stop_decision,
        }

    def render_png(self, path: str | Path, state: dict[str, Any], scene: dict[str, Any], trajectory: list[dict[str, Any]]) -> None:
        image = bytearray(COLORS["white"] * self.width * self.height)
        self._draw_corridor(image)
        self._draw_objects(image, scene)
        self._draw_trajectory(image, trajectory)
        pose = state.get("robot_pose") or [0.0, 0.0, 0.0]
        self._circle(image, *self._world_to_px(float(pose[0]), float(pose[1])), 8, COLORS["green"])
        self._draw_success_radius(image, scene)
        _write_png(path, self.width, self.height, bytes(image))

    def _draw_corridor(self, image: bytearray) -> None:
        y1 = self.height // 2 - 72
        y2 = self.height // 2 + 72
        self._line(image, 60, y1, self.width - 80, y1, COLORS["gray"])
        self._line(image, 60, y2, self.width - 80, y2, COLORS["gray"])
        self._line(image, 300, y1, 300, 70, COLORS["gray"])
        self._line(image, 300, y2, 300, self.height - 70, COLORS["gray"])

    def _draw_objects(self, image: bytearray, scene: dict[str, Any]) -> None:
        for obj in (scene.get("world") or {}).get("objects", []):
            x, y = self._world_to_px(float(obj.get("x", 0.0)), float(obj.get("y", 0.0)))
            color = COLORS.get(str(obj.get("color", "orange")), COLORS["orange"])
            self._rect(image, x - 7, y - 7, x + 7, y + 7, color)

    def _draw_trajectory(self, image: bytearray, trajectory: list[dict[str, Any]]) -> None:
        points = [self._world_to_px(float(row.get("x", 0.0)), float(row.get("y", 0.0))) for row in trajectory]
        for a, b in zip(points, points[1:]):
            self._line(image, a[0], a[1], b[0], b[1], COLORS["blue"])

    def _draw_success_radius(self, image: bytearray, scene: dict[str, Any]) -> None:
        target_name = scene.get("target")
        for obj in (scene.get("world") or {}).get("objects", []):
            if obj.get("name") != target_name:
                continue
            x, y = self._world_to_px(float(obj.get("x", 0.0)), float(obj.get("y", 0.0)))
            radius = int(float(scene.get("success_radius_m", 2.0)) * self.world_scale)
            self._circle_outline(image, x, y, radius, COLORS["yellow"])

    def _world_to_px(self, x: float, y: float) -> tuple[int, int]:
        return int(self.origin[0] + x * self.world_scale), int(self.origin[1] - y * self.world_scale)

    def _set(self, image: bytearray, x: int, y: int, color: tuple[int, int, int]) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        idx = (y * self.width + x) * 3
        image[idx : idx + 3] = bytes(color)

    def _line(self, image: bytearray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self._set(image, x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def _rect(self, image: bytearray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(max(0, y0), min(self.height, y1 + 1)):
            for x in range(max(0, x0), min(self.width, x1 + 1)):
                self._set(image, x, y, color)

    def _circle(self, image: bytearray, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                    self._set(image, x, y, color)

    def _circle_outline(self, image: bytearray, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
        steps = max(48, int(radius * 3))
        for i in range(steps):
            theta = 2.0 * math.pi * i / steps
            self._set(image, int(cx + radius * math.cos(theta)), int(cy + radius * math.sin(theta)), color)


def _waypoint_ahead(pose: list[float]) -> list[float]:
    x, y, yaw = [float(v) for v in (pose + [0.0, 0.0, 0.0])[:3]]
    return [round(x + math.cos(yaw) * 0.5, 3), round(y + math.sin(yaw) * 0.5, 3), round(yaw, 3)]


def _write_png(path: str | Path, width: int, height: int, rgb: bytes) -> None:
    rows = []
    stride = width * 3
    for y in range(height):
        rows.append(b"\x00" + rgb[y * stride : (y + 1) * stride])
    raw = b"".join(rows)
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    Path(path).write_bytes(png)
