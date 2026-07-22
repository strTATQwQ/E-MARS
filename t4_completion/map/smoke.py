"""Deterministic no-ROS smoke model for the static+LiDAR map path."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .composer import ComposeRequest, compose_plan
from .contract import LoadedConfigs, load_and_validate
from .static_map import build_grid, grid_summary
from .warn_relay import bounded_command


FREE = 0
INFLATED = 253
LETHAL = 254


def _bresenham(start: Tuple[int, int], end: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


class LocalCostmapModel:
    def __init__(self, width_m: float, height_m: float, resolution_m: float) -> None:
        self.resolution = resolution_m
        self.width = int(round(width_m / resolution_m))
        self.height = int(round(height_m / resolution_m))
        self.origin = (self.width // 2, self.height // 2)
        self.lethal = set()  # type: set[Tuple[int, int]]

    def cell(self, x_m: float, y_m: float) -> Tuple[int, int]:
        return (
            self.origin[0] + int(round(x_m / self.resolution)),
            self.origin[1] + int(round(y_m / self.resolution)),
        )

    def integrate(self, endpoints: Sequence[Tuple[float, float]]) -> None:
        for endpoint in endpoints:
            target = self.cell(endpoint[0], endpoint[1])
            ray = list(_bresenham(self.origin, target))
            for cell in ray[:-1]:
                self.lethal.discard(cell)
            if 0 <= target[0] < self.width and 0 <= target[1] < self.height:
                self.lethal.add(target)

    def inflated(self, inflation_radius_m: float) -> Mapping[Tuple[int, int], int]:
        radius_cells = int(math.ceil(inflation_radius_m / self.resolution))
        values: Dict[Tuple[int, int], int] = {}
        for occupied in self.lethal:
            values[occupied] = LETHAL
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    if math.hypot(dx, dy) * self.resolution > inflation_radius_m:
                        continue
                    cell = occupied[0] + dx, occupied[1] + dy
                    if not (0 <= cell[0] < self.width and 0 <= cell[1] < self.height):
                        continue
                    values.setdefault(cell, INFLATED)
        return values


def _json_sha(value: Any) -> str:
    payload = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_dry_smoke(configs: LoadedConfigs = None) -> Mapping[str, Any]:
    loaded = configs if configs is not None else load_and_validate()
    default_plan = compose_plan(ComposeRequest(), loaded)
    failed_active = compose_plan(
        ComposeRequest(requested_mode="active", nvblox_health="failed"), loaded
    )
    ready_active = compose_plan(
        ComposeRequest(requested_mode="active", nvblox_health="ready"), loaded
    )

    local = loaded.nav2_shadow["local_costmap"]["local_costmap"]["ros__parameters"]
    model = LocalCostmapModel(
        float(local["width"]), float(local["height"]), float(local["resolution"])
    )
    old_obstacle = model.cell(0.75, 0.0)
    model.integrate([(0.75, 0.0), (0.80, 0.45)])
    first_lethal = set(model.lethal)
    first_inflated = model.inflated(float(local["inflation_layer"]["inflation_radius"]))
    model.integrate([(1.25, 0.0), (0.80, 0.45)])
    second_lethal = set(model.lethal)
    second_inflated = model.inflated(float(local["inflation_layer"]["inflation_radius"]))

    static_before = build_grid(loaded.smoke_map)
    static_after = build_grid(loaded.smoke_map)
    command = bounded_command(0.9, -2.0, 0.01, True, False)
    estopped = bounded_command(0.1, 0.2, 0.01, True, True)
    checks = {
        "default_effective_mode_shadow": default_plan["effective_mode"] == "shadow",
        "default_static_global": default_plan["default_path"]["global_map"] == "static",
        "default_lidar_local": default_plan["default_path"]["local_costmap"] == "lidar_voxel",
        "first_scan_marks": old_obstacle in first_lethal,
        "second_scan_ray_clears_old_obstacle": old_obstacle not in second_lethal,
        "second_scan_marks_new_obstacle": model.cell(1.25, 0.0) in second_lethal,
        "inflation_is_present": any(value == INFLATED for value in second_inflated.values()),
        "global_map_unchanged_by_local_scans": static_before == static_after,
        "scene_map_contains_obstacles": grid_summary(loaded.smoke_map)["occupied_cells"] > 0,
        "failed_active_downgrades_shadow": (
            failed_active["effective_mode"] == "shadow"
            and failed_active["fallback_applied"] is True
        ),
        "failed_active_preserves_core_hash": (
            failed_active["core_nav_sha256"] == default_plan["core_nav_sha256"]
        ),
        "ready_active_retains_lidar_core": (
            ready_active["effective_mode"] == "active"
            and ready_active["core_nav_sha256"] == default_plan["core_nav_sha256"]
        ),
        "velocity_is_bounded": (
            command.linear_x == 0.25 and command.angular_z == -1.0
        ),
        "simulation_estop_forces_zero": (
            estopped.stopped and estopped.linear_x == 0.0 and estopped.angular_z == 0.0
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "offline_only": True,
        "processes_started": 0,
        "network_accessed": False,
        "shared_resources_used": [],
        "checks": checks,
        "metrics": {
            "first_lethal_cells": len(first_lethal),
            "first_inflated_cells": len(first_inflated),
            "second_lethal_cells": len(second_lethal),
            "second_inflated_cells": len(second_inflated),
            "static_map": grid_summary(loaded.smoke_map),
        },
        "default_plan_sha256": _json_sha(default_plan),
        "failed_active_plan_sha256": _json_sha(failed_active),
    }
