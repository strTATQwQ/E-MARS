"""Precomputed frozen 8x180 LiDAR ray grid and low-allocation query loop."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np


def frozen_lidar_local_directions() -> np.ndarray:
    elevations = np.radians(
        np.asarray([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0], dtype=np.float64)
    )[:, None]
    azimuths = (2.0 * math.pi * np.arange(180, dtype=np.float64) / 180.0)[None, :]
    horizontal = np.cos(elevations)
    directions = np.empty((8, 180, 3), dtype=np.float64)
    directions[..., 0] = horizontal * np.cos(azimuths)
    directions[..., 1] = horizontal * np.sin(azimuths)
    directions[..., 2] = np.sin(elevations)
    return np.ascontiguousarray(directions.reshape(1440, 3))


def raycast_frozen_lidar(
    local_directions: np.ndarray,
    rotation_base_to_world: np.ndarray,
    origin_world: np.ndarray,
    raycast_closest: Callable[[tuple[float, float, float], tuple[float, float, float], float], Any],
    distance_workspace: np.ndarray,
) -> np.ndarray:
    local = np.asarray(local_directions, dtype=np.float64)
    rotation = np.asarray(rotation_base_to_world, dtype=np.float64)
    origin = np.asarray(origin_world, dtype=np.float64)
    distances = np.asarray(distance_workspace, dtype=np.float32)
    if local.shape != (1440, 3) or rotation.shape != (3, 3) or origin.shape != (3,):
        raise ValueError("frozen LiDAR grid/pose shape mismatch")
    if distances.shape != (1440,) or not distances.flags.writeable:
        raise ValueError("frozen LiDAR distance workspace must be writable length 1440")
    world_directions = local @ rotation.T
    origin_tuple = (float(origin[0]), float(origin[1]), float(origin[2]))
    distances.fill(np.nan)
    for index, direction in enumerate(world_directions):
        hit = raycast_closest(
            origin_tuple,
            (float(direction[0]), float(direction[1]), float(direction[2])),
            12.0,
        )
        distance = float(hit.get("distance", math.inf))
        if bool(hit.get("hit", False)) and 0.10 <= distance <= 12.0:
            distances[index] = distance
    points = local.astype(np.float32, copy=False) * distances[:, None]
    return np.ascontiguousarray(points.reshape(8, 180, 3))
