"""ROS-independent vectorized XYZ point-cloud validation and filtering."""

from __future__ import annotations

from functools import lru_cache
import math
import threading
from typing import Any, Iterable

import numpy as np

from .contract import (
    DEPTH_CLOUD_HEIGHT,
    DEPTH_CLOUD_TILE_STRIDE,
    DEPTH_CLOUD_WIDTH,
    MIN_VALID_DEPTH_POINTS,
    MIN_VALID_DEPTH_RATIO,
)


XYZ_FIELDS = (("x", 0, 7, 1), ("y", 4, 7, 1), ("z", 8, 7, 1))


def valid_depth_evidence(depth_m: Any) -> tuple[int, float, bool]:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape != (480, 640):
        raise ValueError("valid-depth evidence requires 640x480 depth")
    count = int(np.count_nonzero(np.isfinite(depth) & (depth >= 0.28) & (depth <= 6.0)))
    ratio = count / float(depth.size)
    return count, ratio, count >= MIN_VALID_DEPTH_POINTS and ratio >= MIN_VALID_DEPTH_RATIO


def xyz_array_from_message(message: Any) -> np.ndarray:
    """Return an HxWx3 float32 view after strict PointCloud2 layout checks."""

    fields = tuple(
        (str(item.name), int(item.offset), int(item.datatype), int(item.count))
        for item in message.fields
    )
    width, height = int(message.width), int(message.height)
    if fields != XYZ_FIELDS:
        raise ValueError("PointCloud2 must contain packed float32 x/y/z fields")
    if bool(message.is_bigendian) or int(message.point_step) != 12:
        raise ValueError("PointCloud2 must be little-endian packed xyz float32")
    if width <= 0 or height <= 0 or int(message.row_step) != width * 12:
        raise ValueError("PointCloud2 dimensions/row_step are invalid")
    view = memoryview(message.data)
    if view.nbytes != int(message.row_step) * height:
        raise ValueError("PointCloud2 data length differs from exact layout")
    return np.frombuffer(view, dtype="<f4").reshape(height, width, 3)


def packed_xyz(points: Any) -> tuple[np.ndarray, int, int, bytes, bool]:
    """Normalize HxWx3 or Nx3 points and return zero-copy-ready cloud fields."""

    value = np.asarray(points, dtype=np.float32)
    if value.ndim == 2 and value.shape[1] == 3:
        height, width = 1, int(value.shape[0])
    elif value.ndim == 3 and value.shape[2] == 3:
        height, width = int(value.shape[0]), int(value.shape[1])
    else:
        raise ValueError(f"XYZ array has invalid shape {value.shape}")
    if width <= 0 or height <= 0:
        raise ValueError("XYZ cloud cannot be empty")
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    return contiguous, width, height, contiguous.tobytes(order="C"), bool(np.isfinite(contiguous).all())


def filter_lidar_xyz(
    points_lidar: Any,
    centers_base: Any,
    *,
    minimum_range: float = 0.10,
    maximum_range: float = 12.0,
    minimum_height: float = -0.55,
    maximum_height: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Vectorized organized LiDAR range/height/body/link self filtering."""

    points = np.asarray(points_lidar, dtype=np.float32)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("organized LiDAR points must be HxWx3")
    centers = np.asarray(centers_base, dtype=np.float32)
    if centers.shape != (13, 3) or not np.isfinite(centers).all():
        raise ValueError("LiDAR filter requires 13 finite base-frame centers")
    finite = np.isfinite(points).all(axis=2)
    distance = np.linalg.norm(points, axis=2)
    in_range = (distance >= minimum_range) & (distance <= maximum_range)
    base = points + np.asarray([0.25, 0.0, 0.18], dtype=np.float32)
    in_height = (base[:, :, 2] >= minimum_height) & (base[:, :, 2] <= maximum_height)
    body = (
        (np.abs(base[:, :, 0]) <= 0.36)
        & (np.abs(base[:, :, 1]) <= 0.20)
        & (np.abs(base[:, :, 2]) <= 0.16)
    )
    delta = base[:, :, None, :] - centers[None, None, :, :]
    dynamic = np.sum(delta * delta, axis=3).min(axis=2) <= 0.12**2
    range_bad = finite & ~in_range
    height_bad = finite & in_range & ~in_height
    self_bad = finite & in_range & in_height & (body | dynamic)
    accepted = finite & in_range & in_height & ~(body | dynamic)
    sensor_output = np.full(points.shape, np.nan, dtype=np.float32)
    base_output = np.full(points.shape, np.nan, dtype=np.float32)
    sensor_output[accepted] = points[accepted]
    base_output[accepted] = base[accepted]
    return sensor_output, base_output, {
        "input": int(points.shape[0] * points.shape[1]),
        "finite_input": int(np.count_nonzero(finite)),
        "output": int(np.count_nonzero(accepted)),
        "self": int(np.count_nonzero(self_bad)),
        "range": int(np.count_nonzero(range_bad)),
        "height": int(np.count_nonzero(height_bad)),
    }


def finite_xyz_rows(*clouds: Any) -> np.ndarray:
    rows = [np.asarray(value, dtype=np.float32).reshape(-1, 3) for value in clouds]
    finite = [value[np.isfinite(value).all(axis=1)] for value in rows]
    combined = np.concatenate(finite, axis=0)
    if combined.shape[0] == 0:
        raise ValueError("safety cloud cannot be empty")
    return np.ascontiguousarray(combined, dtype=np.float32)


@lru_cache(maxsize=4)
def _depth_ray_grid(
    height: int,
    width: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    pitch_down_deg: float,
) -> np.ndarray:
    columns = (np.arange(width, dtype=np.float32) - np.float32(cx)) / np.float32(fx)
    rows = (np.arange(height, dtype=np.float32) - np.float32(cy)) / np.float32(fy)
    pitch = np.float32(math.radians(pitch_down_deg))
    sine, cosine = np.float32(math.sin(float(pitch))), np.float32(math.cos(float(pitch)))
    rays = np.empty((height, width, 3), dtype=np.float32)
    rays[..., 0] = -sine * rows[:, None] + cosine
    rays[..., 1] = -columns[None, :]
    rays[..., 2] = -cosine * rows[:, None] - sine
    rays.flags.writeable = False
    return rays


def depth_points_from_calibration(
    depth_m: Any,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    pitch_down_deg: float,
    translation_from_base_m: Any,
) -> np.ndarray:
    """Project depth with a bounded cached fixed-calibration ray grid."""

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2 or depth.shape[0] <= 0 or depth.shape[1] <= 0:
        raise ValueError("depth image must be a non-empty matrix")
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in (fx, fy)):
        raise ValueError("depth focal lengths must be finite positive values")
    translation = np.asarray(translation_from_base_m, dtype=np.float32)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("depth translation must be finite xyz")
    rays = _depth_ray_grid(
        int(depth.shape[0]), int(depth.shape[1]), float(fx), float(fy),
        float(cx), float(cy), float(pitch_down_deg),
    )
    points = rays * depth[..., None]
    points += translation
    return points


def nearest_valid_depth_tiles(
    points_base: Any,
    depth_m: Any,
    rejected: Any,
    *,
    stride: int = DEPTH_CLOUD_TILE_STRIDE,
) -> np.ndarray:
    """Retain the nearest accepted point in each fixed angular image tile.

    Filtering remains full-resolution. Reduction happens afterward and is
    collision-conservative within every tile: a farther return can never
    replace a nearer accepted return.
    """

    points = np.asarray(points_base, dtype=np.float32)
    depth = np.asarray(depth_m, dtype=np.float32)
    rejected_mask = np.asarray(rejected, dtype=bool)
    if (
        points.shape != (480, 640, 3)
        or depth.shape != (480, 640)
        or rejected_mask.shape != depth.shape
        or isinstance(stride, bool)
        or stride != DEPTH_CLOUD_TILE_STRIDE
        or depth.shape[0] % stride
        or depth.shape[1] % stride
    ):
        raise ValueError("depth cloud reduction differs from the frozen 4x4 contract")
    valid = (
        ~rejected_mask
        & np.isfinite(depth)
        & np.isfinite(points).all(axis=2)
    )
    tile_depth = np.where(valid, depth, np.float32(np.inf))
    tile_depth = tile_depth.reshape(
        DEPTH_CLOUD_HEIGHT, stride, DEPTH_CLOUD_WIDTH, stride
    ).transpose(0, 2, 1, 3).reshape(DEPTH_CLOUD_HEIGHT, DEPTH_CLOUD_WIDTH, stride * stride)
    nearest_index = np.argmin(tile_depth, axis=2)
    nearest_depth = np.take_along_axis(
        tile_depth, nearest_index[..., None], axis=2
    )[..., 0]
    selected_rows = (
        np.arange(DEPTH_CLOUD_HEIGHT, dtype=np.intp)[:, None] * stride
        + nearest_index // stride
    )
    selected_columns = (
        np.arange(DEPTH_CLOUD_WIDTH, dtype=np.intp)[None, :] * stride
        + nearest_index % stride
    )
    selected = points[selected_rows, selected_columns]
    output = np.full(
        (DEPTH_CLOUD_HEIGHT, DEPTH_CLOUD_WIDTH, 3),
        np.nan,
        dtype=np.float32,
    )
    has_valid = np.isfinite(nearest_depth)
    output[has_valid] = selected[has_valid]
    return np.ascontiguousarray(output)


class DepthFilterWorkspace:
    """Bounded reusable scratch for the fixed 640x480 dynamic-link filter."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = tuple(int(value) for value in shape)
        if len(self.shape) != 2 or min(self.shape) <= 0:
            raise ValueError("depth filter workspace shape is invalid")
        self.distance = np.empty(self.shape, dtype=np.float32)
        self.temporary = np.empty(self.shape, dtype=np.float32)
        self.dynamic = np.empty(self.shape, dtype=bool)
        self.boolean = np.empty(self.shape, dtype=bool)
        self._lock = threading.Lock()

    def dynamic_mask(
        self, points: np.ndarray, dynamic_values: list[tuple[str, np.ndarray]]
    ) -> np.ndarray:
        if points.shape != self.shape + (3,):
            raise ValueError("depth workspace shape differs from projected points")
        with self._lock:
            self.dynamic.fill(False)
            x, y, z = points[..., 0], points[..., 1], points[..., 2]
            for name, center in dynamic_values:
                radius = np.float32(
                    0.08 if name.endswith("_foot") else 0.09 if name.endswith("_calf") else 0.10
                )
                np.subtract(x, center[0], out=self.distance)
                np.square(self.distance, out=self.distance)
                np.subtract(y, center[1], out=self.temporary)
                np.square(self.temporary, out=self.temporary)
                np.add(self.distance, self.temporary, out=self.distance)
                np.subtract(z, center[2], out=self.temporary)
                np.square(self.temporary, out=self.temporary)
                np.add(self.distance, self.temporary, out=self.distance)
                np.less_equal(self.distance, radius * radius, out=self.boolean)
                np.logical_or(self.dynamic, self.boolean, out=self.dynamic)
            # A bool copy is small and prevents a subsequent callback from
            # mutating the returned evidence mask after the lock is released.
            return self.dynamic.copy()


@lru_cache(maxsize=2)
def _depth_filter_workspace(shape: tuple[int, int]) -> DepthFilterWorkspace:
    return DepthFilterWorkspace(shape)


def ground_rejected(world_z: Any, support_plane_world_z: float, margin_m: float = 0.08) -> np.ndarray:
    if not np.isfinite(support_plane_world_z):
        raise ValueError("support plane must be finite")
    return np.asarray(world_z, dtype=np.float32) <= float(support_plane_world_z) + margin_m


def frozen_depth_rejection_masks(
    depth_m: Any,
    points_base: Any,
    base_center: Any,
    links: Iterable[tuple[str, Any]],
    world_z: Any,
    support_plane_world_z: float,
    workspace: DepthFilterWorkspace | None = None,
) -> dict[str, np.ndarray]:
    """Numerical frozen D435i range/body/link/ground/forward contract."""

    depth = np.asarray(depth_m, dtype=np.float32)
    points = np.asarray(points_base, dtype=np.float32)
    if points.shape != depth.shape + (3,):
        raise ValueError("depth points shape differs from depth image")
    link_values = [(str(name), np.asarray(center, dtype=np.float32)) for name, center in links]
    expected = {"base"} | {
        f"{leg}_{part}"
        for leg in ("FL", "FR", "RL", "RR")
        for part in ("thigh", "calf", "foot")
    }
    if len(link_values) != 13 or {name for name, _center in link_values} != expected:
        raise ValueError("depth filter requires exactly 13 frozen centers")
    if any(center.shape != (3,) or not np.isfinite(center).all() for _name, center in link_values):
        raise ValueError("depth filter centers must be finite xyz")
    base = np.asarray(base_center, dtype=np.float32)
    if base.shape != (3,) or not np.isfinite(base).all():
        raise ValueError("depth filter base center is invalid")
    relative = points - base
    range_mask = (~np.isfinite(depth)) | (depth < 0.28) | (depth > 6.0)
    body_mask = (
        (np.abs(relative[..., 0]) <= 0.34)
        & (np.abs(relative[..., 1]) <= 0.18)
        & (np.abs(relative[..., 2]) <= 0.14)
    )
    dynamic_values = [(name, center) for name, center in link_values if name != "base"]
    selected_workspace = workspace or _depth_filter_workspace(tuple(depth.shape))
    dynamic_mask = selected_workspace.dynamic_mask(points, dynamic_values)
    ground_mask = ground_rejected(world_z, support_plane_world_z)
    forward_mask = points[..., 0] <= 0.12
    rejected = range_mask | body_mask | dynamic_mask | ground_mask | forward_mask
    return {
        "range": range_mask,
        "body": body_mask,
        "dynamic": dynamic_mask,
        "ground": ground_mask,
        "forward": forward_mask,
        "rejected": rejected,
    }
