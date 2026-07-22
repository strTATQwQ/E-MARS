from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import pytest

np = pytest.importorskip("numpy")

from sensor_runtime.pointcloud import (
    filter_lidar_xyz,
    finite_xyz_rows,
    depth_points_from_calibration,
    frozen_depth_rejection_masks,
    ground_rejected,
    nearest_valid_depth_tiles,
    packed_xyz,
    valid_depth_evidence,
    xyz_array_from_message,
)


@dataclass
class Field:
    name: str
    offset: int
    datatype: int = 7
    count: int = 1


class Message:
    def __init__(self, points: np.ndarray) -> None:
        value = np.ascontiguousarray(points, dtype=np.float32)
        self.height, self.width = value.shape[:2]
        self.fields = [Field("x", 0), Field("y", 4), Field("z", 8)]
        self.is_bigendian = False
        self.point_step = 12
        self.row_step = self.width * 12
        self.data = value.tobytes()


def _centers() -> np.ndarray:
    centers = np.zeros((13, 3), dtype=np.float32)
    centers[1:, 0] = np.linspace(-0.4, 0.4, 12)
    centers[1:, 1] = 0.3
    centers[1:, 2] = -0.2
    return centers


def test_vectorized_lidar_preserves_organized_nan_shape_and_counts() -> None:
    lidar = np.zeros((8, 180, 3), dtype=np.float32)
    lidar[:, :, 0] = 2.0
    lidar[0, 0] = np.nan
    lidar[0, 1] = [0.01, 0.0, 0.0]
    sensor, base, counts = filter_lidar_xyz(lidar, _centers())
    assert sensor.shape == base.shape == (8, 180, 3)
    assert np.isnan(sensor[0, 0]).all() and np.isnan(sensor[0, 1]).all()
    assert counts["finite_input"] == 1439
    assert counts["range"] == 1
    assert counts["output"] > 0
    assert counts["output"] + counts["range"] + counts["height"] + counts["self"] <= counts["finite_input"]


def test_strict_pointcloud_layout_rejects_padding_and_truncation() -> None:
    message = Message(np.ones((2, 3, 3), dtype=np.float32))
    assert xyz_array_from_message(message).shape == (2, 3, 3)
    message.row_step += 4
    with pytest.raises(ValueError, match="row_step"):
        xyz_array_from_message(message)
    message = Message(np.ones((2, 3, 3), dtype=np.float32))
    message.data = message.data[:-1]
    with pytest.raises(ValueError, match="data length"):
        xyz_array_from_message(message)
    message = Message(np.ones((2, 3, 3), dtype=np.float32))
    message.fields[0], message.fields[1] = message.fields[1], message.fields[0]
    with pytest.raises(ValueError, match="packed float32"):
        xyz_array_from_message(message)


def test_vectorized_307200_depth_plus_1440_lidar_is_bounded() -> None:
    depth = np.empty((480, 640, 3), dtype=np.float32)
    depth[:, :, 0], depth[:, :, 1], depth[:, :, 2] = 2.0, 0.0, 0.2
    depth_message = Message(depth)
    lidar = np.empty((8, 180, 3), dtype=np.float32)
    lidar[:, :, 0], lidar[:, :, 1], lidar[:, :, 2] = 2.0, 0.0, 0.0
    elapsed: list[float] = []
    for _ in range(7):
        started = time.perf_counter()
        depth_view = xyz_array_from_message(depth_message)
        _sensor, base, counts = filter_lidar_xyz(lidar, _centers())
        safety = finite_xyz_rows(depth_view, base)
        _array, width, height, data, _dense = packed_xyz(safety)
        assert counts["output"] > 0 and width * height == 307200 + counts["output"]
        assert len(data) == width * height * 12
        elapsed.append(time.perf_counter() - started)
    # Use both a stable median and a generous hard ceiling to avoid hiding a
    # single catastrophic Python per-point regression.
    assert statistics.median(elapsed) < 0.10, elapsed
    assert max(elapsed) < 0.20, elapsed


def test_support_plane_margin_keeps_low_real_obstacles() -> None:
    rejected = ground_rejected(np.asarray([0.08, 0.09, 0.15]), 0.0)
    assert rejected.tolist() == [True, False, False]


def test_empty_safety_cloud_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        finite_xyz_rows(np.full((2, 3), np.nan, dtype=np.float32))


def _links() -> list[tuple[str, np.ndarray]]:
    values = [("base", np.zeros(3, dtype=np.float32))]
    for leg_index, leg in enumerate(("FL", "FR", "RL", "RR")):
        for part_index, part in enumerate(("thigh", "calf", "foot")):
            values.append((f"{leg}_{part}", np.asarray([2.0 + leg_index, 2.0 + part_index, 2.0], dtype=np.float32)))
    return values


def _masks(depth: np.ndarray, points: np.ndarray, links=None, world_z=None):
    if world_z is None:
        world_z = np.full(depth.shape, 1.0, dtype=np.float32)
    return frozen_depth_rejection_masks(
        depth,
        points,
        np.zeros(3, dtype=np.float32),
        _links() if links is None else links,
        world_z,
        0.0,
    )


def test_frozen_depth_filter_numeric_boundaries() -> None:
    depth = np.asarray([[0.279, 0.28, 6.0, 6.001]], dtype=np.float32)
    points = np.asarray([[[1.0, 1.0, 1.0]] * 4], dtype=np.float32)
    assert _masks(depth, points)["range"].tolist() == [[True, False, False, True]]

    depth = np.ones((1, 2), dtype=np.float32)
    points = np.asarray([[[0.12, 1.0, 1.0], [0.12001, 1.0, 1.0]]], dtype=np.float32)
    assert _masks(depth, points)["forward"].tolist() == [[True, False]]

    points = np.asarray([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    assert _masks(depth, points, world_z=np.asarray([[0.08, 0.08001]]))["ground"].tolist() == [[True, False]]

    body_points = np.asarray(
        [[[0.34, 0.0, 0.0], [0.0, 0.18, 0.0], [0.0, 0.0, 0.14], [0.34001, 0.18001, 0.14001]]],
        dtype=np.float32,
    )
    assert _masks(np.ones((1, 4), dtype=np.float32), body_points)["body"].tolist() == [[True, True, True, False]]

    links = _links()
    cases = []
    for name, radius in (("FL_thigh", 0.10), ("FL_calf", 0.09), ("FL_foot", 0.08)):
        center = dict(links)[name]
        cases.extend([center + [radius, 0.0, 0.0], center + [radius + 0.0001, 0.0, 0.0]])
    dynamic = _masks(np.ones((1, 6), dtype=np.float32), np.asarray([cases], dtype=np.float32), links=links)["dynamic"]
    assert dynamic.tolist() == [[True, False, True, False, True, False]]

    with pytest.raises(ValueError, match="exactly 13"):
        _masks(np.ones((1, 1), dtype=np.float32), np.ones((1, 1, 3), dtype=np.float32), links=links[:-1])


def test_cached_full_resolution_depth_projection_and_13_center_filter_are_bounded() -> None:
    rng = np.random.default_rng(42)
    depth = rng.uniform(0.28, 6.0, (480, 640)).astype(np.float32)
    links = _links()
    # Warm both the calibration and bounded filter workspaces.
    for _ in range(3):
        warmed = depth_points_from_calibration(
            depth, fx=400.0, fy=410.0, cx=319.5, cy=239.5,
            pitch_down_deg=20.0, translation_from_base_m=[0.2, 0.0, 0.2],
        )
        frozen_depth_rejection_masks(
            depth, warmed, [0.0, 0.0, 0.0], links, warmed[..., 2] + np.float32(0.42), 0.0
        )
    elapsed = []
    for _ in range(20):
        started = time.perf_counter()
        points = depth_points_from_calibration(
            depth, fx=400.0, fy=410.0, cx=319.5, cy=239.5,
            pitch_down_deg=20.0, translation_from_base_m=[0.2, 0.0, 0.2],
        )
        world_z = points[..., 2] + np.float32(0.42)
        masks = frozen_depth_rejection_masks(depth, points, [0.0, 0.0, 0.0], links, world_z, 0.0)
        assert masks["rejected"].shape == depth.shape
        elapsed.append(time.perf_counter() - started)
    assert statistics.median(elapsed) < 0.075, elapsed
    assert max(elapsed) < 0.10, elapsed


def test_depth_transport_keeps_nearest_valid_point_in_every_4x4_tile() -> None:
    depth = np.full((480, 640), 4.0, dtype=np.float32)
    points = np.zeros((480, 640, 3), dtype=np.float32)
    points[..., 0] = depth
    rejected = np.zeros(depth.shape, dtype=bool)
    # Put the nearest return away from every fixed-stride sample location.
    depth[2::4, 3::4] = 0.5
    points[2::4, 3::4, 0] = 0.5
    reduced = nearest_valid_depth_tiles(points, depth, rejected)
    assert reduced.shape == (120, 160, 3)
    assert reduced.nbytes == 120 * 160 * 3 * 4
    assert np.all(reduced[..., 0] == np.float32(0.5))


def test_depth_transport_never_uses_rejected_nearest_return() -> None:
    depth = np.full((480, 640), 4.0, dtype=np.float32)
    points = np.zeros((480, 640, 3), dtype=np.float32)
    points[..., 0] = depth
    rejected = np.zeros(depth.shape, dtype=bool)
    depth[2::4, 3::4] = 0.5
    points[2::4, 3::4, 0] = 0.5
    rejected[2::4, 3::4] = True
    depth[1::4, 2::4] = 0.8
    points[1::4, 2::4, 0] = 0.8
    rejected[:4, :4] = True
    reduced = nearest_valid_depth_tiles(points, depth, rejected)
    assert np.isnan(reduced[0, 0]).all()
    assert np.all(reduced[1:, :, 0] == np.float32(0.8))
    assert np.all(reduced[0, 1:, 0] == np.float32(0.8))
    with pytest.raises(ValueError, match="frozen 4x4"):
        nearest_valid_depth_tiles(points, depth, rejected, stride=8)


def test_valid_depth_content_rejects_uniform_placeholder_ratio_and_accepts_10_percent_boundary() -> None:
    placeholder = np.full((480, 640), 256.0, dtype=np.float32)
    placeholder.reshape(-1)[:7066] = 1.0  # about the known 2.3% uniform placeholder rate
    count, ratio, ready = valid_depth_evidence(placeholder)
    assert count == 7066 and ratio == pytest.approx(7066 / 307200) and ready is False
    boundary = np.full((480, 640), 256.0, dtype=np.float32)
    boundary.reshape(-1)[:30720] = 0.28
    count, ratio, ready = valid_depth_evidence(boundary)
    assert count == 30720 and ratio == pytest.approx(0.10) and ready is True
