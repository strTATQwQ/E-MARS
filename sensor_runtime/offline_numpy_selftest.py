#!/usr/bin/env python3
"""Bundled-NumPy offline performance/strictness gate without pytest."""

from __future__ import annotations

import json
import statistics
import struct
import time

import numpy as np

from .contract import REQUIRED_STREAMS
from .core import SensorBatch
from .image_contract import require_rgb_content, rgb_content_evidence
from .lidar_contract import frozen_lidar_local_directions, raycast_frozen_lidar
from .pointcloud import (
    depth_points_from_calibration,
    filter_lidar_xyz,
    finite_xyz_rows,
    frozen_depth_rejection_masks,
    nearest_valid_depth_tiles,
    packed_xyz,
)
from .wire import MAGIC, decode_batch, encode_batch


def _links() -> list[tuple[str, np.ndarray]]:
    values = [("base", np.zeros(3, dtype=np.float32))]
    for leg_index, leg in enumerate(("FL", "FR", "RL", "RR")):
        for part_index, part in enumerate(("thigh", "calf", "foot")):
            values.append((f"{leg}_{part}", np.asarray([2.0 + leg_index, 2.0 + part_index, 2.0], dtype=np.float32)))
    return values


def _batch() -> SensorBatch:
    rng = np.random.default_rng(17)
    stamp, render = 123_000_000, 4
    payloads = {
        "d435i_rgb": {
            "rgb8": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
            "front_rgb8": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
            "render_id": render, "render_generation": 0,
        },
        "d435i_depth": {
            "depth_m": rng.uniform(0.28, 6.0, (480, 640)).astype(np.float32),
            "render_id": render, "render_generation": 0,
        },
        "lidar": {"points_lidar": rng.uniform(-2, 2, (8, 180, 3)).astype(np.float32)},
        "pose": {"position": np.asarray([0.0, 0.0, 0.42]), "rotation_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0])},
        "tf": {"fixed": [], "base_translation": [0.0, 0.0, 0.42]},
    }
    return SensorBatch(
        stamp, 0, 0, payloads, {name: stamp for name in REQUIRED_STREAMS},
        {
            "identity": "sensor-soak:0:0", "linear_x": 0.0, "angular_z": 0.0,
            "emergency_stop": True, "applied_step_count": 10,
            "steps_since_previous_capture": 10, "physics_step": 10,
            "render_id": render, "render_generation": 0,
            "reset_kind": "continuous_world_articulation_state",
        },
        "initial",
    )


def main() -> int:
    metrics: dict[str, object] = {}
    errors: list[str] = []
    try:
        rng = np.random.default_rng(42)
        constant = np.zeros((12, 16, 3), dtype=np.uint8)
        if rgb_content_evidence(constant)["valid"] is not False:
            raise AssertionError("constant RGB placeholder was accepted")
        try:
            require_rgb_content(constant, rgb_content_evidence(constant), "black fixture")
        except ValueError:
            pass
        else:
            raise AssertionError("black RGB fixture did not fail closed")
        depth = rng.uniform(0.28, 6.0, (480, 640)).astype(np.float32)
        links = _links()
        centers = np.stack([center for _name, center in links])
        lidar = rng.uniform(-2, 2, (8, 180, 3)).astype(np.float32)
        lidar[..., 0] += 3.0
        directions = frozen_lidar_local_directions()
        distance_workspace = np.empty(8 * 180, dtype=np.float32)
        no_hit = {"hit": False, "distance": float("inf")}

        def mock_raycast(_origin, _direction, _maximum):
            return no_hit

        for _ in range(3):
            raycast_frozen_lidar(
                directions, np.eye(3), np.zeros(3), mock_raycast, distance_workspace
            )
        lidar_compute_times = []
        for _ in range(20):
            started = time.perf_counter()
            mock_points = raycast_frozen_lidar(
                directions, np.eye(3), np.zeros(3), mock_raycast, distance_workspace
            )
            lidar_compute_times.append(time.perf_counter() - started)
            if mock_points.shape != (8, 180, 3):
                raise AssertionError("frozen LiDAR mock changed 8x180 geometry")

        def project_filter() -> tuple[np.ndarray, dict[str, np.ndarray]]:
            points = depth_points_from_calibration(
                depth, fx=400.0, fy=410.0, cx=319.5, cy=239.5,
                pitch_down_deg=20.0, translation_from_base_m=[0.2, 0.0, 0.2],
            )
            masks = frozen_depth_rejection_masks(
                depth, points, [0.0, 0.0, 0.0], links, points[..., 2] + 0.42, 0.0
            )
            if masks["rejected"].shape != depth.shape:
                raise AssertionError("depth rejection mask shape mismatch")
            transport = nearest_valid_depth_tiles(
                points,
                depth,
                masks["rejected"],
            )
            if transport.shape != (120, 160, 3) or transport.nbytes != 230400:
                raise AssertionError("bounded depth transport shape/size mismatch")
            return transport, masks

        for _ in range(3):
            project_filter()
        projection_times = []
        bridge_times = []
        for _ in range(20):
            started = time.perf_counter()
            points, _masks = project_filter()
            projection_times.append(time.perf_counter() - started)

            started = time.perf_counter()
            _sensor, base, counts = filter_lidar_xyz(lidar, centers)
            safety = finite_xyz_rows(points, base)
            _array, width, height, data, _dense = packed_xyz(safety)
            if counts["output"] <= 0 or len(data) != width * height * 12:
                raise AssertionError("bridge vector output contract mismatch")
            bridge_times.append(time.perf_counter() - started)

        batch = _batch()
        wire_times = []
        for _ in range(15):
            started = time.perf_counter()
            restored = decode_batch(encode_batch(batch))
            if restored.stamp_ns != batch.stamp_ns:
                raise AssertionError("wire timestamp roundtrip mismatch")
            wire_times.append(time.perf_counter() - started)

        metrics.update(
            {
                "depth_projection_filter_median_sec": statistics.median(projection_times),
                "depth_projection_filter_max_sec": max(projection_times),
                "bridge_vector_median_sec": statistics.median(bridge_times),
                "bridge_vector_max_sec": max(bridge_times),
                "wire_roundtrip_median_sec": statistics.median(wire_times),
                "wire_roundtrip_max_sec": max(wire_times),
                "wire_packet_bytes": len(encode_batch(batch)),
                "sample_count": 20,
                "lidar_python_mock_median_sec": statistics.median(lidar_compute_times),
                "lidar_python_mock_max_sec": max(lidar_compute_times),
            }
        )
        limits = {
            "depth_projection_filter_median_sec": 0.075,
            "depth_projection_filter_max_sec": 0.10,
            "bridge_vector_median_sec": 0.10,
            "bridge_vector_max_sec": 0.20,
            "wire_roundtrip_median_sec": 0.05,
            "wire_roundtrip_max_sec": 0.10,
            "lidar_python_mock_median_sec": 0.01,
            "lidar_python_mock_max_sec": 0.02,
        }
        metrics["limits_sec"] = limits
        for name, limit in limits.items():
            if float(metrics[name]) >= limit:
                errors.append(f"{name}={metrics[name]} is not below {limit}")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        metrics["status"] = "PASS" if not errors else "FAIL"
        metrics["errors"] = errors
        print(json.dumps(metrics, sort_keys=True, allow_nan=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
