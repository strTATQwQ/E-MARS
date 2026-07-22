"""ROS-independent CameraInfo matrix construction and strict comparison."""

from __future__ import annotations

import math
from typing import Any, Mapping


def camera_matrices(width: int, height: int, fx: float, fy: float) -> dict[str, list[float]]:
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    return {
        "d": [0.0] * 5,
        "k": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    }


def matrices_from_fov(
    width: int, height: int, hfov_deg: float, vfov_deg: float
) -> dict[str, list[float]]:
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    return camera_matrices(width, height, fx, fy)


def require_camera_matrices(
    observed: Mapping[str, Any], expected: Mapping[str, list[float]], *, tolerance: float = 1e-5
) -> None:
    for name in ("d", "k", "r", "p"):
        actual = observed.get(name)
        wanted = expected[name]
        if (
            not isinstance(actual, (list, tuple))
            or len(actual) != len(wanted)
            or not all(
                math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
                for a, b in zip(actual, wanted)
            )
        ):
            raise ValueError(f"CameraInfo {name.upper()} matrix violates frozen calibration")
