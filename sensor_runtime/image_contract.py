"""Frozen image-content evidence for headless diagnostic rendering."""

from __future__ import annotations

from typing import Any

import numpy as np


def downsample_rgb_2x2(value: Any) -> np.ndarray:
    """Area-average 2x2 pixels so half-scale intrinsics keep cx/cy=79.5/59.5."""

    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3 or array.shape[0] % 2 or array.shape[1] % 2:
        raise ValueError("2x2 RGB downsample requires an even HxWx3 array")
    if not np.isfinite(array).all():
        raise ValueError("RGB downsample input contains non-finite values")
    height, width, _ = array.shape
    averaged = array.astype(np.float32, copy=False).reshape(
        height // 2, 2, width // 2, 2, 3
    ).mean(axis=(1, 3))
    return np.ascontiguousarray(np.rint(averaged).clip(0, 255), dtype=np.uint8)


def rgb_content_evidence(value: Any) -> dict[str, int | bool]:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
        raise ValueError("RGB content must be a non-empty HxWx3 array")
    if not np.isfinite(array).all():
        raise ValueError("RGB content contains non-finite values")
    minimum = int(np.min(array))
    maximum = int(np.max(array))
    nonzero = int(np.count_nonzero(array))
    valid = nonzero > 0 and minimum < maximum
    return {
        "finite": True,
        "minimum": minimum,
        "maximum": maximum,
        "dynamic_range": maximum - minimum,
        "nonzero_count": nonzero,
        "sample_count": int(array.size),
        "valid": valid,
    }


def require_rgb_content(value: Any, claimed: Any, name: str) -> dict[str, int | bool]:
    evidence = rgb_content_evidence(value)
    if evidence["valid"] is not True:
        raise ValueError(f"{name} is black or constant")
    if claimed != evidence:
        raise ValueError(f"{name} content evidence does not match pixels")
    return evidence
