from __future__ import annotations

import base64
import hashlib
import math
import statistics
import zlib
from array import array
from typing import Any, Iterable


def camera_intrinsics_from_physical(
    width: int,
    height: int,
    *,
    focal_length_mm: float = 24.0,
    horizontal_aperture_mm: float = 20.955,
) -> dict[str, Any]:
    width_i, height_i = int(width), int(height)
    if width_i <= 0 or height_i <= 0:
        raise ValueError("camera dimensions must be positive")
    focal = float(focal_length_mm)
    aperture = float(horizontal_aperture_mm)
    if focal <= 0.0 or aperture <= 0.0:
        raise ValueError("camera focal length and aperture must be positive")
    fx = width_i * focal / aperture
    fy = fx
    cx = (width_i - 1.0) * 0.5
    cy = (height_i - 1.0) * 0.5
    return {
        "width": width_i,
        "height": height_i,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "horizontal_fov_deg": math.degrees(2.0 * math.atan(aperture / (2.0 * focal))),
        "k": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "p": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    }


def ppm_bytes(rgb: bytes, width: int, height: int) -> bytes:
    expected = int(width) * int(height) * 3
    if len(rgb) != expected:
        raise ValueError(f"rgb byte length {len(rgb)} != expected {expected}")
    return f"P6\n{int(width)} {int(height)}\n255\n".encode("ascii") + rgb


def decode_mask_zlib(payload: dict[str, Any], *, width: int, height: int) -> bytes:
    encoded = str(payload.get("mask_zlib_base64") or "")
    if not encoded:
        return bytes(int(width) * int(height))
    raw = zlib.decompress(base64.b64decode(encoded))
    expected = int(width) * int(height)
    if len(raw) != expected:
        raise ValueError(f"mask byte length {len(raw)} != expected {expected}")
    return bytes(1 if value else 0 for value in raw)


def depth_values_mm(depth: bytes, *, width: int, height: int, bigendian: bool = False) -> array:
    expected = int(width) * int(height) * 2
    if len(depth) != expected:
        raise ValueError(f"depth byte length {len(depth)} != expected {expected}")
    values = array("H")
    values.frombytes(depth)
    if bigendian:
        values.byteswap()
    return values


def masked_depth_median_m(
    depth: bytes,
    mask: bytes,
    *,
    width: int,
    height: int,
    min_depth_m: float = 0.05,
    max_depth_m: float = 20.0,
) -> float | None:
    values = depth_values_mm(depth, width=width, height=height)
    if len(mask) != len(values):
        raise ValueError("mask and depth dimensions differ")
    low, high = int(min_depth_m * 1000.0), int(max_depth_m * 1000.0)
    selected = [value for value, enabled in zip(values, mask) if enabled and low <= value <= high]
    if not selected:
        return None
    return float(statistics.median(selected)) / 1000.0


def mask_centroid(mask: bytes, *, width: int, height: int) -> tuple[float, float] | None:
    xs: list[int] = []
    ys: list[int] = []
    for index, enabled in enumerate(mask):
        if enabled:
            xs.append(index % int(width))
            ys.append(index // int(width))
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def free_space_sectors_m(
    depth: bytes,
    *,
    width: int,
    height: int,
    vertical_start: float = 0.35,
    vertical_end: float = 0.85,
    quantile: float = 0.20,
) -> dict[str, float | None]:
    values = depth_values_mm(depth, width=width, height=height)
    y0 = max(0, min(int(height) - 1, int(float(vertical_start) * int(height))))
    y1 = max(y0 + 1, min(int(height), int(float(vertical_end) * int(height))))
    names = ("left", "front", "right")
    result: dict[str, float | None] = {}
    for sector, name in enumerate(names):
        x0 = sector * int(width) // 3
        x1 = (sector + 1) * int(width) // 3
        samples: list[int] = []
        for y in range(y0, y1):
            row = y * int(width)
            samples.extend(value for value in values[row + x0 : row + x1] if 50 <= value <= 20000)
        if not samples:
            result[name] = None
            continue
        samples.sort()
        index = max(0, min(len(samples) - 1, int((len(samples) - 1) * float(quantile))))
        result[name] = round(samples[index] / 1000.0, 4)
    return result


def build_target_observation(
    *,
    episode_id: str,
    target_id: str,
    frame_seq: int,
    source_stamp_sec: float,
    width: int,
    height: int,
    intrinsics: dict[str, Any],
    depth: bytes,
    response: dict[str, Any],
    model_latency_sec: float,
    horizontal_flip: bool = False,
) -> tuple[dict[str, Any], bytes]:
    detections = response.get("detections") if isinstance(response.get("detections"), list) else []
    valid = [row for row in detections if isinstance(row, dict)]
    empty_mask = bytes(int(width) * int(height))
    free_space = free_space_sectors_m(depth, width=width, height=height)
    if horizontal_flip:
        free_space = {
            "left": free_space.get("right"),
            "front": free_space.get("front"),
            "right": free_space.get("left"),
        }
    common = {
        "episode_id": str(episode_id),
        "target_id": str(target_id).strip().lower(),
        "frame_seq": int(frame_seq),
        "source_stamp_sec": float(source_stamp_sec),
        "image_snapshot": {
            "frame_seq": int(frame_seq),
            "width": int(width),
            "height": int(height),
            "encoding": "rgb8",
            "horizontal_flip": bool(horizontal_flip),
            "source": "actual_isaac_viewport",
        },
        "model": dict(response.get("model") or {}),
        "perception_latency_sec": round(float(model_latency_sec), 6),
        "free_space_m": free_space,
        "bearing_horizontal_flip_applied": bool(horizontal_flip),
        "source": "groundingdino_b_sam2_1_hiera_large",
        "temporal_propagation": dict(response.get("temporal_propagation") or {}),
    }
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    candidates: list[dict[str, Any]] = []
    masks: dict[int, bytes] = {}
    for index, detection in enumerate(valid):
        mask = decode_mask_zlib(detection, width=width, height=height)
        masks[index] = mask
        centroid = mask_centroid(mask, width=width, height=height)
        distance = masked_depth_median_m(depth, mask, width=width, height=height)
        candidate: dict[str, Any] = {
            "candidate_index": index,
            "visible": False,
            "confidence": round(float(detection.get("score", 0.0)), 6),
            "label": str(detection.get("label") or ""),
            "candidate_source": str(detection.get("candidate_source") or "groundingdino_sam2_image"),
            "bbox_xyxy_norm": [float(value) for value in detection.get("bbox_xyxy_norm", [])],
            "mask_sha256": hashlib.sha256(mask).hexdigest(),
            "mask_pixel_fraction": round(sum(mask) / max(1, len(mask)), 6),
            "visual_attribute_evidence": dict(detection.get("visual_attribute_evidence") or {}),
        }
        attribute_evidence = candidate["visual_attribute_evidence"]
        attribute_ok = not attribute_evidence or bool(attribute_evidence.get("exists", False))
        if centroid is not None and distance is not None and attribute_ok:
            u, v = centroid
            bearing_u = (int(width) - 1.0 - u) if horizontal_flip else u
            x_right = (bearing_u - cx) * distance / fx
            y_down = (v - cy) * distance / fy
            bearing = math.atan2(-x_right, distance)
            candidate.update(
                {
                    "visible": True,
                    "centroid_uv": [round(u, 3), round(v, 3)],
                    "distance_m": round(distance, 4),
                    "bearing_rad": round(bearing, 6),
                    "point_camera_xyz_m": [round(x_right, 4), round(y_down, 4), round(distance, 4)],
                    "point_base_xy_m": [round(distance, 4), round(-x_right, 4)],
                }
            )
        elif centroid is not None and distance is not None and not attribute_ok:
            candidate["reason"] = "required_visual_attribute_absent"
        candidates.append(candidate)
    geometric = [candidate for candidate in candidates if candidate.get("visible")]
    best = max(geometric, key=lambda row: float(row.get("confidence", 0.0)), default=None)
    if best is None:
        confidence = max((float(row.get("confidence", 0.0)) for row in candidates), default=0.0)
        return common | {
            "visible": False,
            "confidence": confidence,
            "reason": "no_valid_mask_depth" if candidates else "no_detection",
            "candidates": candidates,
        }, empty_mask
    best_index = int(best["candidate_index"])
    return common | dict(best) | {"candidates": candidates}, masks[best_index]


def frame_stamps_synchronized(rgb_stamp: float, depth_stamp: float, *, tolerance_sec: float = 0.002) -> bool:
    return abs(float(rgb_stamp) - float(depth_stamp)) <= abs(float(tolerance_sec))


def result_matches_active_context(
    result: dict[str, Any],
    *,
    episode_id: str,
    target_id: str,
) -> bool:
    return (
        str(result.get("episode_id") or "") == str(episode_id or "")
        and str(result.get("target_id") or "").strip().lower() == str(target_id or "").strip().lower()
    )


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(quantile)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
