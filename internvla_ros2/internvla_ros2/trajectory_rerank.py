"""Deterministic non-oracle reranking of InternVLA diffusion trajectories.

Only the current normalized depth image and candidate-local XY geometry are
used.  Episode success, goal pose, global map, simulator truth and navigation
metrics are deliberately absent from this module's interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any

import numpy as np

from .trajectory_capture import CapturedTrajectoryCandidates


@dataclass(frozen=True)
class TrajectoryRerankConfig:
    expected_candidate_count: int = 32
    depth_max_m: float = 6.0
    depth_hfov_deg: float = 87.0
    footprint_radius_m: float = 0.30
    depth_quantile: float = 0.20
    vertical_band_start: float = 0.42
    vertical_band_end: float = 0.90
    sector_half_width_px: int = 8
    maximum_scored_horizon_m: float = 2.0
    progress_weight: float = 1.0
    clearance_weight: float = 1.5
    collision_penalty_weight: float = 8.0
    smoothness_weight: float = 0.20
    out_of_view_weight: float = 0.50
    minimum_valid_depth_fraction: float = 0.02

    def validate(self) -> None:
        if self.expected_candidate_count != 32:
            raise ValueError("InternVLA rerank contract requires exactly 32 candidates")
        numeric = (
            self.depth_max_m,
            self.depth_hfov_deg,
            self.footprint_radius_m,
            self.maximum_scored_horizon_m,
            self.progress_weight,
            self.clearance_weight,
            self.collision_penalty_weight,
            self.smoothness_weight,
            self.out_of_view_weight,
            self.minimum_valid_depth_fraction,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in numeric):
            raise ValueError("trajectory rerank bounds must be finite and positive")
        if not 0.0 < self.depth_quantile < 1.0:
            raise ValueError("depth_quantile must be inside (0,1)")
        if not 0.0 <= self.vertical_band_start < self.vertical_band_end <= 1.0:
            raise ValueError("invalid vertical depth band")
        if self.sector_half_width_px < 1:
            raise ValueError("sector_half_width_px must be positive")


@dataclass(frozen=True)
class TrajectoryRerankResult:
    selected_trajectory: np.ndarray | None
    selected_index: int | None
    scores: tuple[float | None, ...]
    upstream_shape: tuple[int, ...] | None
    fallback_reason: str | None
    valid_candidate_count: int
    depth_valid_fraction: float
    candidates_sha256: str | None
    depth_sha256: str

    def audit_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": "depth_geometry_non_oracle_v1",
            "oracle_or_ground_truth_inputs": False,
            "selected_index": self.selected_index,
            "scores": list(self.scores),
            "upstream_shape": (
                None if self.upstream_shape is None else list(self.upstream_shape)
            ),
            "fallback_reason": self.fallback_reason,
            "valid_candidate_count": self.valid_candidate_count,
            "depth_valid_fraction": self.depth_valid_fraction,
            "candidates_sha256": self.candidates_sha256,
            "depth_sha256": self.depth_sha256,
        }


def _digest_float32(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _fallback(
    reason: str,
    *,
    candidates: np.ndarray | None,
    depth: np.ndarray,
    valid_fraction: float,
) -> TrajectoryRerankResult:
    return TrajectoryRerankResult(
        selected_trajectory=None,
        selected_index=None,
        scores=(),
        upstream_shape=None if candidates is None else tuple(candidates.shape),
        fallback_reason=reason,
        valid_candidate_count=0,
        depth_valid_fraction=float(valid_fraction),
        candidates_sha256=(
            None if candidates is None else _digest_float32(candidates)
        ),
        depth_sha256=_digest_float32(depth),
    )


def rerank_trajectories(
    candidate_trajectories: Any,
    normalized_depth: Any,
    config: TrajectoryRerankConfig | None = None,
) -> TrajectoryRerankResult:
    """Select one of 32 local trajectories with deterministic depth geometry.

    Malformed/missing upstream candidate tensors produce a recorded fallback;
    callers retain the unmodified upstream mean trajectory in that case.
    """

    cfg = config or TrajectoryRerankConfig()
    cfg.validate()
    depth = np.asarray(normalized_depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2 or depth.size == 0 or not np.isfinite(depth).all():
        safe_depth = np.zeros((1, 1), dtype=np.float32)
        return _fallback(
            "invalid_depth_shape_or_values",
            candidates=None,
            depth=safe_depth,
            valid_fraction=0.0,
        )
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    if float(depth.min()) < 0.0 or float(depth.max()) > 1.0:
        return _fallback(
            "depth_outside_normalized_range",
            candidates=None,
            depth=depth,
            valid_fraction=0.0,
        )
    valid_depth = depth > 0.0
    valid_fraction = float(np.mean(valid_depth))
    raw_shape: tuple[int, ...] | None = None
    if isinstance(candidate_trajectories, CapturedTrajectoryCandidates):
        raw_shape = tuple(candidate_trajectories.raw_shape)
        candidate_trajectories = candidate_trajectories.trajectories
    candidates = (
        None
        if candidate_trajectories is None
        else np.asarray(candidate_trajectories, dtype=np.float32)
    )
    if candidates is None:
        return _fallback(
            "upstream_candidate_tensor_missing",
            candidates=None,
            depth=depth,
            valid_fraction=valid_fraction,
        )
    if raw_shape is not None and raw_shape != (32, 32, 3):
        return _fallback(
            "upstream_raw_diffusion_tensor_is_not_32_by_32_by_3",
            candidates=candidates,
            depth=depth,
            valid_fraction=valid_fraction,
        )
    if (
        candidates.ndim != 3
        or candidates.shape[0] != cfg.expected_candidate_count
        or candidates.shape[1] != 33
        or candidates.shape[2] != 2
    ):
        return _fallback(
            "upstream_candidate_tensor_is_not_32_by_33_by_2",
            candidates=candidates,
            depth=depth,
            valid_fraction=valid_fraction,
        )
    if valid_fraction < cfg.minimum_valid_depth_fraction:
        return _fallback(
            "insufficient_valid_depth",
            candidates=candidates,
            depth=depth,
            valid_fraction=valid_fraction,
        )

    height, width = depth.shape
    row_start = min(height - 1, int(round(height * cfg.vertical_band_start)))
    row_end = max(row_start + 1, int(round(height * cfg.vertical_band_end)))
    row_end = min(height, row_end)
    band = depth[row_start:row_end] * np.float32(cfg.depth_max_m)
    half_fov = math.radians(cfg.depth_hfov_deg) / 2.0
    scores: list[float | None] = []
    valid_indices: list[int] = []

    for index, path in enumerate(candidates):
        if not np.isfinite(path).all():
            scores.append(None)
            continue
        segments = np.diff(path, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        if not np.isfinite(lengths).all() or float(np.sum(lengths)) <= 1e-6:
            scores.append(None)
            continue
        points = path[1:]
        ranges = np.linalg.norm(points, axis=1)
        bearings = np.arctan2(points[:, 1], points[:, 0])
        margins: list[float] = []
        outside = 0
        for radial, bearing, point in zip(ranges, bearings, points):
            if float(point[0]) <= 0.02 or abs(float(bearing)) > half_fov:
                outside += 1
                continue
            normalized_column = 0.5 + float(bearing) / (2.0 * half_fov)
            column = int(round(normalized_column * (width - 1)))
            left = max(0, column - cfg.sector_half_width_px)
            right = min(width, column + cfg.sector_half_width_px + 1)
            samples = band[:, left:right]
            samples = samples[samples > 0.0]
            if samples.size == 0:
                outside += 1
                continue
            clearance = float(np.quantile(samples, cfg.depth_quantile))
            margins.append(clearance - float(radial) - cfg.footprint_radius_m)
        if not margins:
            scores.append(None)
            continue
        minimum_margin = min(margins)
        collision_deficit = max(0.0, -minimum_margin)
        endpoint_progress = min(
            float(np.linalg.norm(path[-1])), cfg.maximum_scored_horizon_m
        )
        if len(segments) >= 2:
            unit = segments / np.maximum(lengths[:, None], 1e-6)
            smoothness = float(np.mean(np.linalg.norm(np.diff(unit, axis=0), axis=1)))
        else:
            smoothness = 0.0
        outside_fraction = outside / max(1, len(points))
        score = (
            cfg.progress_weight * endpoint_progress
            + cfg.clearance_weight * float(np.clip(minimum_margin, -1.0, 1.0))
            - cfg.collision_penalty_weight * collision_deficit
            - cfg.smoothness_weight * smoothness
            - cfg.out_of_view_weight * outside_fraction
        )
        scores.append(float(score))
        valid_indices.append(index)

    if not valid_indices:
        return _fallback(
            "no_geometrically_valid_candidate",
            candidates=candidates,
            depth=depth,
            valid_fraction=valid_fraction,
        )
    # Stable index tie-break keeps the fallback/replay behavior deterministic.
    selected_index = max(
        valid_indices,
        key=lambda index: (float(scores[index]), -index),
    )
    return TrajectoryRerankResult(
        selected_trajectory=np.ascontiguousarray(
            candidates[selected_index], dtype=np.float32
        ),
        selected_index=selected_index,
        scores=tuple(scores),
        upstream_shape=tuple(candidates.shape),
        fallback_reason=None,
        valid_candidate_count=len(valid_indices),
        depth_valid_fraction=valid_fraction,
        candidates_sha256=_digest_float32(candidates),
        depth_sha256=_digest_float32(depth),
    )


def config_mapping(config: TrajectoryRerankConfig | None = None) -> dict[str, Any]:
    value = config or TrajectoryRerankConfig()
    value.validate()
    return asdict(value)
